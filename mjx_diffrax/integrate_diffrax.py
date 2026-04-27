"""Diffrax ODE integration for MJX."""

from typing import Tuple, Type

import diffrax as dfx
import equinox as eqx
import equinox.internal as eqxi
import jax
import mujoco
import mujoco.mjx as mjx
import optimistix as optx
from jax import numpy as jp
from mujoco.mjx._src.types import PyTreeNode

from mjx_diffrax.config import DiffraxConfig
from mjx_diffrax.util import (
    apply_dexpinv_angular,
    clip_act,
    normalize_quaternions,
    qpos_dot_func,
    qpos_dot_linear_only,
    qvel_angular_only,
    reconstruct_quat_positions,
)

Model = mjx.Model
Data = mjx.Data


class SystemState(PyTreeNode):
    qpos0: jax.Array
    qvel0: jax.Array
    qacc0: jax.Array
    act0: jax.Array
    t0: jax.Array
    qpos: jax.Array
    qvel: jax.Array
    act: jax.Array
    omega_integral: jax.Array  # time-integrated angular velocity for exact quaternion reconstruction


def rms_norm_exact_quat_integration(y: SystemState) -> jax.Array:
    return optx.rms_norm((y.qpos, y.omega_integral, y.qvel, y.act))


def rms_norm_ode(y: SystemState) -> jax.Array:
    return optx.rms_norm((y.qpos, y.qvel, y.act))


def diffrax_integrate(m: Model, d: Data, cfg: DiffraxConfig) -> Data:
    ctrls = d.ctrl[None]
    qfrcs_applied = d.qfrc_applied[None]
    d1, _ = diffrax_integrate_multistep(
        m, d, nsteps=1, ctrls=ctrls, qfrcs_applied=qfrcs_applied, cfg=cfg
    )
    return d1


def diffrax_integrate_multistep(
    m: Model,
    d: Data,
    nsteps: int,
    ctrls: jax.Array,
    qfrcs_applied: jax.Array,
    cfg: DiffraxConfig,
) -> Tuple[Data, Data]:
    t0 = jax.lax.stop_gradient(d.time)

    t1 = t0 + nsteps * m.opt.timestep
    ts = jp.linspace(t0, t1, nsteps + 1)

    # hold last ctrl/qfrc constant over final step
    ctrls = jp.concatenate([ctrls, ctrls[-1][None]], axis=0)
    qfrcs_applied = jp.concatenate([qfrcs_applied, qfrcs_applied[-1][None]], axis=0)

    return _diffrax_integrate(m, d, ts, ctrls, qfrcs_applied, cfg)


def _diffrax_integrate(
    m: Model,
    d: Data,
    ts: jax.Array,
    ctrls: jax.Array,
    qfrcs_applied: jax.Array,
    cfg: DiffraxConfig,
) -> Tuple[Data, Data]:
    ts = jax.lax.stop_gradient(ts)
    ts = eqxi.nondifferentiable(ts)
    t0, t1 = ts[0], ts[-1]

    y0 = SystemState(
        qpos0=d.qpos,
        qvel0=d.qvel,
        qacc0=d.qacc,
        act0=d.act,
        t0=t0,
        qpos=d.qpos,
        qvel=d.qvel,
        act=d.act,
        omega_integral=jp.zeros_like(d.qvel),
    )
    solver, stepsize_controller, adjoint = get_diffrax_args(m, cfg)

    if isinstance(stepsize_controller, dfx.ConstantStepSize):
        dt0 = m.opt.timestep
    elif cfg.pid_dt0 is None:
        dt0 = m.opt.timestep
    else:
        dt0 = cfg.pid_dt0

    solution = dfx.diffeqsolve(
        terms=dfx.ODETerm(mjx_ode_fn),
        y0=y0,
        t0=t0,
        t1=t1,
        dt0=dt0,
        max_steps=cfg.max_ode_steps,
        solver=solver,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        args=(m, d, ts, ctrls, qfrcs_applied, cfg),
        saveat=dfx.SaveAt(ts=ts[:-1], t1=True),
    )
    ys = solution.ys
    if not cfg.exact_quat_integration:
        ys = eqx.filter_vmap(
            lambda y: y.replace(
                qpos=normalize_quaternions(m, y.qpos),
                act=clip_act(m, y.act),
            )
        )(ys)
    d_replace_fn = lambda y: mjx.kinematics(
        m, d.replace(qpos=y.qpos, qvel=y.qvel, act=y.act)
    )
    ds = eqx.filter_vmap(d_replace_fn)(ys)
    d1 = jax.tree.map(lambda x: x[-1], ds)
    return d1, ds


def mjx_ode_fn(t, y, args):
    m, d, ts, ctrls, qfrcs_applied, cfg = args

    theta_i = y.omega_integral  # RKMK: Theta_i before stage correction resets it
    y = apply_stage_correction(m=m, d=d, y=y, t=t, cfg=cfg)

    ctrl = dfx.LinearInterpolation(ts, ctrls).evaluate(t)
    qfrc = dfx.LinearInterpolation(ts, qfrcs_applied).evaluate(t)

    d = d.replace(
        qpos=y.qpos,
        qvel=y.qvel,
        qacc=y.qacc0,
        act=y.act,
        time=y.t0,
        ctrl=ctrl,
        qfrc_applied=qfrc,
    )
    d = mjx.forward(m, d)

    # constant across RK stages
    dqpos0 = jp.zeros_like(y.qpos0)
    dqvel0 = jp.zeros_like(y.qvel0)
    dqacc0 = jp.zeros_like(y.qacc0)
    dact0 = jp.zeros_like(y.act0)
    dt0 = jp.zeros_like(y.t0)

    dqvel = d.qacc
    dact = d.act_dot

    if cfg.exact_quat_integration:
        dqpos = qpos_dot_linear_only(m, d.qpos, d.qvel)
        omega = qvel_angular_only(m, d.qvel)
        d_qpos = apply_dexpinv_angular(m, theta_i, omega) if cfg.rkmk_dexpinv else omega
    else:
        dqpos = qpos_dot_func(m, d.qpos, d.qvel)
        d_qpos = jp.zeros_like(y.omega_integral)

    return SystemState(
        qpos0=dqpos0,
        qvel0=dqvel0,
        qacc0=dqacc0,
        act0=dact0,
        t0=dt0,
        qpos=dqpos,
        qvel=dqvel,
        act=dact,
        omega_integral=d_qpos,
    )


def get_diffrax_args(m, cfg):
    solver_classes = {
        "Tsit5": dfx.Tsit5,
        "Dopri5": dfx.Dopri5,
        "Dopri8": dfx.Dopri8,
        "Euler": dfx.Euler,
        "Heun": dfx.Heun,
        "Midpoint": dfx.Midpoint,
        "Ralston": dfx.Ralston,
        "Bosh3": dfx.Bosh3,
    }
    if cfg.solver not in solver_classes:
        raise ValueError(f"Invalid solver: {cfg.solver}")
    
    if cfg.stepsize_controller == "Constant":
        stepsize_controller = dfx.ConstantStepSize()
    elif cfg.stepsize_controller == "PID":
        norm = rms_norm_exact_quat_integration if cfg.exact_quat_integration else rms_norm_ode
        stepsize_controller = dfx.PIDController(
            rtol=cfg.pid_rtol,
            atol=cfg.pid_atol,
            norm=norm,
            pcoeff=0.2,
            icoeff=0.4,
            dcoeff=0.0,
        )
    else:
        raise ValueError(f"Invalid stepsize controller: {cfg.stepsize_controller}")

    if cfg.adjoint == "RecursiveCheckpoint":
        solver = solver_factory(solver_classes[cfg.solver])()
        adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints=cfg.recursive_ncheckpoints)
    elif cfg.adjoint == "Backsolve":
        if cfg.exact_quat_integration:
           raise NotImplementedError(
               'Backsolve not implemented for exact integration.')
        
        solver = solver_classes[cfg.solver]()

        if cfg.stepsize_controller == "Constant":
            adjoint_stepsize_controller = dfx.ConstantStepSize()
        elif cfg.stepsize_controller == "PID":
            adjoint_stepsize_controller = dfx.PIDController(
                rtol=cfg.pid_rtol,
                atol=cfg.pid_atol,
                pcoeff=0.2,
                icoeff=0.4,
                dcoeff=0.0,
                norm=dfx.adjoint_rms_seminorm,  # important to ignore model nan gradients in stepsize selection
                )
        else:
            raise ValueError(f"Invalid stepsize controller for adjoint: {cfg.stepsize_controller}")
        adjoint = dfx.BacksolveAdjoint(
            stepsize_controller=adjoint_stepsize_controller, solver=solver
            )
    else:
        raise ValueError(f"Invalid adjoint: {cfg.adjoint}")

    return solver, stepsize_controller, adjoint


def apply_stage_correction(m, d, y, t, cfg):
    dt = t - y.t0
    qacc = (y.qvel - y.qvel0) / jp.maximum(dt, mujoco.mjMINVAL)

    if cfg.exact_quat_integration:
        omega = y.omega_integral / jp.maximum(dt, mujoco.mjMINVAL)
        qpos = reconstruct_quat_positions(m, y.qpos0, y.qpos, omega, dt)
        y = y.replace(qpos=qpos, act=clip_act(m, y.act), omega_integral=jp.zeros_like(y.omega_integral))
    else:
        y = y.replace(
            qpos=normalize_quaternions(m, y.qpos),
            act=clip_act(m, y.act),
        )

    y = y.replace(qpos0=y.qpos, qvel0=y.qvel, qacc0=qacc, act0=y.act, t0=t)
    return y


class _CorrectedInterpolationWrapper(dfx.AbstractLocalInterpolation):
    """Wraps any local interpolation to apply stage correction on evaluate."""

    inner: dfx.AbstractLocalInterpolation
    m: Model
    d: Data
    cfg: DiffraxConfig

    @property
    def t0(self):
        return self.inner.t0

    @property
    def t1(self):
        return self.inner.t1

    def evaluate(self, t0, t1=None, left=True):
        y1 = self.inner.evaluate(t0, t1, left)
        y1 = apply_stage_correction(m=self.m, d=self.d, y=y1, t=t0, cfg=self.cfg)
        return y1


def solver_factory(solver_class: Type[dfx.AbstractSolver]) -> Type[dfx.AbstractSolver]:
    base_interp_cls = solver_class.interpolation_cls

    if isinstance(base_interp_cls, type):
        # Class-based interpolation (Euler, Tsit5, Dopri5, Dopri8): subclass directly
        class CorrectedInterpolationCls(base_interp_cls):
            m: Model
            d: Data
            cfg: DiffraxConfig

            def __init__(self, m, d, cfg, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.m = m
                self.d = d
                self.cfg = cfg

            def evaluate(self, t0, t1=None, left=True):
                y1 = super().evaluate(t0, t1, left)
                y1 = apply_stage_correction(m=self.m, d=self.d, y=y1, t=t0, cfg=self.cfg)
                return y1

    else:
        # Factory-method interpolation (Heun, Midpoint, Ralston, Bosh3):
        # from_k classmethod can't be subclassed, so wrap the result instead.
        original_factory = base_interp_cls

        def CorrectedInterpolationCls(*, m, d, cfg, **kwargs):
            inner = original_factory(**kwargs)
            return _CorrectedInterpolationWrapper(inner=inner, m=m, d=d, cfg=cfg)

    # When interpolation_cls is a function (not a class), it must be wrapped with
    # staticmethod to prevent Python from binding it as an instance method.
    interp_attr = (
        CorrectedInterpolationCls
        if isinstance(base_interp_cls, type)
        else staticmethod(CorrectedInterpolationCls)
    )

    class CorrectedSolver(solver_class, dfx.AbstractSolver):
        interpolation_cls = interp_attr

        def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
            y1, error_estimate, dense_info, new_solver_state, results = super().step(
                terms, t0, t1, y0, args, solver_state, made_jump
            )
            if not isinstance(y1, SystemState):
                raise ValueError(f"Unsupported y1 type: {type(y1)}")
            m, d, _, _, _, cfg = args
            y1 = apply_stage_correction(m=m, d=d, y=y1, t=t1, cfg=cfg)
            dense_info["y1"] = y1
            dense_info["m"] = m
            dense_info["d"] = d
            dense_info["cfg"] = cfg
            # Invalidate FSAL: stage correction changed y1, so cached derivative is stale
            if new_solver_state is not None:
                new_solver_state = (True, new_solver_state[1])
            return y1, error_estimate, dense_info, new_solver_state, results

    return CorrectedSolver
