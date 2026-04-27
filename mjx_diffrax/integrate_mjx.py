"""Wrapped MJX integrators."""

import jax
import jax.numpy as jp
import mujoco.mjx as mjx
from mujoco.mjx._src.types import IntegratorType
from mujoco.mjx._src.forward import euler, implicit, rungekutta4


def mjx_integrate(m: mjx.Model, d: mjx.Data) -> mjx.Data:  
    d = mjx.forward(m, d)

    if m.opt.integrator == IntegratorType.EULER:
        d = euler(m, d)
    elif m.opt.integrator == IntegratorType.RK4:
        d = rungekutta4(m, d)
    elif m.opt.integrator == IntegratorType.IMPLICITFAST:
        d = implicit(m, d)
    else:
        raise ValueError(f"Unsupported integrator: {m.opt.integrator}")
    
    return d


def mjx_integrate_multistep(m: mjx.Model, d: mjx.Data, nsteps: int, ctrls: jax.Array, qfrcs_applied: jax.Array, unroll_mjx_integrator: bool = False):
    
    def body(_d, y):
        ctrl, qfrc = y
        _d = _d.replace(ctrl=ctrl, qfrc_applied=qfrc)
        _d = mjx_integrate(m, _d)
        return _d, _d

    if unroll_mjx_integrator:
        d_initial = d
        ds_list = []
        for i in range(nsteps):
            d, _ = body(d, (ctrls[i], qfrcs_applied[i]))
            ds_list.append(d)
        ds = jax.tree.map(lambda x0, *xs: jp.stack([x0] + list(xs)), d_initial, *ds_list)
        d1 = d
    else:
        d1, ds = jax.lax.scan(body, d, (ctrls, qfrcs_applied), length=nsteps)
        ds = jax.tree.map(lambda x0, xs: jp.concatenate([x0[None], xs]), d, ds)
    
    return d1, ds