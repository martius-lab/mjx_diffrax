"""Test suite for mjx_diffrax solvers."""

import jax
import jax.numpy as jp
import mujoco
import mujoco.mjx as mjx
import pytest

from mjx_diffrax import DiffraxConfig, multistep, step

XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

GRAVITY = 9.81
Z0 = 1.0


@pytest.fixture(scope="module")
def mjx_model_data():
    mj_model = mujoco.MjModel.from_xml_string(XML)
    m = mjx.put_model(mj_model)
    d = mjx.make_data(m)
    return m, d


# ---------------------------------------------------------------------------
# Solver parametrization helpers
# ---------------------------------------------------------------------------

FIXED_SOLVERS = ["Euler", "Heun", "Midpoint", "Ralston"]
ADAPTIVE_SOLVERS = ["Tsit5", "Dopri5", "Dopri8", "Bosh3"]
ALL_SOLVERS = FIXED_SOLVERS + ADAPTIVE_SOLVERS

SOLVER_PARAMS = [(s, "Constant", None) for s in FIXED_SOLVERS] + [
    (s, "PID", 16) for s in ADAPTIVE_SOLVERS
]

ALL_SOLVER_PARAMS = SOLVER_PARAMS + [
    # Adaptive solvers can also run with Constant stepsize
    (s, "Constant", None) for s in ADAPTIVE_SOLVERS
]


def _make_cfg(solver, controller, max_steps, **kwargs):
    return DiffraxConfig(
        solver=solver, stepsize_controller=controller, max_ode_steps=max_steps, **kwargs
    )


# ---------------------------------------------------------------------------
# 1. Forward integration: every solver × step mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver,controller,max_steps",
    ALL_SOLVER_PARAMS,
    ids=[f"{p[0]}-{p[1]}" for p in ALL_SOLVER_PARAMS],
)
class TestForwardIntegration:
    def test_step(self, mjx_model_data, solver, controller, max_steps):
        m, d = mjx_model_data
        d1 = step(m, d, _make_cfg(solver, controller, max_steps))

        assert jp.all(jp.isfinite(d1.qpos)), "qpos contains NaN/Inf"
        assert jp.all(jp.isfinite(d1.qvel)), "qvel contains NaN/Inf"
        assert d1.qvel[2] < 0, "vz should be negative under gravity"

    def test_multistep(self, mjx_model_data, solver, controller, max_steps):
        m, d = mjx_model_data
        nsteps = 5
        d1, ds = multistep(m, d, nsteps=nsteps, cfg=_make_cfg(solver, controller, max_steps))

        assert jp.all(jp.isfinite(d1.qpos)), "final qpos contains NaN/Inf"
        assert jp.all(jp.isfinite(d1.qvel)), "final qvel contains NaN/Inf"
        assert d1.qvel[2] < 0, "final vz should be negative under gravity"
        # Trajectory has correct batch dimension
        assert ds.qpos.shape[0] == nsteps + 1


# ---------------------------------------------------------------------------
# 2. Quantitative accuracy against analytical free-fall
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver,controller,max_steps,nsteps,atol",
    [
        # High-order adaptive: should match analytical solution closely
        ("Tsit5", "PID", 16, 10, 1e-5),
        ("Dopri5", "PID", 16, 10, 1e-5),
        ("Dopri8", "PID", 16, 10, 1e-6),
        ("Bosh3", "PID", 16, 10, 1e-4),
        # Fixed-step: larger error budget (O(h) for Euler, O(h^2) for others)
        ("Euler", "Constant", None, 10, 5e-2),
        ("Heun", "Constant", None, 10, 5e-3),
        ("Midpoint", "Constant", None, 10, 5e-3),
        ("Ralston", "Constant", None, 10, 5e-3),
    ],
    ids=lambda p: p if isinstance(p, str) else None,
)
def test_freefall_accuracy(mjx_model_data, solver, controller, max_steps, nsteps, atol):
    """Check integrated trajectory against analytical free-fall: z = z0 - g*t^2/2."""
    m, d = mjx_model_data
    dt = float(m.opt.timestep)
    cfg = _make_cfg(solver, controller, max_steps)

    d1, ds = multistep(m, d, nsteps=nsteps, cfg=cfg)

    t_final = nsteps * dt
    z_analytical = Z0 - 0.5 * GRAVITY * t_final**2
    vz_analytical = -GRAVITY * t_final

    assert abs(float(d1.qpos[2]) - z_analytical) < atol, (
        f"z error: got {float(d1.qpos[2]):.8f}, expected {z_analytical:.8f}"
    )
    assert abs(float(d1.qvel[2]) - vz_analytical) < atol, (
        f"vz error: got {float(d1.qvel[2]):.8f}, expected {vz_analytical:.8f}"
    )

    # Horizontal position and quaternion should be unchanged
    assert jp.allclose(d1.qpos[:2], d.qpos[:2], atol=atol), "x/y drifted"
    assert jp.allclose(d1.qpos[3:7], d.qpos[3:7], atol=atol), "quaternion drifted"


# ---------------------------------------------------------------------------
# 3. Quaternion integration modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exact_quat,rkmk",
    [(True, True), (True, False), (False, False)],
    ids=["exact+rkmk", "exact-no-rkmk", "r4-ode"],
)
def test_quat_integration_modes(mjx_model_data, exact_quat, rkmk):
    m, d = mjx_model_data
    cfg = DiffraxConfig(
        solver="Tsit5",
        stepsize_controller="PID",
        max_ode_steps=16,
        exact_quat_integration=exact_quat,
        rkmk_dexpinv=rkmk,
    )
    d1 = step(m, d, cfg)

    assert jp.all(jp.isfinite(d1.qpos)), "qpos contains NaN/Inf"
    assert jp.all(jp.isfinite(d1.qvel)), "qvel contains NaN/Inf"
    assert d1.qpos[2] < d.qpos[2], "z should decrease under gravity"
    # Quaternion should remain unit-norm
    quat = d1.qpos[3:7]
    assert abs(float(jp.linalg.norm(quat)) - 1.0) < 1e-5, "quaternion not unit-norm"


# ---------------------------------------------------------------------------
# 4. Gradient through step() — all solvers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver,controller,max_steps",
    SOLVER_PARAMS,
    ids=[p[0] for p in SOLVER_PARAMS],
)
def test_gradient_step(mjx_model_data, solver, controller, max_steps):
    m, d = mjx_model_data
    cfg = _make_cfg(solver, controller, max_steps)

    @jax.grad
    def grad_z(vz0):
        d0 = d.replace(qvel=d.qvel.at[2].set(vz0))
        d1 = step(m, d0, cfg)
        return d1.qpos[2]

    g = grad_z(jp.float32(0.0))
    assert jp.isfinite(g), f"{solver}: gradient is not finite"
    assert g != 0.0, f"{solver}: gradient is zero"
    # d(z1)/d(vz0) should be positive (more upward velocity → higher z)
    assert g > 0, f"{solver}: expected positive gradient, got {float(g)}"


# ---------------------------------------------------------------------------
# 5. Gradient through multistep()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver,controller,max_steps",
    [("Tsit5", "PID", 16), ("Euler", "Constant", None)],
    ids=["Tsit5-PID", "Euler-Constant"],
)
def test_gradient_multistep(mjx_model_data, solver, controller, max_steps):
    m, d = mjx_model_data
    cfg = _make_cfg(solver, controller, max_steps)

    @jax.grad
    def grad_z(vz0):
        d0 = d.replace(qvel=d.qvel.at[2].set(vz0))
        d1, _ = multistep(m, d0, nsteps=5, cfg=cfg)
        return d1.qpos[2]

    g = grad_z(jp.float32(0.0))
    assert jp.isfinite(g), f"{solver}: multistep gradient is not finite"
    assert g > 0, f"{solver}: expected positive multistep gradient, got {float(g)}"


# ---------------------------------------------------------------------------
# 6. Multistep with custom controls
# ---------------------------------------------------------------------------


def test_multistep_custom_ctrls(mjx_model_data):
    """Passing explicit ctrl/qfrc_applied arrays through multistep."""
    m, d = mjx_model_data
    nsteps = 5
    cfg = DiffraxConfig(solver="Tsit5", stepsize_controller="PID", max_ode_steps=16)

    # Explicit zero controls (nu=0 for this model, but shape must match)
    ctrls = jp.zeros((nsteps, m.nu))
    qfrcs = jp.zeros((nsteps, m.nv))

    d1, ds = multistep(m, d, nsteps=nsteps, cfg=cfg, ctrls=ctrls, qfrcs_applied=qfrcs)
    assert jp.all(jp.isfinite(d1.qpos)), "custom ctrls: qpos contains NaN/Inf"
    assert d1.qvel[2] < 0, "custom ctrls: vz should be negative"

    # Non-zero applied force: push ball upward
    qfrcs_up = jp.zeros((nsteps, m.nv))
    qfrcs_up = qfrcs_up.at[:, 2].set(20.0)  # upward force > gravity
    d1_up, _ = multistep(m, d, nsteps=nsteps, cfg=cfg, qfrcs_applied=qfrcs_up)
    assert jp.all(jp.isfinite(d1_up.qpos)), "qfrcs push: qpos contains NaN/Inf"
    # With upward force exceeding gravity, vz should be positive
    assert d1_up.qvel[2] > 0, (
        f"expected positive vz with upward force, got {float(d1_up.qvel[2])}"
    )


# ---------------------------------------------------------------------------
# 7. Multistep trajectory shape
# ---------------------------------------------------------------------------


def test_multistep_trajectory_shape(mjx_model_data):
    m, d = mjx_model_data
    nsteps = 7
    cfg = DiffraxConfig(solver="Tsit5", stepsize_controller="PID", max_ode_steps=16)
    d1, ds = multistep(m, d, nsteps=nsteps, cfg=cfg)

    assert ds.qpos.shape[0] == nsteps + 1, (
        f"expected trajectory length {nsteps + 1}, got {ds.qpos.shape[0]}"
    )
    assert ds.qvel.shape[0] == nsteps + 1
    # Trajectory z positions should monotonically decrease (free fall)
    zs = ds.qpos[:, 2]
    assert jp.all(zs[1:] <= zs[:-1]), "z trajectory should be monotonically decreasing"


# ---------------------------------------------------------------------------
# 8. Adjoint methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adjoint",
    [
        "RecursiveCheckpoint",
        pytest.param(
            "Backsolve",
            marks=pytest.mark.xfail(reason="BacksolveAdjoint equinox/diffrax compat issue"),
        ),
    ],
    ids=["recursive-checkpoint", "backsolve"],
)
def test_adjoint_methods(mjx_model_data, adjoint):
    m, d = mjx_model_data
    cfg = DiffraxConfig(
        solver="Tsit5", stepsize_controller="PID", max_ode_steps=16, adjoint=adjoint
    )

    @jax.grad
    def grad_z(vz0):
        d0 = d.replace(qvel=d.qvel.at[2].set(vz0))
        d1 = step(m, d0, cfg)
        return d1.qpos[2]

    g = grad_z(jp.float32(0.0))
    assert jp.isfinite(g), f"gradient with {adjoint} adjoint is not finite"
    assert g > 0, f"gradient with {adjoint} adjoint should be positive, got {float(g)}"
