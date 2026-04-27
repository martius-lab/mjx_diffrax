# mjx_diffrax

Adaptive time integration for [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) using [diffrax](https://github.com/patrick-kidger/diffrax).

Provides `step()` and `multistep()` that replace MJX's built-in Euler/RK4 integrators with adaptive ODE solvers (Tsit5, Dopri5, etc.).

## Install

Requires Python 3.10+. Install JAX first, then this package:

```bash
# CPU-only
uv pip install "jax[cpu]"
uv pip install git+ssh://git@github.com/martius-lab/mjx_diffrax.git

# CUDA 12
uv pip install "jax[cuda12]"
uv pip install git+ssh://git@github.com/martius-lab/mjx_diffrax.git
```

Or from a local clone:

```bash
git clone git@github.com:martius-lab/mjx_diffrax.git
cd mjx_diffrax
uv pip install -e .
```

## Usage

```python
import jax
import mujoco
import mujoco.mjx as mjx
import mjx_diffrax

mj_model = mujoco.MjModel.from_xml_path("model.xml")
m = mjx.put_model(mj_model)
d = mjx.make_data(m)

cfg = mjx_diffrax.DiffraxConfig(
    solver="Tsit5",
    stepsize_controller="PID",
    pid_rtol=1e-3,
    pid_atol=1e-6,
    max_ode_steps=16,
)

# Single step (replaces mjx.step)
d = mjx_diffrax.step(m, d, cfg)

# Multi-step with control interpolation
ctrls = jax.numpy.zeros((10, m.nu))
d_final, d_traj = mjx_diffrax.multistep(m, d, nsteps=10, cfg=cfg, ctrls=ctrls)

# Differentiable
grad_fn = jax.grad(lambda d: mjx_diffrax.step(m, d, cfg).qpos[2])
```

See `example.py` for a runnable script.

## Configuration

| Field | Default | Description |
|---|---|---|
| `solver` | `"Tsit5"` | `"Tsit5"`, `"Dopri5"`, `"Euler"`, `"Heun"` |
| `stepsize_controller` | `"PID"` | `"PID"` (adaptive) or `"Constant"` |
| `max_ode_steps` | `16` | Max ODE solver steps per MuJoCo timestep |
| `exact_quat_integration` | `True` | `True`: exponential map for quaternion joints. `False`: R^4 ODE + normalization. |
| `pid_dt0` | `None` | Initial step size for PID. `None` uses `m.opt.timestep`. |
| `pid_rtol` | `1e-3` | Relative tolerance |
| `pid_atol` | `1e-6` | Absolute tolerance |
| `adjoint` | `"RecursiveCheckpoint"` | Adjoint method for backpropagation |
| `recursive_ncheckpoints` | `16` | Checkpoints for recursive adjoint |

## Note on contact differentiation

Upstream MJX's contact solver uses `jax.lax.while_loop`, which does not support reverse-mode differentiation ([mujoco#2259](https://github.com/google-deepmind/mujoco/issues/2259)). Differentiating through contact-rich simulations requires a modified MJX solver (e.g. using `jax.lax.scan` with fixed iterations).
