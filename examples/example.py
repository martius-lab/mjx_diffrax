"""Minimal example: simulate a bouncing ball with adaptive ODE integration.

Drops a ball onto a ground plane using adaptive Tsit5 integration via diffrax,
demonstrating single-step and multi-step rollouts.

Note: differentiating through contacts requires replacing MJX's while_loop
contact solver with scan_loop (see the diffmjx repo).
"""

import mujoco
import mujoco.mjx as mjx

import mjx_diffrax

# -- Model: ball on a ground plane --
XML = """
<mujoco>
  <option gravity="0 0 -2" timestep="0.01" solver="Newton"
          iterations="4" ls_iterations="10" tolerance="1e-8" cone="pyramidal">
    <flag warmstart="enable" eulerdamp="disable"/>
  </option>
  <default>
    <geom solimp="0.0 0.95 0.001 0.5 2" solref="0.005 1.0"/>
  </default>
  <worldbody>
    <geom type="plane" size="5 5 0.1" conaffinity="1" condim="3"/>
    <body pos="0 0 0.2">
      <freejoint/>
      <geom type="sphere" size="0.05" density="1"/>
    </body>
  </worldbody>
</mujoco>
"""

mj_model = mujoco.MjModel.from_xml_string(XML)
m = mjx.put_model(mj_model)
d = mjx.make_data(m)

# Give the ball a horizontal kick and a downward toss
d = d.replace(qvel=d.qvel.at[1].set(-0.1).at[2].set(-1.0))

# -- Configure adaptive integration --
cfg = mjx_diffrax.DiffraxConfig(
    solver="Tsit5",               # 5th-order adaptive Runge-Kutta
    stepsize_controller="PID",    # adaptive step-size control
    pid_rtol=1e-8,
    pid_atol=1e-8,
    max_ode_steps=4096,
)

# -- Single step (one MJX timestep, solved adaptively) --
d1 = mjx_diffrax.step(m, d, cfg)
print(f"After 1 step:      z={float(d1.qpos[2]):.6f}, vz={float(d1.qvel[2]):.6f}")

# -- Multi-step rollout (100 timesteps = 1 second) --
nsteps = int(1.0 / mj_model.opt.timestep)
d_final, d_traj = mjx_diffrax.multistep(m, d, nsteps=nsteps, cfg=cfg)
print(f"After {nsteps} steps:  z={float(d_final.qpos[2]):.6f}, y={float(d_final.qpos[1]):.6f}")
print(f"Trajectory shape:  qpos={d_traj.qpos.shape}, qvel={d_traj.qvel.shape}")
