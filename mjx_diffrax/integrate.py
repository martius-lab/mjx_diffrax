from typing import Optional, Tuple, Union

import jax
import mujoco.mjx as mjx
from mujoco.mjx._src.types import IntegratorType
from jax import numpy as jp

from mjx_diffrax.config import DiffraxConfig
from mjx_diffrax.integrate_mjx import (
    mjx_integrate,
    mjx_integrate_multistep,
)
from mjx_diffrax.integrate_diffrax import (
    diffrax_integrate,
    diffrax_integrate_multistep,
)

def overwrite_mjx_options(m: mjx.Model, 
                          cfg: DiffraxConfig) -> mjx.Model:
    if cfg.solver is not None:
        if cfg.solver == "Euler":
            integrator = IntegratorType.EULER
        elif cfg.solver == "RK4":
            integrator = IntegratorType.RK4
        elif cfg.solver == "ImplicitFast":
            integrator = IntegratorType.IMPLICITFAST
        else:
            raise ValueError(f"Unsupported MJX solver: {cfg.solver}")
        
        m = m.tree_replace({'opt.integrator': integrator})

    if cfg.mjx_timestep is not None:
        m = m.tree_replace({'opt.timestep': cfg.mjx_timestep})
    
    return m   
         
def step(m: mjx.Model,
         d: mjx.Data,
         cfg: Union[None, DiffraxConfig]) -> mjx.Data:
    if cfg is None or cfg.use_mjx:
        if cfg is not None:
            m = overwrite_mjx_options(m, cfg)
        d = mjx_integrate(m, d)
    else:
        d = diffrax_integrate(m, d, cfg)

    return d


def multistep(
    m: mjx.Model,
    d: mjx.Data,
    nsteps: int,
    cfg: Union[None, DiffraxConfig],
    ctrls: Optional[jax.Array] = None,
    qfrcs_applied: Optional[jax.Array] = None,
    unroll_mjx_integrator: bool = False,
) -> Tuple[mjx.Data, mjx.Data]:
    if ctrls is None:
        ctrls = jp.tile(d.ctrl[None], (nsteps, 1))
    if qfrcs_applied is None:
        qfrcs_applied = jp.tile(d.qfrc_applied[None], (nsteps, 1))

    if cfg is None or cfg.use_mjx:
        m = overwrite_mjx_options(m, cfg)
        d1, ds = mjx_integrate_multistep(m, d, nsteps, ctrls, qfrcs_applied, unroll_mjx_integrator)
    else:
        d1, ds = diffrax_integrate_multistep(
        m, d, nsteps=nsteps, ctrls=ctrls, qfrcs_applied=qfrcs_applied, cfg=cfg)

    return d1, ds