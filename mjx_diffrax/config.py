from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class DiffraxConfig:
    solver: str = "Tsit5"
    stepsize_controller: str = "PID"
    max_ode_steps: int = 16
    exact_quat_integration: bool = True
    rkmk_dexpinv: bool = True
    pid_dt0: Optional[float] = None
    pid_rtol: float = 1e-3
    pid_atol: float = 1e-6
    adjoint: str = "RecursiveCheckpoint"
    recursive_ncheckpoints: Optional[int] = 16
    use_mjx: bool = None  # Use MJX's built-in integrators (defined via solver)
    mjx_timestep: bool = None  # Override timestep if MJX integrator is used (default is m.opt.timestep)