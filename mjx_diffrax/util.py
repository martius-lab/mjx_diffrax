"""Reimplementation of MJX internal helpers that accept an explicit dt parameter."""

import jax
from jax import numpy as jp
from mujoco.mjx._src.types import JointType, Model


def normalize(x: jax.Array, axis=None) -> jax.Array:
    is_zero = jp.allclose(x, 0.0)
    x_safe = jp.where(is_zero, jp.ones_like(x), x)
    n = jp.linalg.norm(x_safe, axis=axis)
    n = jp.where(is_zero, 0.0, n)
    return x_safe / (n + 1e-6 * (n == 0.0))


def normalize_with_norm(x: jax.Array, axis=None):
    is_zero = jp.allclose(x, 0.0)
    x_safe = jp.where(is_zero, jp.ones_like(x), x)
    n = jp.linalg.norm(x_safe, axis=axis)
    n = jp.where(is_zero, 0.0, n)
    return x_safe / (n + 1e-6 * (n == 0.0)), n


def quat_mul(u: jax.Array, v: jax.Array) -> jax.Array:
    return jp.array(
        [
            u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
            u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
            u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
            u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
        ]
    )


def quat_integrate(q: jax.Array, v: jax.Array, dt: jax.Array) -> jax.Array:
    v_norm, norm_ = normalize_with_norm(v)
    angle = dt * norm_
    s, c = jp.sin(angle * 0.5), jp.cos(angle * 0.5)
    q_res = jp.insert(v_norm * s, 0, c)
    q_res = quat_mul(q, q_res)
    return normalize(q_res)


def normalize_quaternions(m: Model, qpos: jax.Array) -> jax.Array:
    qi = 0
    for jnt_typ in m.jnt_type:
        if jnt_typ == JointType.FREE:
            qpos = qpos.at[qi + 3 : qi + 7].set(normalize(qpos[qi + 3 : qi + 7]))
            qi = qi + 7
        elif jnt_typ == JointType.BALL:
            qpos = qpos.at[qi : qi + 4].set(normalize(qpos[qi : qi + 4]))
            qi = qi + 4
        elif jnt_typ in (JointType.HINGE, JointType.SLIDE):
            qi = qi + 1
        else:
            raise RuntimeError(f"unrecognized joint type: {jnt_typ}")
    return qpos


def clip_act(m: Model, act: jax.Array) -> jax.Array:
    if not m.na:
        return act

    actrange = jp.where(
        m.actuator_actlimited[:, None],
        m.actuator_actrange,
        jp.array([-jp.inf, jp.inf]),
    )

    for i in range(m.nu):
        adr = int(m.actuator_actadr[i])
        if adr < 0:
            continue
        n = int(m.actuator_actnum[i])
        for k in range(n):
            idx = adr + k
            act = act.at[idx].set(jp.clip(act[idx], actrange[i, 0], actrange[i, 1]))

    return act


def qpos_dot_func(m: Model, qpos: jax.Array, qvel: jax.Array) -> jax.Array:
    """Compute full qpos time-derivative including quaternion kinematics (R^4 ODE mode)."""

    def _qpos_omega_dot(qpos_omega, qvel_omega):
        return 0.5 * quat_mul(qpos_omega, jp.r_[0, qvel_omega])

    qpos_dot = jp.zeros_like(qpos)
    qi, vi = 0, 0
    for jnt_typ in m.jnt_type:
        if jnt_typ == JointType.FREE:
            qpos_xyz_dot = qvel[vi : vi + 3]
            qpos_omega_dot = _qpos_omega_dot(
                qpos[qi + 3 : qi + 7], qvel[vi + 3 : vi + 6]
            )
            qpos_dot = qpos_dot.at[qi : qi + 3].set(qpos_xyz_dot)
            qpos_dot = qpos_dot.at[qi + 3 : qi + 7].set(qpos_omega_dot)
            qi, vi = qi + 7, vi + 6
        elif jnt_typ == JointType.BALL:
            qpos_omega_dot = _qpos_omega_dot(qpos[qi : qi + 4], qvel[vi : vi + 3])
            qpos_dot = qpos_dot.at[qi : qi + 4].set(qpos_omega_dot)
            qi, vi = qi + 4, vi + 3
        elif jnt_typ in (JointType.HINGE, JointType.SLIDE):
            qpos_dot = qpos_dot.at[qi].set(qvel[vi])
            qi, vi = qi + 1, vi + 1
        else:
            raise RuntimeError(f"unrecognized joint type: {jnt_typ}")
    return qpos_dot


def qpos_dot_linear_only(m: Model, qpos: jax.Array, qvel: jax.Array) -> jax.Array:
    """Compute qpos time-derivative for linear joints only; quaternion components are zero."""
    qpos_dot = jp.zeros_like(qpos)
    qi, vi = 0, 0
    for jnt_typ in m.jnt_type:
        if jnt_typ == JointType.FREE:
            qpos_dot = qpos_dot.at[qi : qi + 3].set(qvel[vi : vi + 3])
            # quaternion components (qi+3 : qi+7) remain zero
            qi, vi = qi + 7, vi + 6
        elif jnt_typ == JointType.BALL:
            # quaternion components remain zero
            qi, vi = qi + 4, vi + 3
        elif jnt_typ in (JointType.HINGE, JointType.SLIDE):
            qpos_dot = qpos_dot.at[qi].set(qvel[vi])
            qi, vi = qi + 1, vi + 1
        else:
            raise RuntimeError(f"unrecognized joint type: {jnt_typ}")
    return qpos_dot


def qvel_angular_only(m: Model, qvel: jax.Array) -> jax.Array:
    """Extract angular velocity components (FREE rotation, BALL); zero out linear components."""
    result = jp.zeros_like(qvel)
    vi = 0
    for jnt_typ in m.jnt_type:
        if jnt_typ == JointType.FREE:
            result = result.at[vi + 3 : vi + 6].set(qvel[vi + 3 : vi + 6])
            vi = vi + 6
        elif jnt_typ == JointType.BALL:
            result = result.at[vi : vi + 3].set(qvel[vi : vi + 3])
            vi = vi + 3
        elif jnt_typ in (JointType.HINGE, JointType.SLIDE):
            vi = vi + 1
        else:
            raise RuntimeError(f"unrecognized joint type: {jnt_typ}")
    return result


def dexpinv_so3(phi: jax.Array, v: jax.Array) -> jax.Array:
    """Apply the inverse of the right Jacobian of the SO(3) exponential map to vector v.

    Closed-form: dexpinv(phi) * v = alpha * v + (1 - alpha) * (a . v) * a + (theta/2) * (a x v)
    where alpha = (theta/2) * cot(theta/2), a = phi / theta, theta = |phi|.
    Small-angle fallback: v + 0.5 * cross(phi, v).
    """
    theta_sq = jp.dot(phi, phi)
    is_small = theta_sq < 1e-16

    # Small-angle branch: v + 0.5 * cross(phi, v)
    small = v + 0.5 * jp.cross(phi, v)

    # General branch — guard theta_sq before sqrt to avoid inf gradient at zero
    theta_sq_safe = jp.where(is_small, 1.0, theta_sq)
    theta = jp.sqrt(theta_sq_safe)
    half_theta = 0.5 * theta
    cot_half = jp.cos(half_theta) / jp.sin(half_theta)
    alpha = half_theta * cot_half
    a = phi / theta
    general = alpha * v + (1.0 - alpha) * jp.dot(a, v) * a + half_theta * jp.cross(a, v)

    return jp.where(is_small, small, general)


def apply_dexpinv_angular(m: Model, phi: jax.Array, omega: jax.Array) -> jax.Array:
    """Apply dexpinv_so3 correction to angular velocity components per joint.

    For each rotational DOF block (FREE rotation, BALL), applies dexpinv_so3(phi_block, omega_block).
    Linear and scalar components pass through unchanged.
    """
    result = jp.zeros_like(omega)
    vi = 0
    for jnt_typ in m.jnt_type:
        if jnt_typ == JointType.FREE:
            result = result.at[vi : vi + 3].set(omega[vi : vi + 3])
            result = result.at[vi + 3 : vi + 6].set(
                dexpinv_so3(phi[vi + 3 : vi + 6], omega[vi + 3 : vi + 6])
            )
            vi = vi + 6
        elif jnt_typ == JointType.BALL:
            result = result.at[vi : vi + 3].set(
                dexpinv_so3(phi[vi : vi + 3], omega[vi : vi + 3])
            )
            vi = vi + 3
        elif jnt_typ in (JointType.HINGE, JointType.SLIDE):
            result = result.at[vi].set(omega[vi])
            vi = vi + 1
        else:
            raise RuntimeError(f"unrecognized joint type: {jnt_typ}")
    return result


def reconstruct_quat_positions(
    m: Model, qpos0: jax.Array, qpos: jax.Array, omega: jax.Array, dt: jax.Array
) -> jax.Array:
    """Replace quaternion components in qpos with exponential map reconstruction from qpos0.

    Linear joint components are kept from the ODE-integrated qpos unchanged.
    """
    qi, vi = 0, 0
    for jnt_typ in m.jnt_type:
        if jnt_typ == JointType.FREE:
            # linear position kept from ODE-integrated qpos
            quat = quat_integrate(qpos0[qi + 3 : qi + 7], omega[vi + 3 : vi + 6], dt)
            qpos = qpos.at[qi + 3 : qi + 7].set(quat)
            qi, vi = qi + 7, vi + 6
        elif jnt_typ == JointType.BALL:
            quat = quat_integrate(qpos0[qi : qi + 4], omega[vi : vi + 3], dt)
            qpos = qpos.at[qi : qi + 4].set(quat)
            qi, vi = qi + 4, vi + 3
        elif jnt_typ in (JointType.HINGE, JointType.SLIDE):
            # scalar joints kept from ODE-integrated qpos
            qi, vi = qi + 1, vi + 1
        else:
            raise RuntimeError(f"unrecognized joint type: {jnt_typ}")
    return qpos
