from __future__ import annotations

import math

import numpy as np


# Nominal baseline values.
#
# These currently come from the reference platform/paper,
# but they are model parameters, not hard physical truths
# for every future vehicle.
from tiltable_quadrotor_rl.robot.parameters import (
    SERVO_TIME_CONSTANT as DEFAULT_TIME_CONSTANT,
    SERVO_COMMAND_LIMIT as DEFAULT_COMMAND_LIMIT,
    SERVO_MECHANICAL_LIMIT as DEFAULT_MECHANICAL_LIMIT,
)


def first_order_servo_step(
    alpha: np.ndarray,
    alpha_cmd: np.ndarray,
    dt: float,
    *,
    time_constant: float = DEFAULT_TIME_CONSTANT,
    command_limit: float = DEFAULT_COMMAND_LIMIT,
    mechanical_limit: float = DEFAULT_MECHANICAL_LIMIT,
) -> np.ndarray:
    """
    Advance the four rotor tilt angles by one simulation step.

    Continuous-time model:

        d(alpha)/dt = (alpha_cmd - alpha) / tau

    The update below is the exact analytical solution over dt
    assuming alpha_cmd stays constant during the time step.

    Parameters
    ----------
    alpha:
        Current physical tilt angle(s), rad.

    alpha_cmd:
        Commanded tilt angle(s), rad.

    dt:
        Simulation time step, s.

    time_constant:
        First-order servo time constant tau, s.

    command_limit:
        Maximum commanded angle magnitude, rad.

    mechanical_limit:
        Maximum physical joint-angle magnitude, rad.

    Returns
    -------
    np.ndarray
        Updated physical tilt angle(s).
    """

    alpha = np.asarray(
        alpha,
        dtype=float,
    )

    alpha_cmd = np.asarray(
        alpha_cmd,
        dtype=float,
    )

    if time_constant <= 0.0:
        raise ValueError(
            "time_constant must be > 0"
        )

    if dt <= 0.0:
        raise ValueError(
            "dt must be > 0"
        )

    # Controller command saturation.
    cmd = np.clip(
        alpha_cmd,
        -command_limit,
        command_limit,
    )

    # Exact solution of:
    #
    #   alpha_dot = (cmd - alpha) / tau
    #
    decay = math.exp(
        -dt / time_constant
    )

    alpha_next = (
        cmd
        + (alpha - cmd) * decay
    )

    # Mechanical travel limit.
    alpha_next = np.clip(
        alpha_next,
        -mechanical_limit,
        mechanical_limit,
    )

    return alpha_next

def write_tilt_angles(
    model,
    data,
    alpha: np.ndarray,
) -> None:
    """
    Write externally modeled servo angles into the four
    MuJoCo tilt joints.

    The tilt joints are treated as kinematic orientation
    coordinates. We intentionally set their joint velocities
    to zero so servo motion does not introduce invented
    actuator/inertial reaction torques into the vehicle.
    """

    import mujoco

    alpha = np.asarray(
        alpha,
        dtype=float,
    )

    if alpha.shape != (4,):
        raise ValueError(
            f"Expected alpha shape (4,), got {alpha.shape}"
        )

    for i in range(4):

        joint_name = (
            f"tilt_{i + 1}"
        )

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if jid < 0:
            raise RuntimeError(
                f"Joint not found: {joint_name}"
            )

        qpos_adr = (
            model.jnt_qposadr[jid]
        )

        dof_adr = (
            model.jnt_dofadr[jid]
        )

        data.qpos[qpos_adr] = (
            alpha[i]
        )

        # The servo mechanics themselves are deliberately
        # outside the rigid-body dynamics model.
        data.qvel[dof_adr] = 0.0