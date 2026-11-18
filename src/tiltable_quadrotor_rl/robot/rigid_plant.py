from __future__ import annotations

import mujoco
import numpy as np

from tiltable_quadrotor_rl.robot.rigid_actuation import (
    apply_rotor_wrench,
)

from tiltable_quadrotor_rl.robot.tilt_servo import (
    first_order_servo_step,
)


def full_state_from_mujoco(
    data: mujoco.MjData,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Assemble the 17-state representation used by the NMPC:

        x = [
            position_W,      3
            velocity_W,      3
            quaternion_WB,   4
            omega_B,         3
            alpha,           4
        ]
    """

    alpha = np.asarray(
        alpha,
        dtype=float,
    )

    if alpha.shape != (4,):
        raise ValueError(
            f"Expected alpha shape (4,), got {alpha.shape}"
        )

    return np.concatenate(
        [
            data.qpos[0:3],
            data.qvel[0:3],
            data.qpos[3:7],
            data.qvel[3:6],
            alpha,
        ]
    )


def step_rigid_plant(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    alpha: np.ndarray,
    thrust: np.ndarray,
    alpha_cmd: np.ndarray,
) -> np.ndarray:
    """
    Advance the nominal rigid plant by one MuJoCo timestep.

    The tilt servos are explicit actuator states outside
    MuJoCo's mechanical coordinates.

    During one physics timestep the real tilt angle changes
    continuously. MuJoCo needs one constant wrench over that
    timestep, so we evaluate the rotor wrench at the exact
    first-order servo angle halfway through the interval.

    This is substantially better than using only alpha_k or
    alpha_{k+1}.

    Returns
    -------
    alpha_next:
        Physical servo angles at the end of the timestep.
    """

    dt = float(
        model.opt.timestep
    )

    alpha = np.asarray(
        alpha,
        dtype=float,
    )

    thrust = np.asarray(
        thrust,
        dtype=float,
    )

    alpha_cmd = np.asarray(
        alpha_cmd,
        dtype=float,
    )

    if alpha.shape != (4,):
        raise ValueError(
            f"Expected alpha shape (4,), got {alpha.shape}"
        )

    if thrust.shape != (4,):
        raise ValueError(
            f"Expected thrust shape (4,), got {thrust.shape}"
        )

    if alpha_cmd.shape != (4,):
        raise ValueError(
            f"Expected alpha_cmd shape (4,), got {alpha_cmd.shape}"
        )

    # Exact first-order angle at the middle of the physics step.
    #
    # We use this angle for the constant wrench supplied to
    # MuJoCo during the interval.
    alpha_mid = first_order_servo_step(
        alpha,
        alpha_cmd,
        0.5 * dt,
    )

    apply_rotor_wrench(
        model,
        data,
        thrust,
        alpha_mid,
    )

    mujoco.mj_step(
        model,
        data,
    )

    # Exact first-order actuator state at the end of the step.
    alpha_next = first_order_servo_step(
        alpha,
        alpha_cmd,
        dt,
    )

    return alpha_next
