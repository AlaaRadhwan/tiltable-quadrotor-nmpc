from __future__ import annotations

import mujoco
import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    rotor_commands_to_virtual,
    virtual_to_wrench,
)


def rotor_wrench_body(
    thrust: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Compute the total rotor wrench in the vehicle BODY frame.

    Returns:

        [Fx, Fy, Fz, Tx, Ty, Tz]
    """

    z = rotor_commands_to_virtual(
        thrust,
        alpha,
    )

    return virtual_to_wrench(
        z
    )


def apply_rotor_wrench(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    thrust: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Convert physical rotor states into a body-frame wrench
    and pass it to the rigid MuJoCo vehicle.

    Physical actuator representation:

        thrust[4]
        alpha[4]

    Internal MuJoCo interface:

        [Fx, Fy, Fz, Tx, Ty, Tz]

    The six MuJoCo site actuators are aligned with the body
    frame, so MuJoCo handles transformation into world
    coordinates internally.
    """

    if model.nu != 6:
        raise RuntimeError(
            "Rigid plant is expected to have exactly "
            f"6 internal wrench controls, got nu={model.nu}."
        )

    wrench_body = rotor_wrench_body(
        thrust,
        alpha,
    )

    data.ctrl[:] = wrench_body

    return wrench_body