from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from tiltable_quadrotor_rl.control.nmpc_model import (
    build_dynamics_function,
)

from tiltable_quadrotor_rl.robot.parameters import (
    HOVER_THRUST,
)

from tiltable_quadrotor_rl.robot.rigid_plant import (
    full_state_from_mujoco,
    step_rigid_plant,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

XML_PATH = (
    PROJECT_ROOT
    / "src"
    / "tiltable_quadrotor_rl"
    / "robot"
    / "mjcf"
    / "tiltable_quadrotor_rigid.xml"
)


DURATION = 0.40

# Each MuJoCo step is 2 ms.
#
# The CasADi reference model is integrated much more finely so
# that its integration error is negligible compared with the
# MuJoCo timestep.
REFERENCE_SUBSTEPS = 8


def normalize_quaternion(
    quaternion: np.ndarray,
) -> np.ndarray:

    return (
        quaternion
        / np.linalg.norm(quaternion)
    )


def quaternion_angle_error(
    q1: np.ndarray,
    q2: np.ndarray,
) -> float:
    """
    Orientation difference between two unit quaternions.

    q and -q represent the same physical orientation, so the
    absolute value of the dot product is used.
    """

    q1 = normalize_quaternion(
        q1
    )

    q2 = normalize_quaternion(
        q2
    )

    dot = float(
        np.abs(
            np.dot(q1, q2)
        )
    )

    dot = np.clip(
        dot,
        0.0,
        1.0,
    )

    return float(
        2.0
        * np.arccos(dot)
    )


def command_at_time(
    time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic piecewise-constant test sequence.

    The commands deliberately excite translation, rotation,
    and the tilt-servo dynamics.
    """

    if time < 0.10:

        thrust = (
            HOVER_THRUST
            + np.array(
                [
                    0.7,
                    -0.5,
                    0.4,
                    -0.6,
                ]
            )
        )

        alpha_cmd = np.array(
            [
                0.45,
                -0.35,
                0.25,
                -0.15,
            ]
        )

    elif time < 0.20:

        thrust = np.array(
            [
                8.3,
                7.0,
                8.0,
                6.8,
            ]
        )

        alpha_cmd = np.array(
            [
                -0.35,
                0.50,
                -0.40,
                0.30,
            ]
        )

    elif time < 0.30:

        thrust = np.array(
            [
                7.0,
                8.2,
                6.9,
                8.4,
            ]
        )

        alpha_cmd = np.array(
            [
                0.20,
                0.20,
                -0.25,
                -0.25,
            ]
        )

    else:

        thrust = np.full(
            4,
            HOVER_THRUST,
        )

        alpha_cmd = np.zeros(
            4
        )

    return (
        thrust,
        alpha_cmd,
    )


def dynamics_numpy(
    dynamics,
    x: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:

    return np.asarray(
        dynamics(
            x,
            u,
        )
    ).reshape(-1)


def rk4_step(
    dynamics,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    One fourth-order Runge-Kutta step for the full 17-state
    continuous NMPC model.
    """

    k1 = dynamics_numpy(
        dynamics,
        x,
        u,
    )

    k2 = dynamics_numpy(
        dynamics,
        x + 0.5 * dt * k1,
        u,
    )

    k3 = dynamics_numpy(
        dynamics,
        x + 0.5 * dt * k2,
        u,
    )

    k4 = dynamics_numpy(
        dynamics,
        x + dt * k3,
        u,
    )

    x_next = (
        x
        + (dt / 6.0)
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )
    )

    # Quaternion normalization is numerical housekeeping only.
    #
    # The continuous differential equation itself preserves
    # unit norm, but finite-precision integration can introduce
    # tiny drift.
    x_next[6:10] = (
        normalize_quaternion(
            x_next[6:10]
        )
    )

    return x_next


def statistics(
    values: list[float],
) -> tuple[float, float, float]:

    values = np.asarray(
        values,
        dtype=float,
    )

    final = float(
        values[-1]
    )

    rms = float(
        np.sqrt(
            np.mean(values**2)
        )
    )

    maximum = float(
        np.max(values)
    )

    return (
        final,
        rms,
        maximum,
    )


def main():

    print()
    print("=" * 84)
    print("FINITE-TIME FULL MODEL vs MUJOCO")
    print("=" * 84)

    print()
    print("Loading:")
    print(XML_PATH)


    # ========================================================
    # Load rigid MuJoCo plant
    # ========================================================

    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )
    

    data = mujoco.MjData(
        model
    )

    dt = float(
        model.opt.timestep
    )

    reference_dt = (
        dt / REFERENCE_SUBSTEPS
    )

    dynamics = (
        build_dynamics_function()
    )


    # ========================================================
    # Initial state
    # ========================================================

    position_0 = np.array(
        [
            0.0,
            0.0,
            1.5,
        ]
    )

    velocity_0 = np.array(
        [
            0.25,
            -0.15,
            0.10,
        ]
    )

    quaternion_0 = np.array(
        [
            0.96,
            0.12,
            -0.18,
            0.18,
        ],
        dtype=float,
    )

    quaternion_0 = (
        normalize_quaternion(
            quaternion_0
        )
    )

    omega_0 = np.array(
        [
            0.30,
            -0.20,
            0.40,
        ]
    )

    alpha_0 = np.array(
        [
            0.10,
            -0.05,
            0.08,
            -0.02,
        ]
    )


    # ========================================================
    # Initialize MuJoCo
    # ========================================================

    mujoco.mj_resetData(
        model,
        data,
    )

    data.qpos[0:3] = (
        position_0
    )

    data.qpos[3:7] = (
        quaternion_0
    )

    data.qvel[0:3] = (
        velocity_0
    )

    data.qvel[3:6] = (
        omega_0
    )

    mujoco.mj_forward(
        model,
        data,
    )

    alpha_mujoco = (
        alpha_0.copy()
    )


    # ========================================================
    # Initialize continuous reference model
    # ========================================================

    x_reference = np.concatenate(
        [
            position_0,
            velocity_0,
            quaternion_0,
            omega_0,
            alpha_0,
        ]
    )


    # ========================================================
    # Error histories
    # ========================================================

    position_error = []
    velocity_error = []
    attitude_error = []
    omega_error = []
    alpha_error = []

    reference_quaternion_norm_error = []
    mujoco_quaternion_norm_error = []


    # ========================================================
    # Simulate
    # ========================================================

    number_of_steps = int(
        round(
            DURATION / dt
        )
    )

    for step in range(
        number_of_steps
    ):

        time = (
            step * dt
        )

        thrust, alpha_cmd = (
            command_at_time(
                time
            )
        )

        u = np.concatenate(
            [
                thrust,
                alpha_cmd,
            ]
        )


        # ----------------------------------------------------
        # High-resolution reference integration
        # ----------------------------------------------------

        for _ in range(
            REFERENCE_SUBSTEPS
        ):

            x_reference = rk4_step(
                dynamics,
                x_reference,
                u,
                reference_dt,
            )


        # ----------------------------------------------------
        # MuJoCo + exact external servo state
        # ----------------------------------------------------

        alpha_mujoco = (
            step_rigid_plant(
                model,
                data,
                alpha_mujoco,
                thrust,
                alpha_cmd,
            )
        )

        x_mujoco = (
            full_state_from_mujoco(
                data,
                alpha_mujoco,
            )
        )


        # ----------------------------------------------------
        # Compare
        # ----------------------------------------------------

        position_error.append(
            np.linalg.norm(
                x_reference[0:3]
                - x_mujoco[0:3]
            )
        )

        velocity_error.append(
            np.linalg.norm(
                x_reference[3:6]
                - x_mujoco[3:6]
            )
        )

        attitude_error.append(
            quaternion_angle_error(
                x_reference[6:10],
                x_mujoco[6:10],
            )
        )

        omega_error.append(
            np.linalg.norm(
                x_reference[10:13]
                - x_mujoco[10:13]
            )
        )

        alpha_error.append(
            np.linalg.norm(
                x_reference[13:17]
                - x_mujoco[13:17]
            )
        )

        reference_quaternion_norm_error.append(
            abs(
                np.linalg.norm(
                    x_reference[6:10]
                )
                - 1.0
            )
        )

        mujoco_quaternion_norm_error.append(
            abs(
                np.linalg.norm(
                    x_mujoco[6:10]
                )
                - 1.0
            )
        )


    # ========================================================
    # Report
    # ========================================================

    print()
    print(f"Duration              : {DURATION:.3f} s")
    print(f"MuJoCo timestep       : {1e3 * dt:.3f} ms")
    print(
        f"Reference RK4 timestep: "
        f"{1e3 * reference_dt:.3f} ms"
    )

    print()
    print(
        f"{'quantity':22s}"
        f"{'final':>16s}"
        f"{'RMS':>16s}"
        f"{'max':>16s}"
    )

    print("-" * 70)

    quantities = [
        (
            "position [m]",
            position_error,
        ),
        (
            "velocity [m/s]",
            velocity_error,
        ),
        (
            "attitude [rad]",
            attitude_error,
        ),
        (
            "omega [rad/s]",
            omega_error,
        ),
        (
            "alpha [rad]",
            alpha_error,
        ),
    ]

    for name, values in quantities:

        final, rms, maximum = (
            statistics(
                values
            )
        )

        print(
            f"{name:22s}"
            f"{final:16.6e}"
            f"{rms:16.6e}"
            f"{maximum:16.6e}"
        )


    print()
    print(
        "Max reference quaternion norm error : "
        f"{max(reference_quaternion_norm_error):.3e}"
    )

    print(
        "Max MuJoCo quaternion norm error    : "
        f"{max(mujoco_quaternion_norm_error):.3e}"
    )

    print()
    print("Final reference alpha:")
    print(
        x_reference[13:17]
    )

    print()
    print("Final MuJoCo alpha:")
    print(
        alpha_mujoco
    )


if __name__ == "__main__":
    main()
