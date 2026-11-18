from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from tiltable_quadrotor_rl.control.nmpc_model import (
    build_dynamics_function,
)

from tiltable_quadrotor_rl.robot.rigid_actuation import (
    apply_rotor_wrench,
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

N_TRIALS = 100

RNG = np.random.default_rng(
    12345
)


def random_unit_quaternion() -> np.ndarray:
    """
    Generate a random unit quaternion in MuJoCo/CasADi ordering:

        [qw, qx, qy, qz]
    """

    q = RNG.normal(
        size=4
    )

    q /= np.linalg.norm(
        q
    )

    return q


def rms_and_max(
    values: list[np.ndarray],
) -> tuple[float, float]:

    array = np.asarray(
        values
    )

    rms = float(
        np.sqrt(
            np.mean(array**2)
        )
    )

    maximum = float(
        np.max(
            np.abs(array)
        )
    )

    return rms, maximum


def run_trials(
    dynamics,
    model,
    data,
):
    errors = {
        "position_dot": [],
        "velocity_dot": [],
        "quaternion_dot": [],
        "omega_dot": [],
    }

    base_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "base",
    )

    if base_id < 0:
        raise RuntimeError(
            "Body 'base' not found."
        )

    for _ in range(N_TRIALS):

        position = RNG.uniform(
            -2.0,
            2.0,
            size=3,
        )

        position[2] = RNG.uniform(
            0.5,
            3.0,
        )

        quaternion = (
            random_unit_quaternion()
        )

        velocity = RNG.uniform(
            -2.0,
            2.0,
            size=3,
        )

        omega = RNG.uniform(
            -3.0,
            3.0,
            size=3,
        )

        alpha = RNG.uniform(
            -1.5,
            1.5,
            size=4,
        )

        thrust = RNG.uniform(
            0.0,
            20.0,
            size=4,
        )

        # For this instantaneous rigid-body comparison we set
        # alpha_cmd = alpha so:
        #
        #     alpha_dot = 0
        #
        # and servo dynamics do not affect the comparison.
        alpha_cmd = alpha.copy()


        # ====================================================
        # MuJoCo state
        # ====================================================

        mujoco.mj_resetData(
            model,
            data,
        )

        data.qpos[0:3] = (
            position
        )

        data.qpos[3:7] = (
            quaternion
        )

        # Free-joint translational velocity is represented in
        # world coordinates in this project comparison.
        data.qvel[0:3] = (
            velocity
        )

        # Rotational free-joint velocity is body-local.
        data.qvel[3:6] = (
            omega
        )


        apply_rotor_wrench(
            model,
            data,
            thrust,
            alpha,
        )

        mujoco.mj_forward(
            model,
            data,
        )


        # ====================================================
        # CasADi state/input
        # ====================================================

        x = np.concatenate(
            [
                position,
                velocity,
                quaternion,
                omega,
                alpha,
            ]
        )

        u = np.concatenate(
            [
                thrust,
                alpha_cmd,
            ]
        )

        x_dot = np.asarray(
            dynamics(
                x,
                u,
            )
        ).reshape(-1)


        # ====================================================
        # Quaternion derivative from MuJoCo
        # ====================================================

        # Our state omega is body-frame angular velocity.
        #
        # For mju_derivQuat in this diagnostic we first construct
        # the equivalent world-frame angular velocity.
        omega_world = np.zeros(
            3
        )

        mujoco.mju_rotVecQuat(
            omega_world,
            omega,
            quaternion,
        )

        quaternion_dot_mujoco = np.zeros(
            4
        )

        mujoco.mju_derivQuat(
            quaternion_dot_mujoco,
            quaternion,
            omega_world,
        )


        # ====================================================
        # Record differences
        # ====================================================

        errors[
            "position_dot"
        ].append(
            x_dot[0:3]
            - data.qvel[0:3]
        )

        errors[
            "velocity_dot"
        ].append(
            x_dot[3:6]
            - data.qacc[0:3]
        )

        errors[
            "quaternion_dot"
        ].append(
            x_dot[6:10]
            - quaternion_dot_mujoco
        )

        errors[
            "omega_dot"
        ].append(
            x_dot[10:13]
            - data.qacc[3:6]
        )

    return errors


def print_results(
    errors,
):
    for name in [
        "position_dot",
        "velocity_dot",
        "quaternion_dot",
        "omega_dot",
    ]:

        rms, maximum = rms_and_max(
            errors[name]
        )

        print(
            f"{name:20s} "
            f"RMS = {rms:.3e}   "
            f"max = {maximum:.3e}"
        )


def main():

    print()
    print("=" * 72)
    print("RIGID CASADI NMPC MODEL vs MUJOCO")
    print("=" * 72)

    print()
    print("Loading:")
    print(XML_PATH)

    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )

    data = mujoco.MjData(
        model
    )

    assert model.nq == 7
    assert model.nv == 6

    dynamics = (
        build_dynamics_function()
    )


    errors = run_trials(
        dynamics,
        model,
        data,
    )


    print()
    print("=" * 72)
    print(
        "100 RANDOM STATES — "
        "ARBITRARY ATTITUDE AND ANGULAR VELOCITY"
    )
    print("=" * 72)

    print_results(
        errors
    )


if __name__ == "__main__":
    main()
