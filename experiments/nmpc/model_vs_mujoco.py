from pathlib import Path

import mujoco
import numpy as np

from tiltable_quadrotor_rl.control.nmpc_model import (
    NX,
    NU,
    build_dynamics_function,
)

from tiltable_quadrotor_rl.robot.tilt_servo import (
    write_tilt_angles,
)


XML_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "tiltable_quadrotor_rl"
    / "robot"
    / "mjcf"
    / "tiltable_quadrotor.xml"
)


def random_unit_quaternion(rng):
    """
    Generate an arbitrary normalized quaternion
    in MuJoCo / project ordering:

        [qw, qx, qy, qz]
    """

    q = rng.normal(size=4)
    q /= np.linalg.norm(q)

    return q


def run_trials(
    *,
    dynamics,
    model,
    data,
    rng,
    num_trials,
    moving_rotation,
):
    """
    Compare instantaneous derivatives from:

        1. our CasADi rigid-body model
        2. MuJoCo forward dynamics
    """

    errors = {
        "position_dot": [],
        "velocity_dot": [],
        "quaternion_dot": [],
        "omega_dot_free": [],
        "omega_dot_locked": [],
        "tilt_qacc": [],
    }

    for _ in range(num_trials):

        mujoco.mj_resetData(
            model,
            data,
        )

        # -----------------------------------------------------
        # Arbitrary physical state
        # -----------------------------------------------------

        position = rng.uniform(
            low=[-1.0, -1.0, 0.5],
            high=[1.0, 1.0, 2.0],
        )

        quaternion = random_unit_quaternion(
            rng
        )

        velocity = rng.uniform(
            -2.0,
            2.0,
            size=3,
        )

        if moving_rotation:
            omega = rng.uniform(
                -3.0,
                3.0,
                size=3,
            )
        else:
            omega = np.zeros(3)

        alpha = rng.uniform(
            -1.5,
            1.5,
            size=4,
        )

        thrust = rng.uniform(
            0.0,
            20.0,
            size=4,
        )

        # Keep alpha_cmd = alpha.
        #
        # Therefore the servo derivative is zero during this
        # instantaneous rigid-body comparison.
        alpha_cmd = alpha.copy()

        # -----------------------------------------------------
        # Write the same state into MuJoCo
        # -----------------------------------------------------

        data.qpos[0:3] = position
        data.qpos[3:7] = quaternion

        # MuJoCo free-joint qvel:
        #
        #   [world linear velocity,
        #    body angular velocity]
        data.qvel[0:3] = velocity
        data.qvel[3:6] = omega

        write_tilt_angles(
            model,
            data,
            alpha,
        )

        data.ctrl[:] = thrust

        # Evaluate MuJoCo forward dynamics at this exact state.
        mujoco.mj_forward(
            model,
            data,
        )

        # -----------------------------------------------------
        # Inspect MuJoCo's full articulated mass matrix.
        # -----------------------------------------------------

        M = np.zeros(
            (model.nv, model.nv),
            dtype=np.float64,
        )

        mujoco.mj_fullM(
            model,
            data,
            M,
        )

        # If the four tilt accelerations were constrained to zero,
        # only the 6x6 base block participates in the base solve.
        locked_base_qacc = np.linalg.solve(
            M[0:6, 0:6],
            data.qfrc_smooth[0:6],
        )

        # -----------------------------------------------------
        # Construct the SAME state/input for our CasADi model
        # -----------------------------------------------------

        x = np.zeros(NX)

        x[0:3] = position
        x[3:6] = velocity
        x[6:10] = quaternion
        x[10:13] = omega
        x[13:17] = alpha

        u = np.zeros(NU)

        u[0:4] = thrust
        u[4:8] = alpha_cmd

        x_dot = np.asarray(
            dynamics(x, u)
        ).reshape(-1)

        # -----------------------------------------------------
        # MuJoCo reference derivatives
        # -----------------------------------------------------

        mujoco_position_dot = (
            data.qvel[0:3].copy()
        )

        mujoco_velocity_dot = (
            data.qacc[0:3].copy()
        )

        mujoco_omega_dot = (
            data.qacc[3:6].copy()
        )

        # MuJoCo provides the exact quaternion derivative
        # corresponding to its quaternion convention and
        # local angular velocity.
        mujoco_quaternion_dot = np.zeros(4)
 
        omega_world = np.zeros(3)

        mujoco.mju_rotVecQuat(
            omega_world,
            omega,
            quaternion,
        )

        mujoco_quaternion_dot = np.zeros(4)

        mujoco.mju_derivQuat(
            mujoco_quaternion_dot,
            quaternion,
            omega_world,
        )

        # -----------------------------------------------------
        # Store errors
        # -----------------------------------------------------

        errors["position_dot"].append(
            x_dot[0:3]
            - mujoco_position_dot
        )

        errors["velocity_dot"].append(
            x_dot[3:6]
            - mujoco_velocity_dot
        )

        errors["quaternion_dot"].append(
            x_dot[6:10]
            - mujoco_quaternion_dot
        )

        errors["omega_dot_free"].append(
            x_dot[10:13]
            - data.qacc[3:6]
        )

        errors["omega_dot_locked"].append(
            x_dot[10:13]
            - locked_base_qacc[3:6]
        )

        errors["tilt_qacc"].append(
            data.qacc[6:10].copy()
        )

    return {
        name: np.asarray(values)
        for name, values in errors.items()
    }


def print_statistics(
    title,
    errors,
):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    for name, error in errors.items():

        rms = np.sqrt(
            np.mean(error**2)
        )

        maximum = np.max(
            np.abs(error)
        )

        print(
            f"{name:20s} "
            f"RMS = {rms:.3e}   "
            f"max = {maximum:.3e}"
        )


def main():

    print()
    print("=" * 72)
    print("CASADI NMPC MODEL vs MUJOCO")
    print("=" * 72)

    print(f"\nLoading:\n{XML_PATH}")

    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )

    data = mujoco.MjData(model)

    dynamics = build_dynamics_function()

    rng = np.random.default_rng(42)

    # ---------------------------------------------------------
    # Case A:
    # no body rotation.
    #
    # This isolates:
    #   - thrust geometry
    #   - quaternion rotation matrix
    #   - gravity
    #   - mass
    #   - torque / inertia response
    # ---------------------------------------------------------

    errors_static_rotation = run_trials(
        dynamics=dynamics,
        model=model,
        data=data,
        rng=rng,
        num_trials=100,
        moving_rotation=False,
    )

    print_statistics(
        "CASE A — ARBITRARY ATTITUDE, OMEGA = 0",
        errors_static_rotation,
    )

    # ---------------------------------------------------------
    # Case B:
    # nonzero angular velocity.
    #
    # This additionally tests:
    #   - quaternion kinematics
    #   - gyroscopic coupling omega x (I omega)
    # ---------------------------------------------------------

    errors_moving_rotation = run_trials(
        dynamics=dynamics,
        model=model,
        data=data,
        rng=rng,
        num_trials=100,
        moving_rotation=True,
    )

    print_statistics(
        "CASE B — ARBITRARY ATTITUDE AND ANGULAR VELOCITY",
        errors_moving_rotation,
    )


if __name__ == "__main__":
    main()
