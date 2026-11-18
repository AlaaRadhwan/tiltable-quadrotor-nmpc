from __future__ import annotations

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from tiltable_quadrotor_rl.control.end_effector import (
    DEFAULT_END_EFFECTOR,
    body_to_end_effector_kinematics,
)
from tiltable_quadrotor_rl.control.nmpc_effector_controller import (
    EffectorNmpcController,
)
from tiltable_quadrotor_rl.control.nmpc_effector_reference import (
    end_effector_reference_to_body_reference,
)
from tiltable_quadrotor_rl.control.nmpc_effector_trajectory import (
    EffectorPoseWaypoint,
    QuinticEffectorPoseTrajectory,
    quaternion_from_rpy,
)
from tiltable_quadrotor_rl.control.nmpc_ocp import (
    CONTROL_PERIOD,
)
from tiltable_quadrotor_rl.control.nmpc_reference import (
    normalize_quaternion,
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

BUILD_DIR = (
    PROJECT_ROOT
    / "build"
    / "acados"
    / "tiltable_quadrotor_effector_nmpc"
)

JSON_FILE = (
    BUILD_DIR
    / "tiltable_quadrotor_effector_nmpc.json"
)

LOG_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "nmpc"
)

VIEWER_HZ = 50.0
TERMINAL_HZ = 1.0


def pose_waypoint(
    time_s: float,
    position_W: list[float],
    rpy_deg: list[float],
) -> EffectorPoseWaypoint:

    roll, pitch, yaw = np.radians(
        np.asarray(
            rpy_deg,
            dtype=float,
        )
    )

    return EffectorPoseWaypoint(
        time_s=time_s,
        position_W=np.asarray(
            position_W,
            dtype=float,
        ),
        quaternion_WE=quaternion_from_rpy(
            roll,
            pitch,
            yaw,
        ),
    )


# One smooth free-space 6-DoF END-EFFECTOR trajectory.
#
# Roll and pitch are included deliberately. With a 0.15 m vertical
# tool offset, attitude changes move the tool relative to the CoG,
# so this trajectory genuinely exercises the rigid tool transform.

TRAJECTORY_WAYPOINTS = [
    pose_waypoint(
        0.0,
        [0.0, 0.0, 1.5],
        [0.0, 0.0, 0.0],
    ),
    pose_waypoint(
        2.0,
        [0.0, 0.0, 1.5],
        [0.0, 0.0, 0.0],
    ),
    pose_waypoint(
        7.0,
        [0.8, 0.5, 1.7],
        [10.0, -8.0, 25.0],
    ),
    pose_waypoint(
        12.0,
        [-0.4, 0.8, 1.4],
        [-12.0, 10.0, -20.0],
    ),
    pose_waypoint(
        17.0,
        [-0.6, -0.4, 1.65],
        [8.0, 12.0, 35.0],
    ),
    pose_waypoint(
        22.0,
        [0.4, -0.5, 1.45],
        [-8.0, -10.0, -25.0],
    ),
    pose_waypoint(
        27.0,
        [0.0, 0.0, 1.5],
        [0.0, 0.0, 0.0],
    ),
    pose_waypoint(
        30.0,
        [0.0, 0.0, 1.5],
        [0.0, 0.0, 0.0],
    ),
]


def quaternion_to_euler_deg(
    quaternion: np.ndarray,
) -> np.ndarray:

    qw, qx, qy, qz = normalize_quaternion(
        quaternion
    )

    sinr_cosp = 2.0 * (
        qw * qx
        + qy * qz
    )

    cosr_cosp = (
        1.0
        - 2.0
        * (
            qx * qx
            + qy * qy
        )
    )

    roll = math.atan2(
        sinr_cosp,
        cosr_cosp,
    )

    sinp = 2.0 * (
        qw * qy
        - qz * qx
    )

    pitch = math.asin(
        float(
            np.clip(
                sinp,
                -1.0,
                1.0,
            )
        )
    )

    siny_cosp = 2.0 * (
        qw * qz
        + qx * qy
    )

    cosy_cosp = (
        1.0
        - 2.0
        * (
            qy * qy
            + qz * qz
        )
    )

    yaw = math.atan2(
        siny_cosp,
        cosy_cosp,
    )

    return np.degrees(
        [
            roll,
            pitch,
            yaw,
        ]
    )


def quaternion_error_deg(
    actual: np.ndarray,
    reference: np.ndarray,
) -> float:

    actual = normalize_quaternion(
        actual
    )

    reference = normalize_quaternion(
        reference
    )

    dot = abs(
        float(
            np.dot(
                actual,
                reference,
            )
        )
    )

    dot = float(
        np.clip(
            dot,
            0.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            2.0
            * np.arccos(
                dot
            )
        )
    )


def reset_vehicle_to_first_effector_reference(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: QuinticEffectorPoseTrajectory,
) -> np.ndarray:

    mujoco.mj_resetData(
        model,
        data,
    )

    effector_reference = trajectory.sample(
        trajectory.start_time_s
    )

    body_reference = (
        end_effector_reference_to_body_reference(
            effector_reference,
            transform=DEFAULT_END_EFFECTOR,
        )
    )

    data.qpos[0:3] = (
        body_reference.position_W
    )

    data.qpos[3:7] = (
        body_reference.quaternion_WB
    )

    data.qvel[0:3] = (
        body_reference.velocity_W
    )

    data.qvel[3:6] = (
        body_reference.omega_B
    )

    data.ctrl[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    return np.zeros(
        4,
        dtype=float,
    )


def update_reference_marker(
    viewer,
    reference_position_W: np.ndarray,
) -> None:
    """
    Green sphere = desired tool position.

    The ACTUAL tool is already visible in the MJCF as the orange/red
    rigid end-effector geometry, so we no longer create a fake blue
    "actual" marker in Python.
    """

    target = (
        viewer.user_scn.geoms[0]
    )

    mujoco.mjv_initGeom(
        target,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array(
            [
                0.045,
                0.045,
                0.045,
            ],
            dtype=float,
        ),
        np.asarray(
            reference_position_W,
            dtype=float,
        ),
        np.eye(3).reshape(-1),
        np.array(
            [
                0.2,
                0.9,
                0.2,
                0.55,
            ],
            dtype=float,
        ),
    )

    viewer.user_scn.ngeom = 1


class EffectorLogger:
    def __init__(self) -> None:

        LOG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.csv_path = (
            LOG_DIR
            / f"effector_nmpc_closed_loop_{timestamp}.csv"
        )

        self.summary_path = (
            LOG_DIR
            / f"effector_nmpc_closed_loop_{timestamp}_summary.txt"
        )

        self._file = open(
            self.csv_path,
            "w",
            newline="",
        )

        self._fieldnames = [
            "time_s",
            "solve_time_ms",
            "solver_status",
            "deadline_missed",

            "body_px_m",
            "body_py_m",
            "body_pz_m",

            "eff_px_m",
            "eff_py_m",
            "eff_pz_m",

            "eff_px_ref_m",
            "eff_py_ref_m",
            "eff_pz_ref_m",

            "eff_vx_mps",
            "eff_vy_mps",
            "eff_vz_mps",

            "eff_vx_ref_mps",
            "eff_vy_ref_mps",
            "eff_vz_ref_mps",

            "eff_roll_deg",
            "eff_pitch_deg",
            "eff_yaw_deg",

            "eff_roll_ref_deg",
            "eff_pitch_ref_deg",
            "eff_yaw_ref_deg",

            "position_error_m",
            "velocity_error_mps",
            "attitude_error_deg",

            "thrust_1_N",
            "thrust_2_N",
            "thrust_3_N",
            "thrust_4_N",

            "alpha_cmd_1_rad",
            "alpha_cmd_2_rad",
            "alpha_cmd_3_rad",
            "alpha_cmd_4_rad",
        ]

        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fieldnames,
        )

        self._writer.writeheader()

        self.solve_times_ms = []
        self.position_errors = []
        self.velocity_errors = []
        self.attitude_errors_deg = []

        self.max_thrust_N = 0.0
        self.max_abs_alpha_cmd_rad = 0.0

        self.deadline_misses = 0
        self.control_updates = 0
        self.solver_failures = 0
        self.closed = False

    def record(
        self,
        *,
        time_s: float,
        state: np.ndarray,
        effector,
        reference,
        thrust: np.ndarray,
        alpha_cmd: np.ndarray,
        solve_time_s: float,
        solver_status: int,
    ) -> None:

        rpy = quaternion_to_euler_deg(
            effector.quaternion_WE
        )

        rpy_ref = quaternion_to_euler_deg(
            reference.quaternion_WE
        )

        position_error = float(
            np.linalg.norm(
                effector.position_W
                - reference.position_W
            )
        )

        velocity_error = float(
            np.linalg.norm(
                effector.velocity_W
                - reference.velocity_W
            )
        )

        attitude_error = (
            quaternion_error_deg(
                effector.quaternion_WE,
                reference.quaternion_WE,
            )
        )

        solve_time_ms = (
            1000.0
            * solve_time_s
        )

        deadline_missed = (
            solve_time_s
            > CONTROL_PERIOD
        )

        self.solve_times_ms.append(
            solve_time_ms
        )

        self.position_errors.append(
            position_error
        )

        self.velocity_errors.append(
            velocity_error
        )

        self.attitude_errors_deg.append(
            attitude_error
        )

        self.max_thrust_N = max(
            self.max_thrust_N,
            float(
                np.max(
                    thrust
                )
            ),
        )

        self.max_abs_alpha_cmd_rad = max(
            self.max_abs_alpha_cmd_rad,
            float(
                np.max(
                    np.abs(
                        alpha_cmd
                    )
                )
            ),
        )

        self.control_updates += 1

        if deadline_missed:
            self.deadline_misses += 1

        self._writer.writerow(
            {
                "time_s": time_s,
                "solve_time_ms": solve_time_ms,
                "solver_status": solver_status,
                "deadline_missed": int(
                    deadline_missed
                ),

                "body_px_m": state[0],
                "body_py_m": state[1],
                "body_pz_m": state[2],

                "eff_px_m": effector.position_W[0],
                "eff_py_m": effector.position_W[1],
                "eff_pz_m": effector.position_W[2],

                "eff_px_ref_m": reference.position_W[0],
                "eff_py_ref_m": reference.position_W[1],
                "eff_pz_ref_m": reference.position_W[2],

                "eff_vx_mps": effector.velocity_W[0],
                "eff_vy_mps": effector.velocity_W[1],
                "eff_vz_mps": effector.velocity_W[2],

                "eff_vx_ref_mps": reference.velocity_W[0],
                "eff_vy_ref_mps": reference.velocity_W[1],
                "eff_vz_ref_mps": reference.velocity_W[2],

                "eff_roll_deg": rpy[0],
                "eff_pitch_deg": rpy[1],
                "eff_yaw_deg": rpy[2],

                "eff_roll_ref_deg": rpy_ref[0],
                "eff_pitch_ref_deg": rpy_ref[1],
                "eff_yaw_ref_deg": rpy_ref[2],

                "position_error_m": position_error,
                "velocity_error_mps": velocity_error,
                "attitude_error_deg": attitude_error,

                "thrust_1_N": thrust[0],
                "thrust_2_N": thrust[1],
                "thrust_3_N": thrust[2],
                "thrust_4_N": thrust[3],

                "alpha_cmd_1_rad": alpha_cmd[0],
                "alpha_cmd_2_rad": alpha_cmd[1],
                "alpha_cmd_3_rad": alpha_cmd[2],
                "alpha_cmd_4_rad": alpha_cmd[3],
            }
        )

    def note_solver_failure(
        self,
    ) -> None:
        self.solver_failures += 1

    def close(
        self,
        *,
        completed: bool,
        final_time_s: float,
    ) -> None:

        if self.closed:
            return

        self._file.flush()
        self._file.close()

        solve = np.asarray(
            self.solve_times_ms,
            dtype=float,
        )

        position = np.asarray(
            self.position_errors,
            dtype=float,
        )

        velocity = np.asarray(
            self.velocity_errors,
            dtype=float,
        )

        attitude = np.asarray(
            self.attitude_errors_deg,
            dtype=float,
        )

        def rms(
            values: np.ndarray,
        ) -> float:

            if values.size == 0:
                return float("nan")

            return float(
                np.sqrt(
                    np.mean(
                        values * values
                    )
                )
            )

        with open(
            self.summary_path,
            "w",
        ) as summary:

            summary.write(
                "END-EFFECTOR NMPC CLOSED-LOOP SUMMARY\n"
            )

            summary.write(
                "=" * 72
                + "\n\n"
            )

            summary.write(
                f"Run completed normally: {completed}\n"
            )

            summary.write(
                f"Final simulation time [s]: {final_time_s:.3f}\n"
            )

            summary.write(
                "Tool offset body frame [m]: "
                f"{DEFAULT_END_EFFECTOR.position_B.tolist()}\n"
            )

            summary.write(
                f"Control updates: {self.control_updates}\n"
            )

            summary.write(
                f"Solver failures: {self.solver_failures}\n"
            )

            summary.write(
                f"10 ms deadline misses: {self.deadline_misses}\n\n"
            )

            if solve.size > 0:
                summary.write(
                    "Solver timing [ms]\n"
                )

                summary.write(
                    f"  mean: {np.mean(solve):.4f}\n"
                )

                summary.write(
                    f"  p95 : {np.percentile(solve, 95):.4f}\n"
                )

                summary.write(
                    f"  max : {np.max(solve):.4f}\n\n"
                )

            summary.write(
                "END-EFFECTOR TRACKING\n"
            )

            summary.write(
                f"  position RMS [m]: {rms(position):.6f}\n"
            )

            summary.write(
                "  position max [m]: "
                f"{np.max(position) if position.size else float('nan'):.6f}\n"
            )

            summary.write(
                f"  velocity RMS [m/s]: {rms(velocity):.6f}\n"
            )

            summary.write(
                f"  attitude RMS [deg]: {rms(attitude):.6f}\n"
            )

            summary.write(
                "  attitude max [deg]: "
                f"{np.max(attitude) if attitude.size else float('nan'):.6f}\n\n"
            )

            summary.write(
                "Actuator usage\n"
            )

            summary.write(
                f"  max thrust [N]: {self.max_thrust_N:.6f}\n"
            )

            summary.write(
                "  max |alpha_cmd| [deg]: "
                f"{np.degrees(self.max_abs_alpha_cmd_rad):.6f}\n\n"
            )

            summary.write(
                f"CSV telemetry:\n{self.csv_path}\n"
            )

        self.closed = True


def main() -> None:

    trajectory = (
        QuinticEffectorPoseTrajectory(
            TRAJECTORY_WAYPOINTS
        )
    )

    print()
    print("=" * 80)
    print("END-EFFECTOR-CENTRIC NMPC CLOSED-LOOP RUN")
    print("=" * 80)

    print()
    print(
        "Tool offset B frame : "
        f"{DEFAULT_END_EFFECTOR.position_B} m"
    )

    print(
        f"Trajectory duration  : "
        f"{trajectory.end_time_s:.1f} s"
    )

    print(
        f"NMPC period          : "
        f"{1000.0 * CONTROL_PERIOD:.1f} ms "
        f"({1.0 / CONTROL_PERIOD:.0f} Hz)"
    )

    model = mujoco.MjModel.from_xml_path(
        str(
            XML_PATH
        )
    )

    data = mujoco.MjData(
        model
    )

    physics_dt = float(
        model.opt.timestep
    )

    physics_steps_per_control = int(
        round(
            CONTROL_PERIOD
            / physics_dt
        )
    )

    if not np.isclose(
        physics_steps_per_control
        * physics_dt,
        CONTROL_PERIOD,
        atol=1e-12,
    ):
        raise RuntimeError(
            "CONTROL_PERIOD must be an integer multiple "
            "of the MuJoCo timestep."
        )

    if not JSON_FILE.exists():
        raise FileNotFoundError(
            "Effector NMPC solver JSON was not found:\n"
            f"{JSON_FILE}\n"
            "Run experiments/nmpc/build_effector_nmpc.py first."
        )

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "end_effector",
    )

    if site_id < 0:
        raise RuntimeError(
            "MJCF is missing the end_effector site."
        )

    print(
        f"MuJoCo timestep      : "
        f"{1000.0 * physics_dt:.1f} ms "
        f"({1.0 / physics_dt:.0f} Hz)"
    )

    print(
        f"Physics steps/NMPC   : "
        f"{physics_steps_per_control}"
    )

    print()
    print(
        "Loading already-built end-effector solver..."
    )

    controller = (
        EffectorNmpcController(
            build_directory=BUILD_DIR,
            json_file=JSON_FILE,
            generate=False,
            build=False,
            verbose=False,
        )
    )

    alpha = (
        reset_vehicle_to_first_effector_reference(
            model,
            data,
            trajectory,
        )
    )

    thrust = np.full(
        4,
        HOVER_THRUST,
        dtype=float,
    )

    alpha_cmd = np.zeros(
        4,
        dtype=float,
    )

    logger = EffectorLogger()

    completed = False

    viewer_period = (
        1.0 / VIEWER_HZ
    )

    terminal_period = (
        1.0 / TERMINAL_HZ
    )

    next_viewer_time = 0.0
    next_terminal_time = 0.0

    print()
    print(
        "Starting effector-centric closed loop..."
    )

    try:

        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:

            viewer.cam.distance = 4.2
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -22.0
            viewer.cam.lookat[:] = np.array(
                [
                    0.0,
                    0.0,
                    1.4,
                ]
            )

            while (
                viewer.is_running()
                and data.time
                < trajectory.end_time_s
            ):

                wall_cycle_start = (
                    time.perf_counter()
                )

                current_time = float(
                    data.time
                )

                state = (
                    full_state_from_mujoco(
                        data,
                        alpha,
                    )
                )

                effector = (
                    body_to_end_effector_kinematics(
                        position_W=state[0:3],
                        velocity_W=state[3:6],
                        quaternion_WB=state[6:10],
                        omega_B=state[10:13],
                    )
                )

                reference = (
                    trajectory.sample(
                        current_time
                    )
                )

                try:
                    command = (
                        controller.solve(
                            state=state,
                            trajectory=trajectory,
                            time_s=current_time,
                        )
                    )

                except RuntimeError as exc:
                    logger.note_solver_failure()

                    print()
                    print(
                        "NMPC FAILURE:"
                    )
                    print(exc)

                    break

                thrust = (
                    command.thrust
                )

                alpha_cmd = (
                    command.alpha_cmd
                )

                logger.record(
                    time_s=current_time,
                    state=state,
                    effector=effector,
                    reference=reference,
                    thrust=thrust,
                    alpha_cmd=alpha_cmd,
                    solve_time_s=command.solve_time_s,
                    solver_status=command.status,
                )

                if (
                    current_time
                    + 1e-12
                    >= next_terminal_time
                ):

                    position_error = float(
                        np.linalg.norm(
                            effector.position_W
                            - reference.position_W
                        )
                    )

                    attitude_error = (
                        quaternion_error_deg(
                            effector.quaternion_WE,
                            reference.quaternion_WE,
                        )
                    )

                    print(
                        f"t={current_time:6.2f}s | "
                        f"pE={np.round(effector.position_W, 3)} | "
                        f"pEref={np.round(reference.position_W, 3)} | "
                        f"ep={position_error:.3f}m | "
                        f"eq={attitude_error:.2f}deg | "
                        f"solve={1000.0 * command.solve_time_s:.3f}ms"
                    )

                    next_terminal_time += (
                        terminal_period
                    )

                # Zero-order hold over five 2 ms plant steps.
                for _ in range(
                    physics_steps_per_control
                ):
                    alpha = (
                        step_rigid_plant(
                            model,
                            data,
                            alpha,
                            thrust,
                            alpha_cmd,
                        )
                    )

                if (
                    current_time
                    + 1e-12
                    >= next_viewer_time
                ):

                    reference_after = (
                        trajectory.sample(
                            float(
                                data.time
                            )
                        )
                    )

                    with viewer.lock():
                        update_reference_marker(
                            viewer,
                            reference_after.position_W,
                        )

                    viewer.sync()

                    next_viewer_time += (
                        viewer_period
                    )

                wall_elapsed = (
                    time.perf_counter()
                    - wall_cycle_start
                )

                sleep_time = (
                    CONTROL_PERIOD
                    - wall_elapsed
                )

                if sleep_time > 0.0:
                    time.sleep(
                        sleep_time
                    )

            completed = (
                viewer.is_running()
                and data.time
                >= trajectory.end_time_s
            )

    finally:

        logger.close(
            completed=completed,
            final_time_s=float(
                data.time
            ),
        )

        print()
        print("=" * 80)
        print("END-EFFECTOR CLOSED-LOOP RUN FINISHED")
        print("=" * 80)

        print()
        print("Saved telemetry:")
        print(logger.csv_path)

        print()
        print("Saved summary:")
        print(logger.summary_path)

        if not completed:
            print()
            print(
                "Run ended before the full trajectory completed."
            )


if __name__ == "__main__":
    main()
