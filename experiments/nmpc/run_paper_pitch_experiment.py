from __future__ import annotations

import csv
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
# from tiltable_quadrotor_rl.control.nmpc_effector_paper_pitch_trajectory import (
#     PaperStyleFixedEffectorPitchTrajectory,
# )
from tiltable_quadrotor_rl.control.nmpc_effector_paper_maneuvers import (
    PaperStyleFixedEffectorTrajectory,
)
from tiltable_quadrotor_rl.control.nmpc_effector_reference import (
    end_effector_reference_to_body_reference,
    make_effector_stage_reference,
)
from tiltable_quadrotor_rl.control.nmpc_ocp import (
    CONTROL_PERIOD,
)
from tiltable_quadrotor_rl.control.nmpc_reference import (
    P_ALPHA,
    P_THRUST,
    feedforward_wrench_body,
)
from tiltable_quadrotor_rl.control.allocation import (
    virtual_to_rotor_commands,
    wrench_to_virtual,
)
from tiltable_quadrotor_rl.robot.parameters import (
    HOVER_THRUST,
)
from tiltable_quadrotor_rl.robot.rigid_plant import (
    full_state_from_mujoco,
    step_rigid_plant,
)

from tiltable_quadrotor_rl.visualization.servo_overlay import (
    update_servo_overlay,
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

TOOL_POSITION_W = np.array(
    [
        0.0,
        0.0,
        1.5,
    ],
    dtype=float,
)

VIEWER_HZ = 50.0
TERMINAL_HZ = 1.0


def quaternion_error_deg(
    actual: np.ndarray,
    reference: np.ndarray,
) -> float:

    actual = np.asarray(
        actual,
        dtype=float,
    ).reshape(4)

    reference = np.asarray(
        reference,
        dtype=float,
    ).reshape(4)

    actual = (
        actual
        / np.linalg.norm(
            actual
        )
    )

    reference = (
        reference
        / np.linalg.norm(
            reference
        )
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


def reset_vehicle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: PaperStyleFixedEffectorTrajectory,
) -> np.ndarray:

    mujoco.mj_resetData(
        model,
        data,
    )

    reference = trajectory.sample(
        0.0
    )

    body_reference = (
        end_effector_reference_to_body_reference(
            reference,
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


def update_fixed_tool_marker(
    viewer,
) -> None:

    marker = (
        viewer.user_scn.geoms[0]
    )

    mujoco.mjv_initGeom(
        marker,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array(
            [
                0.042,
                0.042,
                0.042,
            ],
            dtype=float,
        ),
        TOOL_POSITION_W,
        np.eye(
            3
        ).reshape(-1),
        np.array(
            [
                0.1,
                1.0,
                0.1,
                0.65,
            ],
            dtype=float,
        ),
    )

    viewer.user_scn.ngeom = 1


class Logger:
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
            / f"paper_pitch_closed_loop_{timestamp}.csv"
        )

        self.summary_path = (
            LOG_DIR
            / f"paper_pitch_closed_loop_{timestamp}_summary.txt"
        )

        self._file = open(
            self.csv_path,
            "w",
            newline="",
        )

        self._fieldnames = [
            "time_s",
            "pitch_ref_deg",
            "pitch_rate_ref_degps",

            "solve_time_ms",
            "solver_status",
            "deadline_missed",

            "tool_error_m",
            "attitude_error_deg",

            "body_px_m",
            "body_py_m",
            "body_pz_m",

            "qw",
            "qx",
            "qy",
            "qz",

            "omega_x_radps",
            "omega_y_radps",
            "omega_z_radps",
        ]

        for rotor in range(
            1,
            5,
        ):
            self._fieldnames.extend(
                [
                    f"raw_thrust_ref_{rotor}_N",
                    f"raw_alpha_ref_{rotor}_deg",

                    f"refined_thrust_ref_{rotor}_N",
                    f"refined_alpha_ref_{rotor}_deg",

                    f"actual_alpha_{rotor}_deg",
                    f"alpha_cmd_{rotor}_deg",
                    f"thrust_cmd_{rotor}_N",
                ]
            )

        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fieldnames,
        )

        self._writer.writeheader()

        self.solve_times_ms: list[float] = []
        self.tool_errors: list[float] = []
        self.attitude_errors: list[float] = []

        self.max_abs_actual_alpha_deg = 0.0
        self.max_abs_alpha_cmd_deg = 0.0
        self.max_thrust_cmd_N = 0.0

        self.control_updates = 0
        self.deadline_misses = 0
        self.solver_failures = 0
        self.closed = False

    def record(
        self,
        *,
        time_s: float,
        pitch_ref_deg: float,
        pitch_rate_ref_degps: float,
        state: np.ndarray,
        effector,
        reference,
        raw_thrust_ref: np.ndarray,
        raw_alpha_ref: np.ndarray,
        refined_thrust_ref: np.ndarray,
        refined_alpha_ref: np.ndarray,
        command,
    ) -> None:

        tool_error = float(
            np.linalg.norm(
                effector.position_W
                - TOOL_POSITION_W
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
            * command.solve_time_s
        )

        deadline_missed = (
            command.solve_time_s
            > CONTROL_PERIOD
        )

        actual_alpha_deg = np.degrees(
            state[13:17]
        )

        alpha_cmd_deg = np.degrees(
            command.alpha_cmd
        )

        raw_alpha_ref_deg = np.degrees(
            raw_alpha_ref
        )

        refined_alpha_ref_deg = np.degrees(
            refined_alpha_ref
        )

        row = {
            "time_s": time_s,
            "pitch_ref_deg": pitch_ref_deg,
            "pitch_rate_ref_degps": pitch_rate_ref_degps,

            "solve_time_ms": solve_time_ms,
            "solver_status": command.status,
            "deadline_missed": int(
                deadline_missed
            ),

            "tool_error_m": tool_error,
            "attitude_error_deg": attitude_error,

            "body_px_m": state[0],
            "body_py_m": state[1],
            "body_pz_m": state[2],

            "qw": state[6],
            "qx": state[7],
            "qy": state[8],
            "qz": state[9],

            "omega_x_radps": state[10],
            "omega_y_radps": state[11],
            "omega_z_radps": state[12],
        }

        for rotor in range(
            4
        ):
            index = (
                rotor + 1
            )

            row[
                f"raw_thrust_ref_{index}_N"
            ] = raw_thrust_ref[rotor]

            row[
                f"raw_alpha_ref_{index}_deg"
            ] = raw_alpha_ref_deg[rotor]

            row[
                f"refined_thrust_ref_{index}_N"
            ] = refined_thrust_ref[rotor]

            row[
                f"refined_alpha_ref_{index}_deg"
            ] = refined_alpha_ref_deg[rotor]

            row[
                f"actual_alpha_{index}_deg"
            ] = actual_alpha_deg[rotor]

            row[
                f"alpha_cmd_{index}_deg"
            ] = alpha_cmd_deg[rotor]

            row[
                f"thrust_cmd_{index}_N"
            ] = command.thrust[rotor]

        self._writer.writerow(
            row
        )

        self.solve_times_ms.append(
            solve_time_ms
        )

        self.tool_errors.append(
            tool_error
        )

        self.attitude_errors.append(
            attitude_error
        )

        self.max_abs_actual_alpha_deg = max(
            self.max_abs_actual_alpha_deg,
            float(
                np.max(
                    np.abs(
                        actual_alpha_deg
                    )
                )
            ),
        )

        self.max_abs_alpha_cmd_deg = max(
            self.max_abs_alpha_cmd_deg,
            float(
                np.max(
                    np.abs(
                        alpha_cmd_deg
                    )
                )
            ),
        )

        self.max_thrust_cmd_N = max(
            self.max_thrust_cmd_N,
            float(
                np.max(
                    command.thrust
                )
            ),
        )

        self.control_updates += 1

        if deadline_missed:
            self.deadline_misses += 1

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

        tool = np.asarray(
            self.tool_errors,
            dtype=float,
        )

        attitude = np.asarray(
            self.attitude_errors,
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
                        values
                        * values
                    )
                )
            )

        with open(
            self.summary_path,
            "w",
        ) as summary:

            summary.write(
                "PAPER-STYLE FIXED-EFFECTOR PITCH CLOSED LOOP\n"
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
                "Tracking\n"
            )

            summary.write(
                f"  tool position RMS [m]: {rms(tool):.6f}\n"
            )

            summary.write(
                "  tool position max [m]: "
                f"{np.max(tool) if tool.size else float('nan'):.6f}\n"
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
                "  max |actual alpha| [deg]: "
                f"{self.max_abs_actual_alpha_deg:.6f}\n"
            )

            summary.write(
                "  max |alpha_cmd| [deg]: "
                f"{self.max_abs_alpha_cmd_deg:.6f}\n"
            )

            summary.write(
                "  max thrust command [N]: "
                f"{self.max_thrust_cmd_N:.6f}\n\n"
            )

            summary.write(
                f"CSV telemetry:\n{self.csv_path}\n"
            )

        self.closed = True


def main() -> None:

    trajectory = PaperStyleFixedEffectorTrajectory(
        position_W=np.array(
            [0.0, 0.0, 1.5],
            dtype=float,
        ),
        maneuver="roll",
    )

    print()
    print("=" * 80)
    print("PAPER-STYLE FIXED-END-EFFECTOR PITCH CLOSED LOOP")
    print("=" * 80)

    print()
    print(
        "Fixed tool position : "
        f"{TOOL_POSITION_W} m"
    )

    print()
    print(
        "Trajectory:"
    )

    print(
        "   0-2 s  : hold 0 deg"
    )

    print(
        "   2-7 s  : 0 -> +180 deg"
    )

    print(
        "   7-10 s : hold +180 deg"
    )

    print(
        "  10-20 s : +180 -> -180 deg (full 360 deg)"
    )

    print(
        "  20-23 s : hold -180 deg"
    )

    print(
        "  23-28 s : -180 -> 0 deg"
    )

    print(
        "  28-31 s : hold 0 deg"
    )

    if not JSON_FILE.exists():
        raise FileNotFoundError(
            "Built effector NMPC solver was not found:\n"
            f"{JSON_FILE}"
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

    controller = EffectorNmpcController(
        build_directory=BUILD_DIR,
        json_file=JSON_FILE,
        generate=False,
        build=False,
        verbose=False,
    )

    alpha = reset_vehicle(
        model,
        data,
        trajectory,
    )

    logger = Logger()

    completed = False

    next_viewer_time = 0.0
    next_terminal_time = 0.0

    try:

        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:

            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -12.0
            viewer.cam.lookat[:] = (
                TOOL_POSITION_W
            )

            while (
                viewer.is_running()
                and data.time
                < trajectory.end_time_s
            ):

                wall_start = (
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

                reference = trajectory.sample(
                    current_time
                )

                (
                    pitch_ref_rad,
                    pitch_rate_ref_radps,
                    _,
                ) = trajectory.pitch_state(
                    current_time
                )

                effector = (
                    body_to_end_effector_kinematics(
                        position_W=state[0:3],
                        velocity_W=state[3:6],
                        quaternion_WB=state[6:10],
                        omega_B=state[10:13],
                    )
                )

                # Reconstruct the stage-0 reference exactly as the
                # controller does, using the CURRENT physical alpha
                # as the continuity seed.
                stage0_reference = (
                    make_effector_stage_reference(
                        reference,
                        previous_effector_quaternion=effector.quaternion_WE,
                        previous_body_quaternion=state[6:10],
                        previous_alpha=state[13:17],
                        transform=DEFAULT_END_EFFECTOR,
                    )
                )

                refined_thrust_ref = (
                    stage0_reference.parameters[
                        P_THRUST
                    ].copy()
                )

                refined_alpha_ref = (
                    stage0_reference.parameters[
                        P_ALPHA
                    ].copy()
                )

                # Raw pseudoinverse reference, before the paper's
                # low-thrust refinement.
                body_reference = (
                    end_effector_reference_to_body_reference(
                        reference,
                        transform=DEFAULT_END_EFFECTOR,
                    )
                )

                wrench_ref = (
                    feedforward_wrench_body(
                        body_reference
                    )
                )

                raw_virtual = np.asarray(
                    wrench_to_virtual(
                        wrench_ref
                    ),
                    dtype=float,
                ).reshape(8)

                (
                    raw_thrust_ref,
                    raw_alpha_ref,
                ) = virtual_to_rotor_commands(
                    raw_virtual
                )

                raw_thrust_ref = np.asarray(
                    raw_thrust_ref,
                    dtype=float,
                ).reshape(4)

                raw_alpha_ref = np.asarray(
                    raw_alpha_ref,
                    dtype=float,
                ).reshape(4)

                try:
                    command = controller.solve(
                        state=state,
                        trajectory=trajectory,
                        time_s=current_time,
                    )

                except RuntimeError as exc:
                    logger.note_solver_failure()

                    print()
                    print(
                        "NMPC FAILURE:"
                    )

                    print(
                        exc
                    )

                    break

                logger.record(
                    time_s=current_time,
                    pitch_ref_deg=float(
                        np.degrees(
                            pitch_ref_rad
                        )
                    ),
                    pitch_rate_ref_degps=float(
                        np.degrees(
                            pitch_rate_ref_radps
                        )
                    ),
                    state=state,
                    effector=effector,
                    reference=reference,
                    raw_thrust_ref=raw_thrust_ref,
                    raw_alpha_ref=raw_alpha_ref,
                    refined_thrust_ref=refined_thrust_ref,
                    refined_alpha_ref=refined_alpha_ref,
                    command=command,
                )

                if (
                    current_time
                    + 1e-12
                    >= next_terminal_time
                ):

                    tool_error_mm = (
                        1000.0
                        * np.linalg.norm(
                            effector.position_W
                            - TOOL_POSITION_W
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
                        f"pitch_ref={np.degrees(pitch_ref_rad):7.1f}deg | "
                        f"tool_err={tool_error_mm:7.2f}mm | "
                        f"att_err={attitude_error:6.2f}deg | "
                        f"alpha={np.round(np.degrees(state[13:17]), 1)} | "
                        f"alpha_ref={np.round(np.degrees(refined_alpha_ref), 1)} | "
                        f"alpha_cmd={np.round(np.degrees(command.alpha_cmd), 1)}"
                    )

                    next_terminal_time += (
                        1.0 / TERMINAL_HZ
                    )

                for _ in range(
                    physics_steps_per_control
                ):
                    alpha = step_rigid_plant(
                        model,
                        data,
                        alpha,
                        command.thrust,
                        command.alpha_cmd,
                    )

                if (
                    current_time
                    + 1e-12
                    >= next_viewer_time
                ):

                    display_state = full_state_from_mujoco(
                        data,
                        alpha,
                    )

                    with viewer.lock():
                        update_fixed_tool_marker(
                            viewer
                        )

                        update_servo_overlay(
                            viewer,
                            body_position_W=display_state[0:3],
                            body_quaternion_WB=display_state[6:10],
                            alpha=display_state[13:17],
                        )

                    viewer.sync()

                    next_viewer_time += (
                        1.0 / VIEWER_HZ
                    )

                elapsed = (
                    time.perf_counter()
                    - wall_start
                )

                sleep_time = (
                    CONTROL_PERIOD
                    - elapsed
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
        print("PAPER-STYLE PITCH EXPERIMENT FINISHED")
        print("=" * 80)

        print()
        print("CSV:")
        print(logger.csv_path)

        print()
        print("Summary:")
        print(logger.summary_path)


if __name__ == "__main__":
    main()
