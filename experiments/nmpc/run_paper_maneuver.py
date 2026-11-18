from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    virtual_to_rotor_commands,
    wrench_to_virtual,
)
from tiltable_quadrotor_rl.control.end_effector import (
    DEFAULT_END_EFFECTOR,
    body_to_end_effector_kinematics,
)
from tiltable_quadrotor_rl.control.nmpc_effector_controller import (
    EffectorNmpcController,
)
from tiltable_quadrotor_rl.control.nmpc_predictive_effector_controller import (
    PredictivePrepositionEffectorNmpcController,
    TimedEffectorNmpcController,
)
from tiltable_quadrotor_rl.control.nmpc_effector_paper_maneuvers import (
    RATE_BIAS_GAIN_M_PER_DEGPS,
    PaperStyleCombinedManeuverTrajectory,
    PaperStyleFixedEffectorTrajectory,
    PaperStyleStaticCartwheelHoldTrajectory,
)
from tiltable_quadrotor_rl.control.nmpc_effector_reference import (
    end_effector_reference_to_body_reference,
    make_effector_stage_reference,
)
from tiltable_quadrotor_rl.control.nmpc_ocp import (
    CONTROL_PERIOD,
    PREDICTION_STEP,
)
from tiltable_quadrotor_rl.control.nmpc_reference import (
    P_ALPHA,
    P_THRUST,
    feedforward_wrench_body,
)
from tiltable_quadrotor_rl.robot.rigid_plant import (
    full_state_from_mujoco,
    step_rigid_plant,
)
from tiltable_quadrotor_rl.visualization.cad_maneuver import (
    CadManeuverVisual,
)
from tiltable_quadrotor_rl.visualization.servo_overlay import (
    update_servo_overlay,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    [0.0, 0.0, 1.5],
    dtype=float,
)

# Fixed main camera for the combined maneuver (frames the +-0.55 m motion envelope about the
# fixed tool point, tool point left of center so the bottom-right inset never covers the aircraft)
# and the independent end-effector close-up inset.
COMBINED_MAIN_CAMERA = dict(
    distance=1.65,
    azimuth=135.0,
    elevation=-12.0,
    lookat=(0.12, 0.10, 1.5),
)
COMBINED_TOOL_CAMERA = dict(
    distance=0.5,
    azimuth=110.0,
    elevation=-10.0,
)
COMBINED_INSET_SCALE = 0.34

VIEWER_HZ = 50.0
TERMINAL_HZ = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a paper-style fixed-end-effector maneuver "
            "with the end-effector NMPC."
        )
    )

    parser.add_argument(
        "--maneuver",
        required=True,
        choices=[
            "pitch",
            "roll",
            "cartwheel",
            "cartwheel-static",
            "yaw",
            "combined",
        ],
        help=(
            "'combined' = one continuous run: hover, 360 roll, 360 pitch, "
            "pitch to horizontal, 360 spin about the tool axis, return."
        ),
    )

    parser.add_argument(
        "--cartwheel-angle",
        type=float,
        choices=[90.0, 180.0, 270.0],
        default=None,
        help=(
            "Static cartwheel angle in degrees. Required only "
            "for --maneuver cartwheel-static."
        ),
    )

    parser.add_argument(
        "--bias-compensation",
        action="store_true",
        help=(
            "Enable empirical rate-proportional fixed-point "
            "bias compensation. Valid only for pitch/roll."
        ),
    )

    parser.add_argument(
        "--predictive-preposition",
        action="store_true",
        help=(
            "Enable predictive low-thrust rotor pre-positioning "
            "inside the NMPC reference horizon."
        ),
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run without the MuJoCo viewer (no real-time pacing)."
        ),
    )

    parser.add_argument(
        "--cad-visual",
        action="store_true",
        help=(
            "Display/render with the CAD visual MJCF and spinning "
            "propellers. Physics and control still run on the original "
            "rigid plant; the CAD model only mirrors its state."
        ),
    )

    parser.add_argument(
        "--render-mp4",
        type=Path,
        default=None,
        help=(
            "Render the maneuver offscreen to this MP4 (implies "
            "--cad-visual and no interactive viewer)."
        ),
    )

    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="MP4 frame rate (simulation-time sampled).",
    )

    parser.add_argument(
        "--video-size",
        type=int,
        nargs=2,
        default=(1920, 1080),
        metavar=("W", "H"),
        help="MP4 resolution.",
    )

    args = parser.parse_args()

    if args.render_mp4 is not None:
        args.cad_visual = True

    if args.cad_visual and args.headless and args.render_mp4 is None:
        parser.error(
            "--cad-visual with --headless needs --render-mp4"
        )

    if (
        args.maneuver in (
            "cartwheel",
            "cartwheel-static",
            "yaw",
        )
        and args.bias_compensation
    ):
        parser.error(
            "--bias-compensation is not calibrated "
            "for cartwheel/yaw maneuvers."
        )

    if (
        args.maneuver == "cartwheel-static"
        and args.cartwheel_angle is None
    ):
        parser.error(
            "--cartwheel-angle is required for "
            "--maneuver cartwheel-static."
        )

    if (
        args.maneuver != "cartwheel-static"
        and args.cartwheel_angle is not None
    ):
        parser.error(
            "--cartwheel-angle is only valid with "
            "--maneuver cartwheel-static."
        )

    return args


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

    actual = actual / np.linalg.norm(actual)
    reference = reference / np.linalg.norm(reference)

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
            2.0 * np.arccos(dot)
        )
    )


def reset_vehicle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: (
        PaperStyleFixedEffectorTrajectory
        | PaperStyleCombinedManeuverTrajectory
        | PaperStyleStaticCartwheelHoldTrajectory
    ),
) -> np.ndarray:
    """
    Reset the rigid body to the first trajectory point.

    The external servo state is initialized from the same stage-0
    actuator reference used by the NMPC reference generator.

    This is important for the cartwheel: its initial body attitude is
    +90 deg roll, so alpha=0 is not a hover-equilibrium actuator
    geometry. Starting the external servo state from the stage-0
    refined reference avoids injecting an artificial servo transient
    before the cartwheel even begins.

    For the existing level pitch/roll tests this remains effectively
    the normal hover initialization.
    """
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

    initial_alpha_seed = np.zeros(
        4,
        dtype=float,
    )

    stage0_reference = (
        make_effector_stage_reference(
            reference,
            previous_effector_quaternion=(
                reference.quaternion_WE
            ),
            previous_body_quaternion=(
                body_reference.quaternion_WB
            ),
            previous_alpha=initial_alpha_seed,
            transform=DEFAULT_END_EFFECTOR,
        )
    )

    return (
        stage0_reference.parameters[
            P_ALPHA
        ].copy()
    )


def update_fixed_tool_marker(
    viewer,
) -> None:
    marker = viewer.user_scn.geoms[0]

    mujoco.mjv_initGeom(
        marker,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array(
            [0.042, 0.042, 0.042],
            dtype=float,
        ),
        TOOL_POSITION_W,
        np.eye(3).reshape(-1),
        np.array(
            [0.1, 1.0, 0.1, 0.65],
            dtype=float,
        ),
    )

    viewer.user_scn.ngeom = 1


class Logger:
    def __init__(
        self,
        *,
        maneuver: str,
        bias_compensation: bool,
        trajectory: (
        PaperStyleFixedEffectorTrajectory
        | PaperStyleStaticCartwheelHoldTrajectory
    ),
    ) -> None:
        self.maneuver = maneuver
        self.bias_compensation = (
            bias_compensation
        )
        self.trajectory = trajectory

        LOG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        suffix = (
            "_bias_comp"
            if bias_compensation
            else ""
        )

        if getattr(
            trajectory,
            "_predictive_preposition_enabled",
            False,
        ):
            suffix += (
                "_predictive"
            )

        file_label = (
            maneuver.replace(
                "-",
                "_",
            )
        )

        if isinstance(
            trajectory,
            PaperStyleStaticCartwheelHoldTrajectory,
        ):
            file_label = (
                f"{file_label}_"
                f"{trajectory.cartwheel_angle_deg:03.0f}deg"
            )

        self.csv_path = (
            LOG_DIR
            / f"paper_{file_label}_closed_loop{suffix}_{timestamp}.csv"
        )

        self.summary_path = (
            LOG_DIR
            / f"paper_{file_label}_closed_loop{suffix}_{timestamp}_summary.txt"
        )

        self._file = open(
            self.csv_path,
            "w",
            newline="",
        )

        self._fieldnames = [
            "time_s",
            "angle_ref_deg",
            "angle_rate_ref_degps",
            "angle_accel_ref_degps2",

            "compensation_x_m",
            "compensation_y_m",
            "compensation_z_m",

            "controller_ref_px_m",
            "controller_ref_py_m",
            "controller_ref_pz_m",

            "reference_generation_time_ms",
            "solve_time_ms",
            "total_controller_time_ms",
            "solver_status",
            "solver_deadline_missed",
            "deadline_missed",

            "preposition_event_count",
            "preposition_attempted_stage_count",
            "preposition_applied_stage_count",
            "preposition_fallback_stage_count",
            "preposition_max_wrench_residual",

            "tool_error_m",
            "controller_reference_error_m",
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

                    f"preposition_event_{rotor}",
                    f"preposition_low_start_preview_s_{rotor}",
                    f"preposition_reengage_preview_s_{rotor}",
                    f"preposition_start_alpha_{rotor}_deg",
                    f"preposition_target_alpha_{rotor}_deg",
                    f"preposition_applied_stages_{rotor}",
                    f"preposition_fallback_stages_{rotor}",
                    f"preposition_first_modified_stage_{rotor}",
                    f"preposition_first_modified_alpha_{rotor}_deg",
                ]
            )

        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fieldnames,
        )
        self._writer.writeheader()

        self.times_s: list[float] = []
        self.reference_generation_times_ms: list[float] = []
        self.solve_times_ms: list[float] = []
        self.total_controller_times_ms: list[float] = []

        self.tool_errors: list[float] = []
        self.controller_reference_errors: list[float] = []
        self.attitude_errors: list[float] = []

        self.max_abs_actual_alpha_deg = 0.0
        self.max_abs_alpha_cmd_deg = 0.0
        self.max_thrust_cmd_N = 0.0

        self.control_updates = 0
        self.deadline_misses = 0
        self.solver_deadline_misses = 0
        self.solver_failures = 0

        self.preposition_cycles_with_events = 0
        self.preposition_cycles_with_applied_stages = 0
        self.preposition_total_attempted_stages = 0
        self.preposition_total_applied_stages = 0
        self.preposition_total_fallback_stages = 0
        self.preposition_max_wrench_residual = 0.0

        self.closed = False

    def record(
        self,
        *,
        time_s: float,
        state: np.ndarray,
        effector,
        reference,
        raw_thrust_ref: np.ndarray,
        raw_alpha_ref: np.ndarray,
        refined_thrust_ref: np.ndarray,
        refined_alpha_ref: np.ndarray,
        command,
        reference_generation_time_s: float,
        total_controller_time_s: float,
        predictive_diagnostics,
    ) -> None:
        kinematics = (
            self.trajectory.rotation_kinematics(
                time_s
            )
        )

        compensation = (
            self.trajectory.bias_compensation_state(
                time_s
            )
        )

        tool_error = float(
            np.linalg.norm(
                effector.position_W
                - TOOL_POSITION_W
            )
        )

        controller_reference_error = float(
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

        reference_generation_time_ms = (
            1000.0
            * float(
                reference_generation_time_s
            )
        )

        solve_time_ms = (
            1000.0
            * command.solve_time_s
        )

        total_controller_time_ms = (
            1000.0
            * float(
                total_controller_time_s
            )
        )

        solver_deadline_missed = (
            command.solve_time_s
            > CONTROL_PERIOD
        )

        # This is the real controller deadline metric:
        # complete controller.solve(...) wall time, including reference
        # horizon generation, parameter updates, warm-start handling,
        # acados and wrapper overhead.
        deadline_missed = (
            total_controller_time_s
            > CONTROL_PERIOD
        )

        if predictive_diagnostics is None:
            preposition_event_count = 0
            preposition_attempted_stage_count = 0
            preposition_applied_stage_count = 0
            preposition_fallback_stage_count = 0
            preposition_max_wrench_residual = 0.0
        else:
            preposition_event_count = len(
                predictive_diagnostics.events
            )
            preposition_attempted_stage_count = (
                predictive_diagnostics.attempted_stage_count
            )
            preposition_applied_stage_count = (
                predictive_diagnostics.applied_stage_count
            )
            preposition_fallback_stage_count = (
                predictive_diagnostics.fallback_stage_count
            )
            preposition_max_wrench_residual = (
                predictive_diagnostics.maximum_wrench_residual
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
            "angle_ref_deg": np.degrees(
                kinematics.angle_rad
            ),
            "angle_rate_ref_degps": np.degrees(
                kinematics.rate_radps
            ),
            "angle_accel_ref_degps2": np.degrees(
                kinematics.acceleration_radps2
            ),

            "compensation_x_m":
                compensation.position_offset_W[0],
            "compensation_y_m":
                compensation.position_offset_W[1],
            "compensation_z_m":
                compensation.position_offset_W[2],

            "controller_ref_px_m":
                reference.position_W[0],
            "controller_ref_py_m":
                reference.position_W[1],
            "controller_ref_pz_m":
                reference.position_W[2],

            "reference_generation_time_ms":
                reference_generation_time_ms,
            "solve_time_ms": solve_time_ms,
            "total_controller_time_ms":
                total_controller_time_ms,
            "solver_status": command.status,
            "solver_deadline_missed": int(
                solver_deadline_missed
            ),
            "deadline_missed": int(
                deadline_missed
            ),

            "preposition_event_count":
                preposition_event_count,
            "preposition_attempted_stage_count":
                preposition_attempted_stage_count,
            "preposition_applied_stage_count":
                preposition_applied_stage_count,
            "preposition_fallback_stage_count":
                preposition_fallback_stage_count,
            "preposition_max_wrench_residual":
                preposition_max_wrench_residual,

            "tool_error_m": tool_error,
            "controller_reference_error_m":
                controller_reference_error,
            "attitude_error_deg":
                attitude_error,

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

        for rotor in range(4):
            index = rotor + 1

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

            if predictive_diagnostics is None:
                event = None
                applied_stages = 0
                fallback_stages = 0
                first_modified_stage = None
                first_modified_alpha = None
            else:
                event = (
                    predictive_diagnostics.first_event_for_rotor(
                        rotor
                    )
                )

                applied_stages = (
                    predictive_diagnostics
                    .applied_stage_count_by_rotor[
                        rotor
                    ]
                )

                fallback_stages = (
                    predictive_diagnostics
                    .fallback_stage_count_by_rotor[
                        rotor
                    ]
                )

                first_modified_stage = (
                    predictive_diagnostics
                    .first_modified_stage_by_rotor[
                        rotor
                    ]
                )

                first_modified_alpha = (
                    predictive_diagnostics
                    .first_modified_alpha_rad_by_rotor[
                        rotor
                    ]
                )

            row[
                f"preposition_event_{index}"
            ] = int(
                event is not None
            )

            row[
                f"preposition_low_start_preview_s_{index}"
            ] = (
                float("nan")
                if event is None
                else event.low_start_stage
                * PREDICTION_STEP
            )

            row[
                f"preposition_reengage_preview_s_{index}"
            ] = (
                float("nan")
                if event is None
                else event.reengage_stage
                * PREDICTION_STEP
            )

            row[
                f"preposition_start_alpha_{index}_deg"
            ] = (
                float("nan")
                if event is None
                else np.degrees(
                    event.start_alpha_rad
                )
            )

            row[
                f"preposition_target_alpha_{index}_deg"
            ] = (
                float("nan")
                if event is None
                else np.degrees(
                    event.target_alpha_rad
                )
            )

            row[
                f"preposition_applied_stages_{index}"
            ] = applied_stages

            row[
                f"preposition_fallback_stages_{index}"
            ] = fallback_stages

            row[
                f"preposition_first_modified_stage_{index}"
            ] = (
                -1
                if first_modified_stage is None
                else first_modified_stage
            )

            row[
                f"preposition_first_modified_alpha_{index}_deg"
            ] = (
                float("nan")
                if first_modified_alpha is None
                else np.degrees(
                    first_modified_alpha
                )
            )

        self._writer.writerow(
            row
        )

        self.times_s.append(
            float(time_s)
        )

        self.reference_generation_times_ms.append(
            reference_generation_time_ms
        )

        self.solve_times_ms.append(
            solve_time_ms
        )

        self.total_controller_times_ms.append(
            total_controller_time_ms
        )

        self.tool_errors.append(
            tool_error
        )

        self.controller_reference_errors.append(
            controller_reference_error
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

        if solver_deadline_missed:
            self.solver_deadline_misses += 1

        if deadline_missed:
            self.deadline_misses += 1

        if predictive_diagnostics is not None:
            if predictive_diagnostics.events:
                self.preposition_cycles_with_events += 1

            if (
                predictive_diagnostics.applied_stage_count
                > 0
            ):
                self.preposition_cycles_with_applied_stages += 1

            self.preposition_total_attempted_stages += (
                predictive_diagnostics.attempted_stage_count
            )

            self.preposition_total_applied_stages += (
                predictive_diagnostics.applied_stage_count
            )

            self.preposition_total_fallback_stages += (
                predictive_diagnostics.fallback_stage_count
            )

            self.preposition_max_wrench_residual = max(
                self.preposition_max_wrench_residual,
                float(
                    predictive_diagnostics.maximum_wrench_residual
                ),
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

        times = np.asarray(
            self.times_s,
            dtype=float,
        )

        reference_generation = np.asarray(
            self.reference_generation_times_ms,
            dtype=float,
        )

        solve = np.asarray(
            self.solve_times_ms,
            dtype=float,
        )

        total_controller = np.asarray(
            self.total_controller_times_ms,
            dtype=float,
        )

        tool = np.asarray(
            self.tool_errors,
            dtype=float,
        )

        controller_reference = np.asarray(
            self.controller_reference_errors,
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
                        values * values
                    )
                )
            )

        title = (
            "PAPER-STYLE FIXED-EFFECTOR "
            f"{self.maneuver.upper()} CLOSED LOOP"
        )

        with open(
            self.summary_path,
            "w",
        ) as summary:
            summary.write(
                title + "\n"
            )
            summary.write(
                "=" * 72
                + "\n\n"
            )

            summary.write(
                f"Maneuver: {self.maneuver}\n"
            )
            summary.write(
                "Predictive low-thrust pre-positioning: "
                f"{getattr(self.trajectory, '_predictive_preposition_enabled', False)}\n"
            )

            if isinstance(
                self.trajectory,
                PaperStyleStaticCartwheelHoldTrajectory,
            ):
                summary.write(
                    "Static cartwheel angle [deg]: "
                    f"{self.trajectory.cartwheel_angle_deg:.3f}\n"
                )
            summary.write(
                "Smooth angular-rate profile: True\n"
            )
            summary.write(
                "Maximum angular rate [deg/s]: "
                f"{self.trajectory.max_angular_rate_degps:.3f}\n"
            )
            summary.write(
                "Angular-rate ramp duration [s]: "
                f"{self.trajectory.angular_rate_ramp_duration_s:.3f}\n"
            )

            if self.maneuver in (
                "cartwheel",
                "cartwheel-static",
            ):
                summary.write(
                    "Initial roll [deg]: 90.000\n"
                )
                summary.write(
                    "Rotation axis: body/tool Z\n"
                )

            summary.write(
                "Rate-bias compensation enabled: "
                f"{self.bias_compensation}\n"
            )
            summary.write(
                "Rate-bias gain [m/(deg/s)]: "
                f"{RATE_BIAS_GAIN_M_PER_DEGPS:.8f}\n"
            )
            summary.write(
                "Maximum compensation magnitude [m]: "
                f"{self.trajectory.maximum_compensation_m:.6f}\n\n"
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
                "10 ms acados-only deadline misses: "
                f"{self.solver_deadline_misses}\n"
            )
            summary.write(
                "10 ms TOTAL-controller deadline misses: "
                f"{self.deadline_misses}\n\n"
            )

            def write_timing_block(
                title: str,
                values: np.ndarray,
            ) -> None:
                if values.size == 0:
                    return

                summary.write(
                    title + "\n"
                )
                summary.write(
                    f"  mean: {np.mean(values):.4f}\n"
                )
                summary.write(
                    f"  p95 : {np.percentile(values, 95):.4f}\n"
                )
                summary.write(
                    f"  max : {np.max(values):.4f}\n\n"
                )

            write_timing_block(
                "Reference-horizon generation timing [ms]",
                reference_generation,
            )

            write_timing_block(
                "Acados solve timing [ms]",
                solve,
            )

            write_timing_block(
                "TOTAL controller.solve() timing [ms]",
                total_controller,
            )

            if getattr(
                self.trajectory,
                "_predictive_preposition_enabled",
                False,
            ):
                summary.write(
                    "Predictive pre-position diagnostics\n"
                )
                summary.write(
                    "  control cycles with preview events: "
                    f"{self.preposition_cycles_with_events}\n"
                )
                summary.write(
                    "  control cycles with applied preview stages: "
                    f"{self.preposition_cycles_with_applied_stages}\n"
                )
                summary.write(
                    "  total attempted preview stages: "
                    f"{self.preposition_total_attempted_stages}\n"
                )
                summary.write(
                    "  total applied preview stages: "
                    f"{self.preposition_total_applied_stages}\n"
                )
                summary.write(
                    "  total full-stage fallbacks: "
                    f"{self.preposition_total_fallback_stages}\n"
                )
                summary.write(
                    "  max exact-wrench residual: "
                    f"{self.preposition_max_wrench_residual:.3e}\n\n"
                )

            summary.write(
                "Tracking relative to TRUE fixed tool point\n"
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

            if isinstance(
                self.trajectory,
                PaperStyleStaticCartwheelHoldTrajectory,
            ):
                steady_mask = (
                    times
                    >= self.trajectory.steady_state_start_s
                )

                steady_tool = tool[
                    steady_mask
                ]
                steady_attitude = attitude[
                    steady_mask
                ]

                summary.write(
                    "Static steady-state tracking "
                    f"(t >= {self.trajectory.steady_state_start_s:.1f} s)\n"
                )
                summary.write(
                    "  tool position RMS [m]: "
                    f"{rms(steady_tool):.6f}\n"
                )
                summary.write(
                    "  tool position max [m]: "
                    f"{np.max(steady_tool) if steady_tool.size else float('nan'):.6f}\n"
                )
                summary.write(
                    "  attitude RMS [deg]: "
                    f"{rms(steady_attitude):.6f}\n"
                )
                summary.write(
                    "  attitude max [deg]: "
                    f"{np.max(steady_attitude) if steady_attitude.size else float('nan'):.6f}\n\n"
                )

            summary.write(
                "Tracking relative to controller reference\n"
            )
            summary.write(
                "  controller-reference position RMS [m]: "
                f"{rms(controller_reference):.6f}\n"
            )
            summary.write(
                "  controller-reference position max [m]: "
                f"{np.max(controller_reference) if controller_reference.size else float('nan'):.6f}\n\n"
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
    args = parse_args()

    if args.maneuver == "cartwheel-static":
        trajectory = (
            PaperStyleStaticCartwheelHoldTrajectory(
                position_W=TOOL_POSITION_W,
                cartwheel_angle_deg=(
                    args.cartwheel_angle
                ),
            )
        )
    elif args.maneuver == "combined":
        trajectory = (
            PaperStyleCombinedManeuverTrajectory(
                position_W=TOOL_POSITION_W,
                bias_compensation=args.bias_compensation,
            )
        )
    else:
        trajectory = (
            PaperStyleFixedEffectorTrajectory(
                position_W=TOOL_POSITION_W,
                maneuver=args.maneuver,
                bias_compensation=args.bias_compensation,
            )
        )

    # The trajectory remains purely kinematic. This marker is used
    # only so artifact filenames/summaries clearly distinguish the
    # predictive A/B run.
    object.__setattr__(
        trajectory,
        "_predictive_preposition_enabled",
        bool(
            args.predictive_preposition
        ),
    )

    print()
    print("=" * 80)
    print(
        "PAPER-STYLE FIXED-END-EFFECTOR "
        f"{args.maneuver.upper()} CLOSED LOOP"
    )
    print("=" * 80)

    print()
    print(
        "Fixed TRUE tool position : "
        f"{TOOL_POSITION_W} m"
    )

    if args.maneuver in (
        "cartwheel",
        "cartwheel-static",
    ):
        print(
            "Initial orientation       : "
            "+90 deg roll"
        )
        print(
            "Cartwheel rotation axis   : "
            "body/tool Z"
        )

    if args.maneuver == "cartwheel-static":
        print(
            "Static cartwheel angle    : "
            f"{trajectory.cartwheel_angle_deg:.1f} deg"
        )

    print(
        "Bias compensation        : "
        f"{args.bias_compensation}"
    )
    print(
        "Predictive pre-position  : "
        f"{args.predictive_preposition}"
    )
    print(
        "Max angular rate         : "
        f"{trajectory.max_angular_rate_degps:.1f} deg/s"
    )
    print(
        "Angular-rate ramp        : "
        f"{trajectory.angular_rate_ramp_duration_s:.1f} s"
    )

    print()
    print("Trajectory:")
    for line in trajectory.schedule_lines():
        print("  " + line)

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

    controller_class = (
        PredictivePrepositionEffectorNmpcController
        if args.predictive_preposition
        else TimedEffectorNmpcController
    )

    controller = controller_class(
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

    print()
    print(
        "Initial physical alpha    : "
        f"{np.round(np.degrees(alpha), 2)} deg"
    )

    logger = Logger(
        maneuver=args.maneuver,
        bias_compensation=args.bias_compensation,
        trajectory=trajectory,
    )

    completed = False
    control_steps = 0
    next_viewer_time = 0.0
    next_terminal_time = 0.0

    visual = None
    if args.cad_visual:
        combined = args.maneuver == "combined"
        visual = CadManeuverVisual(
            main_camera=(
                COMBINED_MAIN_CAMERA
                if combined
                else None
            ),
            tool_camera=(
                COMBINED_TOOL_CAMERA
                if combined
                else None
            ),
        )

        if args.render_mp4 is not None:
            visual.open_video(
                args.render_mp4,
                fps=args.video_fps,
                size=tuple(args.video_size),
                inset_scale=(
                    COMBINED_INSET_SCALE
                    if combined
                    else 0.3
                ),
            )

    def mirror_to_visual(
        display_state: np.ndarray,
    ) -> np.ndarray:
        display_effector = body_to_end_effector_kinematics(
            position_W=display_state[0:3],
            velocity_W=display_state[3:6],
            quaternion_WB=display_state[6:10],
            omega_B=display_state[10:13],
        )
        visual.set_state(
            display_state,
            target_W=TOOL_POSITION_W,
            effector_W=display_effector.position_W,
        )

        return display_effector.position_W

    def write_visual_frame_if_due() -> None:
        if (
            visual is None
            or not visual.recording
            or not visual.frame_due(
                float(data.time)
            )
        ):
            return

        tool_W = mirror_to_visual(
            full_state_from_mujoco(
                data,
                alpha,
            ),
        )
        visual.write_frame(
            float(data.time),
            tool_W,
        )

    def run_control_loop(
        viewer=None,
    ) -> bool:
        nonlocal alpha
        nonlocal control_steps
        nonlocal next_viewer_time
        nonlocal next_terminal_time

        while (
            data.time
            < trajectory.end_time_s
            and (
                viewer is None
                or viewer.is_running()
            )
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

            reference = (
                trajectory.sample(
                    current_time
                )
            )

            kinematics = (
                trajectory.rotation_kinematics(
                    current_time
                )
            )

            compensation = (
                trajectory.bias_compensation_state(
                    current_time
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
                controller_start = (
                    time.perf_counter()
                )

                command = controller.solve(
                    state=state,
                    trajectory=trajectory,
                    time_s=current_time,
                )

                total_controller_time_s = (
                    time.perf_counter()
                    - controller_start
                )

            except RuntimeError as exc:
                logger.note_solver_failure()

                print()
                print("NMPC FAILURE:")
                print(exc)
                return False

            if (
                visual is not None
                and control_steps == 0
            ):
                write_visual_frame_if_due()

            logger.record(
                time_s=current_time,
                state=state,
                effector=effector,
                reference=reference,
                raw_thrust_ref=raw_thrust_ref,
                raw_alpha_ref=raw_alpha_ref,
                refined_thrust_ref=refined_thrust_ref,
                refined_alpha_ref=refined_alpha_ref,
                command=command,
                reference_generation_time_s=(
                    controller.last_reference_generation_time_s
                ),
                total_controller_time_s=(
                    total_controller_time_s
                ),
                predictive_diagnostics=(
                    controller.last_predictive_diagnostics
                ),
            )

            if (
                current_time
                + 1e-12
                >= next_terminal_time
            ):
                true_tool_error_mm = (
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
                    f"{args.maneuver}_ref="
                    f"{np.degrees(kinematics.angle_rad):7.1f}deg | "
                    f"rate="
                    f"{np.degrees(kinematics.rate_radps):6.1f}deg/s | "
                    f"comp="
                    f"{1000.0 * compensation.position_offset_W}mm | "
                    f"tool_err={true_tool_error_mm:7.2f}mm | "
                    f"att_err={attitude_error:6.2f}deg | "
                    f"ctrl={1000.0 * total_controller_time_s:5.2f}ms"
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

                if visual is not None:
                    visual.advance_props(
                        physics_dt,
                        command.thrust,
                    )
                    write_visual_frame_if_due()

            control_steps += 1

            if (
                viewer is not None
                and current_time
                + 1e-12
                >= next_viewer_time
            ):
                display_state = (
                    full_state_from_mujoco(
                        data,
                        alpha,
                    )
                )

                with viewer.lock():
                    if visual is not None:
                        mirror_to_visual(
                            display_state,
                        )
                        viewer.user_scn.ngeom = 0
                        visual.decorate(
                            viewer.user_scn
                        )
                    else:
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

            if viewer is not None:
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

        return (
            data.time
            >= trajectory.end_time_s
        )

    try:
        if args.headless or args.render_mp4 is not None:
            completed = run_control_loop(
                viewer=None
            )
        else:
            with mujoco.viewer.launch_passive(
                model if visual is None else visual.model,
                data if visual is None else visual.data,
                show_left_ui=True,
                show_right_ui=True,
            ) as viewer:
                if visual is None:
                    viewer.cam.distance = 2.0
                    viewer.cam.azimuth = 135.0
                    viewer.cam.elevation = -12.0
                    viewer.cam.lookat[:] = (
                        TOOL_POSITION_W
                    )
                elif args.maneuver == "combined":
                    viewer.cam.distance = COMBINED_MAIN_CAMERA["distance"]
                    viewer.cam.azimuth = COMBINED_MAIN_CAMERA["azimuth"]
                    viewer.cam.elevation = COMBINED_MAIN_CAMERA["elevation"]
                    viewer.cam.lookat[:] = (
                        visual.cam_main.lookat
                    )
                else:
                    viewer.cam.distance = 2.0
                    viewer.cam.azimuth = 135.0
                    viewer.cam.elevation = -15.0
                    viewer.cam.lookat[:] = (
                        visual.cam_main.lookat
                    )

                completed = run_control_loop(
                    viewer=viewer
                )

    finally:
        if visual is not None:
            visual.close_video()

        logger.close(
            completed=completed,
            final_time_s=float(
                data.time
            ),
        )

        print()
        print("=" * 80)
        print(
            "PAPER-STYLE "
            f"{args.maneuver.upper()} EXPERIMENT FINISHED"
        )
        print("=" * 80)

        print()
        print(
            f"Control steps            : {control_steps}"
        )
        print(
            "Final sim time [s]       : "
            f"{float(data.time):.6f}"
        )
        print(
            "Final qpos               : "
            f"{np.array2string(data.qpos, precision=9, separator=',', max_line_width=200)}"
        )
        print(
            "Final alpha [deg]        : "
            f"{np.array2string(np.degrees(alpha), precision=9, separator=',')}"
        )

        if visual is not None and args.render_mp4 is not None:
            print(
                f"MP4                      : {args.render_mp4} "
                f"({visual.frames} frames)"
            )

        print()
        print("CSV:")
        print(
            logger.csv_path
        )

        print()
        print("Summary:")
        print(
            logger.summary_path
        )


if __name__ == "__main__":
    main()
