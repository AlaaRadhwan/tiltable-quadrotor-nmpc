from __future__ import annotations

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    allocate_exact_wrench,
)

from tiltable_quadrotor_rl.robot.parameters import (
    WEIGHT,
)

from tiltable_quadrotor_rl.robot.rigid_actuation import (
    rotor_wrench_body,
)

from tiltable_quadrotor_rl.robot.rigid_plant import (
    step_rigid_plant,
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

LOG_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "validation"
)


# ============================================================
# Visual-test timing
# ============================================================

SCENARIO_DURATION = 5.0

# Within each full-actuator scenario:
#
#   0.0 - 1.0 s : hover
#   1.0 - 2.0 s : positive excitation
#   2.0 - 3.0 s : negative excitation
#   3.0 - 5.0 s : hover
#
# Every scenario begins from the same vehicle state.
POSITIVE_START = 1.0
NEGATIVE_START = 2.0
RETURN_START = 3.0

VIEWER_UPDATE_HZ = 60.0
TELEMETRY_HZ = 5.0

# Detailed CSV telemetry.
#
# 100 Hz gives one sample every 10 ms, which is fine enough
# to inspect the nominal 48 ms first-order servo transient.
LOG_HZ = 100.0


# ============================================================
# Full actuator-chain scenarios
# ============================================================
#
# Each delta wrench is added to/subtracted from the nominal
# hover wrench:
#
#     [0, 0, mg, 0, 0, 0]
#
# The command path for these scenarios is:
#
# desired wrench
#   -> static allocation
#   -> thrust + alpha_cmd
#   -> first-order servo alpha
#   -> B(alpha) f
#   -> rigid MuJoCo plant
#
# This means X/Y tests can show small cross-axis effects during
# servo transients. That is expected and is part of what these
# scenarios are meant to expose.

SCENARIOS = [
    (
        "1 — PURE HOVER",
        np.zeros(6),
    ),
    (
        "2 — VERTICAL FORCE",
        np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
    ),
    (
        "3 — BODY +X FORCE",
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ),
    (
        "4 — BODY +Y FORCE",
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    ),
    (
        "5 — ROLL TORQUE +Tx",
        np.array([0.0, 0.0, 0.0, 0.045, 0.0, 0.0]),
    ),
    (
        "6 — PITCH TORQUE +Ty",
        np.array([0.0, 0.0, 0.0, 0.0, 0.045, 0.0]),
    ),
    (
        "7 — YAW TORQUE +Tz",
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.045]),
    ),
    (
        "8 — COMBINED 6-D WRENCH",
        np.array([0.60, -0.50, 1.00, 0.025, -0.020, 0.030]),
    ),
    (
        "9 — FINAL HOVER",
        np.zeros(6),
    ),
]


# ============================================================
# Helpers
# ============================================================

def reset_vehicle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> np.ndarray:
    """
    Return every visual scenario to exactly the same state.

    This makes force/torque directions much easier to inspect.

    Returns
    -------
    alpha:
        Four external first-order tilt-servo states [rad].
    """

    mujoco.mj_resetData(model, data)

    data.qpos[0:3] = np.array([0.0, 0.0, 1.5], dtype=float)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    mujoco.mj_forward(model, data)

    return np.zeros(4, dtype=float)


def quaternion_to_euler_deg(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to roll/pitch/yaw [deg]."""

    w, x, y, z = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.degrees([roll, pitch, yaw])


def allocate_wrench(
    desired_wrench: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Allocate one desired body wrench into thrust + tilt command.

    Static allocation is used only as a command generator for
    this validation viewer. The actual plant still uses the
    first-order tilt dynamics.
    """

    thrust, alpha_cmd, info = allocate_exact_wrench(desired_wrench)

    if not info["feasible"]:
        raise RuntimeError(
            "Visual-test wrench is not actuator-feasible:\n"
            f"{desired_wrench}\n"
            f"{info}"
        )

    return thrust, alpha_cmd


def build_commands(
    delta_wrench: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Precompute hover/positive/negative commands."""

    hover = np.array(
        [0.0, 0.0, WEIGHT, 0.0, 0.0, 0.0],
        dtype=float,
    )

    positive = hover + delta_wrench
    negative = hover - delta_wrench

    hover_thrust, hover_alpha_cmd = allocate_wrench(hover)
    positive_thrust, positive_alpha_cmd = allocate_wrench(positive)
    negative_thrust, negative_alpha_cmd = allocate_wrench(negative)

    return {
        "hover": (hover, hover_thrust, hover_alpha_cmd),
        "positive": (positive, positive_thrust, positive_alpha_cmd),
        "negative": (negative, negative_thrust, negative_alpha_cmd),
    }


def rotor_directions_body(alpha: np.ndarray) -> np.ndarray:
    """
    Unit rotor thrust directions in the body frame.

    The geometry matches the allocation model:

        R1 : [ 0,       -sin(a1), cos(a1)]
        R2 : [ sin(a2),  0,       cos(a2)]
        R3 : [ 0,        sin(a3), cos(a3)]
        R4 : [-sin(a4),  0,       cos(a4)]
    """

    s = np.sin(alpha)
    c = np.cos(alpha)

    return np.array(
        [
            [0.0, -s[0], c[0]],
            [s[1], 0.0, c[1]],
            [0.0, s[2], c[2]],
            [-s[3], 0.0, c[3]],
        ],
        dtype=float,
    )


def update_rotor_arrows(
    viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    alpha: np.ndarray,
    thrust: np.ndarray,
) -> None:
    """
    Render actual rotor thrust directions.

    These arrows are purely visual. They introduce no mechanical
    DoFs and do not participate in the MuJoCo dynamics.
    """

    base_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "base",
    )

    if base_id < 0:
        raise RuntimeError("Body 'base' not found.")

    R_WB = data.xmat[base_id].reshape(3, 3)
    directions_body = rotor_directions_body(alpha)

    rotor_site_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"rotor_{i}",
        )
        for i in range(1, 5)
    ]

    if any(site_id < 0 for site_id in rotor_site_ids):
        raise RuntimeError("One or more rotor sites were not found.")

    colors = [
        np.array([1.0, 0.25, 0.25, 1.0]),
        np.array([0.25, 1.0, 0.25, 1.0]),
        np.array([0.25, 0.45, 1.0, 1.0]),
        np.array([1.0, 0.85, 0.20, 1.0]),
    ]

    viewer.user_scn.ngeom = 0

    for i in range(4):
        start = data.site_xpos[rotor_site_ids[i]].copy()
        direction_world = R_WB @ directions_body[i]

        arrow_length = 0.16 + 0.008 * thrust[i]
        end = start + arrow_length * direction_world

        geom = viewer.user_scn.geoms[i]

        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).reshape(-1),
            colors[i],
        )

        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.012,
            start,
            end,
        )

    viewer.user_scn.ngeom = 4


def print_telemetry(
    scenario_name: str,
    local_time: float,
    phase: str,
    data: mujoco.MjData,
    thrust: np.ndarray,
    alpha_cmd: np.ndarray,
    alpha: np.ndarray,
    target_wrench: np.ndarray,
) -> None:
    """Human-readable terminal telemetry."""

    actual_wrench = rotor_wrench_body(thrust, alpha)
    rpy_deg = quaternion_to_euler_deg(data.qpos[3:7])

    print()
    print(
        f"[{scenario_name}] "
        f"t = {local_time:4.1f} s "
        f"phase={phase}"
    )
    print("  p [m]        =", np.round(data.qpos[0:3], 3))
    print("  v [m/s]      =", np.round(data.qvel[0:3], 3))
    print("  RPY [deg]    =", np.round(rpy_deg, 2))
    print("  omega [rad/s]=", np.round(data.qvel[3:6], 3))
    print("  thrust [N]   =", np.round(thrust, 2))
    print("  alpha_cmd °  =", np.round(np.degrees(alpha_cmd), 1))
    print("  alpha actual°=", np.round(np.degrees(alpha), 1))
    print("  target wrench=", np.round(target_wrench, 3))
    print("  actual wrench=", np.round(actual_wrench, 3))


# ============================================================
# Validation logger
# ============================================================

class ValidationLogger:
    """
    Save detailed validation telemetry to CSV and a short summary.

    Terminal output:
        TELEMETRY_HZ

    CSV:
        LOG_HZ
    """

    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.csv_path = (
            LOG_DIR
            / f"rigid_plant_validation_{timestamp}.csv"
        )

        self.summary_path = (
            LOG_DIR
            / f"rigid_plant_validation_{timestamp}_summary.txt"
        )

        self.file = open(
            self.csv_path,
            "w",
            newline="",
        )

        self.fieldnames = [
            "scenario_index",
            "scenario_name",
            "phase",
            "time_s",
            "px_m",
            "py_m",
            "pz_m",
            "vx_mps",
            "vy_mps",
            "vz_mps",
            "qw",
            "qx",
            "qy",
            "qz",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "omega_x_radps",
            "omega_y_radps",
            "omega_z_radps",
            "thrust_1_N",
            "thrust_2_N",
            "thrust_3_N",
            "thrust_4_N",
            "alpha_cmd_1_rad",
            "alpha_cmd_2_rad",
            "alpha_cmd_3_rad",
            "alpha_cmd_4_rad",
            "alpha_1_rad",
            "alpha_2_rad",
            "alpha_3_rad",
            "alpha_4_rad",
            "target_Fx_N",
            "target_Fy_N",
            "target_Fz_N",
            "target_Tx_Nm",
            "target_Ty_Nm",
            "target_Tz_Nm",
            "actual_Fx_N",
            "actual_Fy_N",
            "actual_Fz_N",
            "actual_Tx_Nm",
            "actual_Ty_Nm",
            "actual_Tz_Nm",
        ]

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.fieldnames,
        )
        self.writer.writeheader()

        self.scenario_summaries: list[str] = []
        self.closed = False

    def log(
        self,
        *,
        scenario_index: int,
        scenario_name: str,
        phase: str,
        local_time: float,
        data: mujoco.MjData,
        thrust: np.ndarray,
        alpha_cmd: np.ndarray,
        alpha: np.ndarray,
        target_wrench: np.ndarray,
    ) -> None:
        actual_wrench = rotor_wrench_body(thrust, alpha)
        rpy = quaternion_to_euler_deg(data.qpos[3:7])

        row = {
            "scenario_index": scenario_index,
            "scenario_name": scenario_name,
            "phase": phase,
            "time_s": float(local_time),
            "px_m": float(data.qpos[0]),
            "py_m": float(data.qpos[1]),
            "pz_m": float(data.qpos[2]),
            "vx_mps": float(data.qvel[0]),
            "vy_mps": float(data.qvel[1]),
            "vz_mps": float(data.qvel[2]),
            "qw": float(data.qpos[3]),
            "qx": float(data.qpos[4]),
            "qy": float(data.qpos[5]),
            "qz": float(data.qpos[6]),
            "roll_deg": float(rpy[0]),
            "pitch_deg": float(rpy[1]),
            "yaw_deg": float(rpy[2]),
            "omega_x_radps": float(data.qvel[3]),
            "omega_y_radps": float(data.qvel[4]),
            "omega_z_radps": float(data.qvel[5]),
            "thrust_1_N": float(thrust[0]),
            "thrust_2_N": float(thrust[1]),
            "thrust_3_N": float(thrust[2]),
            "thrust_4_N": float(thrust[3]),
            "alpha_cmd_1_rad": float(alpha_cmd[0]),
            "alpha_cmd_2_rad": float(alpha_cmd[1]),
            "alpha_cmd_3_rad": float(alpha_cmd[2]),
            "alpha_cmd_4_rad": float(alpha_cmd[3]),
            "alpha_1_rad": float(alpha[0]),
            "alpha_2_rad": float(alpha[1]),
            "alpha_3_rad": float(alpha[2]),
            "alpha_4_rad": float(alpha[3]),
            "target_Fx_N": float(target_wrench[0]),
            "target_Fy_N": float(target_wrench[1]),
            "target_Fz_N": float(target_wrench[2]),
            "target_Tx_Nm": float(target_wrench[3]),
            "target_Ty_Nm": float(target_wrench[4]),
            "target_Tz_Nm": float(target_wrench[5]),
            "actual_Fx_N": float(actual_wrench[0]),
            "actual_Fy_N": float(actual_wrench[1]),
            "actual_Fz_N": float(actual_wrench[2]),
            "actual_Tx_Nm": float(actual_wrench[3]),
            "actual_Ty_Nm": float(actual_wrench[4]),
            "actual_Tz_Nm": float(actual_wrench[5]),
        }

        self.writer.writerow(row)

    def add_scenario_summary(
        self,
        scenario_index: int,
        scenario_name: str,
        data: mujoco.MjData,
        alpha: np.ndarray,
    ) -> None:
        rpy = quaternion_to_euler_deg(data.qpos[3:7])

        self.scenario_summaries.append(
            (
                f"Scenario {scenario_index}: {scenario_name}\n"
                f"  final position [m]   = {data.qpos[0:3]}\n"
                f"  final velocity [m/s] = {data.qvel[0:3]}\n"
                f"  final RPY [deg]      = {rpy}\n"
                f"  final omega [rad/s]  = {data.qvel[3:6]}\n"
                f"  final alpha [rad]    = {alpha}\n"
            )
        )

    def close(self, *, completed: bool) -> None:
        if self.closed:
            return

        self.file.flush()
        self.file.close()

        with open(self.summary_path, "w") as summary_file:
            summary_file.write("RIGID PLANT VISUAL VALIDATION SUMMARY\n")
            summary_file.write("=" * 72 + "\n\n")
            summary_file.write(f"Run completed normally: {completed}\n")
            summary_file.write(f"CSV logging rate: {LOG_HZ:.1f} Hz\n")
            summary_file.write(f"Scenario duration: {SCENARIO_DURATION:.3f} s\n")
            summary_file.write(f"Scenario count: {len(SCENARIOS)}\n\n")
            summary_file.write(f"CSV telemetry:\n{self.csv_path}\n\n")

            for summary in self.scenario_summaries:
                summary_file.write(summary)
                summary_file.write("\n")

        self.closed = True


# ============================================================
# Main
# ============================================================

def main() -> None:
    print()
    print("=" * 80)
    print("LONG-DURATION RIGID PLANT VISUAL VALIDATION")
    print("=" * 80)

    print()
    print(f"Model: {XML_PATH}")

    print()
    print(
        f"{len(SCENARIOS)} scenarios × "
        f"{SCENARIO_DURATION:.1f} s = "
        f"{len(SCENARIOS) * SCENARIO_DURATION:.1f} s "
        "of simulated test time"
    )

    print()
    print(
        "Each scenario resets the vehicle so the response "
        "is easy to interpret."
    )

    print()
    print("Rotor arrows show ACTUAL alpha, not alpha_cmd.")

    print()
    print(
        f"Detailed CSV telemetry will be saved at {LOG_HZ:.0f} Hz."
    )
    print()

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    dt = float(model.opt.timestep)

    assert np.isclose(dt, 0.002), (
        "Expected the nominal 2 ms timestep. "
        f"Got {dt}."
    )

    # Precompute physical commands before opening the viewer.
    command_sets = []

    for name, delta in SCENARIOS:
        command_sets.append(
            (
                name,
                build_commands(delta),
            )
        )

    logger = ValidationLogger()
    completed = False

    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -20.0
            viewer.cam.lookat[:] = np.array([0.0, 0.0, 1.3])

            viewer_period = 1.0 / VIEWER_UPDATE_HZ
            telemetry_period = 1.0 / TELEMETRY_HZ
            log_period = 1.0 / LOG_HZ

            for scenario_index, (
                scenario_name,
                commands,
            ) in enumerate(
                command_sets,
                start=1,
            ):
                if not viewer.is_running():
                    break

                alpha = reset_vehicle(model, data)

                print()
                print("=" * 80)
                print(
                    f"SCENARIO "
                    f"{scenario_index}/{len(command_sets)}"
                )
                print(scenario_name)
                print("=" * 80)

                next_viewer_update = 0.0
                next_telemetry = 0.0
                next_log = 0.0

                while (
                    viewer.is_running()
                    and data.time < SCENARIO_DURATION
                ):
                    step_wall_start = time.perf_counter()
                    local_time = float(data.time)

                    if local_time < POSITIVE_START:
                        phase = "hover"
                    elif local_time < NEGATIVE_START:
                        phase = "positive"
                    elif local_time < RETURN_START:
                        phase = "negative"
                    else:
                        phase = "hover"

                    (
                        target_wrench,
                        thrust,
                        alpha_cmd,
                    ) = commands[phase]

                    alpha = step_rigid_plant(
                        model,
                        data,
                        alpha,
                        thrust,
                        alpha_cmd,
                    )

                    if local_time + 1e-12 >= next_log:
                        logger.log(
                            scenario_index=scenario_index,
                            scenario_name=scenario_name,
                            phase=phase,
                            local_time=local_time,
                            data=data,
                            thrust=thrust,
                            alpha_cmd=alpha_cmd,
                            alpha=alpha,
                            target_wrench=target_wrench,
                        )
                        next_log += log_period

                    if local_time + 1e-12 >= next_viewer_update:
                        with viewer.lock():
                            update_rotor_arrows(
                                viewer,
                                model,
                                data,
                                alpha,
                                thrust,
                            )

                        viewer.sync()
                        next_viewer_update += viewer_period

                    if local_time + 1e-12 >= next_telemetry:
                        print_telemetry(
                            scenario_name,
                            local_time,
                            phase,
                            data,
                            thrust,
                            alpha_cmd,
                            alpha,
                            target_wrench,
                        )
                        next_telemetry += telemetry_period

                    elapsed = time.perf_counter() - step_wall_start
                    remaining = dt - elapsed

                    if remaining > 0.0:
                        time.sleep(remaining)

                logger.add_scenario_summary(
                    scenario_index,
                    scenario_name,
                    data,
                    alpha,
                )

                if not viewer.is_running():
                    break

                final_wait_start = time.perf_counter()

                while (
                    viewer.is_running()
                    and (
                        time.perf_counter()
                        - final_wait_start
                        < 0.5
                    )
                ):
                    viewer.sync()
                    time.sleep(1.0 / 60.0)

            completed = (
                viewer.is_running()
                and len(logger.scenario_summaries) == len(SCENARIOS)
            )

    finally:
        logger.close(completed=completed)

        print()
        print("=" * 80)
        print("VALIDATION RUN FINISHED")
        print("=" * 80)

        print()
        print("Saved detailed telemetry:")
        print(logger.csv_path)

        print()
        print("Saved validation summary:")
        print(logger.summary_path)

        if not completed:
            print()
            print(
                "NOTE: The viewer was closed or the run ended "
                "before all scenarios completed."
            )


if __name__ == "__main__":
    main()
