from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    allocation_matrix,
    rotor_commands_to_virtual,
)
from tiltable_quadrotor_rl.control.end_effector import (
    DEFAULT_END_EFFECTOR,
    body_to_end_effector_kinematics,
    quaternion_to_rotation_np,
)
from tiltable_quadrotor_rl.control.nmpc_effector_6dof_trajectory import (
    CENTER_POSITION_W,
    EULER_AMPLITUDE_DEG,
    EULER_PERIOD_S,
    TRANSLATION_AMPLITUDE_M,
    TRANSLATION_PERIOD_S,
    RATE_BIAS_GAIN_M_PER_DEGPS,
    General6DofEndEffectorTrajectory,
)
from tiltable_quadrotor_rl.control.nmpc_effector_controller import (
    EffectorNmpcController,
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
)
from tiltable_quadrotor_rl.robot.parameters import (
    ARM_LENGTH,
    GRAVITY,
    INERTIA_DIAG as NOMINAL_MODEL_INERTIA_DIAG,
    KQ_OVER_KT,
    MASS as NOMINAL_MODEL_MASS_KG,
    SERVO_COMMAND_LIMIT,
    SERVO_MECHANICAL_LIMIT,
    SERVO_TIME_CONSTANT,
)
from tiltable_quadrotor_rl.robot.rigid_plant import (
    full_state_from_mujoco,
)
from tiltable_quadrotor_rl.robot.tilt_servo import (
    first_order_servo_step,
)
from tiltable_quadrotor_rl.visualization.cad_maneuver import (
    CadManeuverVisual,
)
from tiltable_quadrotor_rl.visualization.servo_overlay import (
    update_servo_overlay,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

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

NOMINAL_WEIGHT_N = (
    NOMINAL_MODEL_MASS_KG
    * GRAVITY
)

CHARACTERISTIC_TORQUE_NM = (
    NOMINAL_WEIGHT_N
    * ARM_LENGTH
)

DEFAULT_DISTURBANCE_START_S = 5.0


@dataclass(frozen=True)
class RobustnessConfig:
    name: str
    mass_scale: float = 1.0
    inertia_scale: float = 1.0
    servo_time_scale: float = 1.0
    thrust_effectiveness_scale: float = 1.0
    rotor_effectiveness: tuple[float, float, float, float] = (
        1.0, 1.0, 1.0, 1.0
    )
    reaction_torque_scale: float = 1.0
    disturbance_force_W_N: tuple[float, float, float] = (
        0.0, 0.0, 0.0
    )
    disturbance_torque_W_Nm: tuple[float, float, float] = (
        0.0, 0.0, 0.0
    )
    disturbance_start_s: float = DEFAULT_DISTURBANCE_START_S


ROBUSTNESS_CASES = {
    "nominal": RobustnessConfig(
        name="nominal",
    ),
    "mass_low_10": RobustnessConfig(
        name="mass_low_10",
        mass_scale=0.90,
    ),
    "mass_high_10": RobustnessConfig(
        name="mass_high_10",
        mass_scale=1.10,
    ),
    "inertia_low_10": RobustnessConfig(
        name="inertia_low_10",
        inertia_scale=0.90,
    ),
    "inertia_high_10": RobustnessConfig(
        name="inertia_high_10",
        inertia_scale=1.10,
    ),
    "servo_fast_25": RobustnessConfig(
        name="servo_fast_25",
        servo_time_scale=0.75,
    ),
    "servo_slow_25": RobustnessConfig(
        name="servo_slow_25",
        servo_time_scale=1.25,
    ),
    "thrust_low_10": RobustnessConfig(
        name="thrust_low_10",
        thrust_effectiveness_scale=0.90,
    ),
    "thrust_high_10": RobustnessConfig(
        name="thrust_high_10",
        thrust_effectiveness_scale=1.10,
    ),
    "rotor1_low_10": RobustnessConfig(
        name="rotor1_low_10",
        rotor_effectiveness=(
            0.90, 1.0, 1.0, 1.0
        ),
    ),
    "force_x_5pct_weight": RobustnessConfig(
        name="force_x_5pct_weight",
        disturbance_force_W_N=(
            0.05 * NOMINAL_WEIGHT_N,
            0.0,
            0.0,
        ),
    ),
    "yaw_torque_5pct": RobustnessConfig(
        name="yaw_torque_5pct",
        disturbance_torque_W_Nm=(
            0.0,
            0.0,
            0.05 * CHARACTERISTIC_TORQUE_NM,
        ),
    ),
}

NEW_ROBUSTNESS_BATCH = (
    "servo_fast_25",
    "servo_slow_25",
    "thrust_low_10",
    "thrust_high_10",
    "rotor1_low_10",
    "force_x_5pct_weight",
    "yaw_torque_5pct",
)


def _parse_csv_vector(
    raw: str,
    *,
    length: int,
    label: str,
) -> tuple[float, ...]:
    values = tuple(
        float(item.strip())
        for item in raw.split(",")
    )

    if len(values) != length:
        raise ValueError(
            f"{label} must contain exactly {length} comma-separated values."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{label} must contain finite values."
        )

    return values


def resolve_robustness_config(
    args: argparse.Namespace,
) -> RobustnessConfig:
    base = ROBUSTNESS_CASES[
        args.robustness_case
    ]

    def choose(override, default):
        return default if override is None else override

    rotor_effectiveness = base.rotor_effectiveness
    if args.rotor_effectiveness is not None:
        rotor_effectiveness = _parse_csv_vector(
            args.rotor_effectiveness,
            length=4,
            label="--rotor-effectiveness",
        )

    disturbance_force = base.disturbance_force_W_N
    if args.disturbance_force_world is not None:
        disturbance_force = _parse_csv_vector(
            args.disturbance_force_world,
            length=3,
            label="--disturbance-force-world",
        )

    disturbance_torque = base.disturbance_torque_W_Nm
    if args.disturbance_torque_world is not None:
        disturbance_torque = _parse_csv_vector(
            args.disturbance_torque_world,
            length=3,
            label="--disturbance-torque-world",
        )

    config = RobustnessConfig(
        name=base.name,
        mass_scale=float(
            choose(
                args.plant_mass_scale,
                base.mass_scale,
            )
        ),
        inertia_scale=float(
            choose(
                args.plant_inertia_scale,
                base.inertia_scale,
            )
        ),
        servo_time_scale=float(
            choose(
                args.servo_time_scale,
                base.servo_time_scale,
            )
        ),
        thrust_effectiveness_scale=float(
            choose(
                args.thrust_effectiveness_scale,
                base.thrust_effectiveness_scale,
            )
        ),
        rotor_effectiveness=tuple(
            float(v)
            for v in rotor_effectiveness
        ),
        reaction_torque_scale=float(
            choose(
                args.reaction_torque_scale,
                base.reaction_torque_scale,
            )
        ),
        disturbance_force_W_N=tuple(
            float(v)
            for v in disturbance_force
        ),
        disturbance_torque_W_Nm=tuple(
            float(v)
            for v in disturbance_torque
        ),
        disturbance_start_s=float(
            choose(
                args.disturbance_start_s,
                base.disturbance_start_s,
            )
        ),
    )

    for label, value in {
        "mass_scale": config.mass_scale,
        "inertia_scale": config.inertia_scale,
        "servo_time_scale": config.servo_time_scale,
        "thrust_effectiveness_scale": config.thrust_effectiveness_scale,
        "reaction_torque_scale": config.reaction_torque_scale,
    }.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{label} must be finite and > 0."
            )

    if np.any(
        np.asarray(
            config.rotor_effectiveness,
            dtype=float,
        )
        <= 0.0
    ):
        raise ValueError(
            "All rotor effectiveness scales must be > 0."
        )

    return config


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


def apply_plant_parameter_scales(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    mass_scale: float,
    inertia_scale: float,
) -> tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
    """
    Apply controlled rigid-body parameter mismatch to the MuJoCo plant.

    The NMPC model remains fixed at the nominal mass/inertia imported
    from robot.parameters.

    mass_scale:
        Scales only the MuJoCo body mass.

    inertia_scale:
        Scales all three MuJoCo principal inertias together while
        preserving their nominal ratios.

    This lets mass and inertia mismatch be studied independently.
    """
    for name, value in (
        ("plant mass scale", mass_scale),
        ("plant inertia scale", inertia_scale),
    ):
        if not np.isfinite(
            value
        ):
            raise ValueError(
                f"{name} must be finite."
            )

        if value <= 0.0:
            raise ValueError(
                f"{name} must be positive."
            )

    base_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "base",
    )

    if base_body_id < 0:
        raise RuntimeError(
            "Could not find MuJoCo body named 'base'."
        )

    nominal_plant_mass_kg = float(
        model.body_mass[
            base_body_id
        ]
    )

    nominal_plant_inertia_diag = np.asarray(
        model.body_inertia[
            base_body_id
        ],
        dtype=float,
    ).copy()

    actual_plant_mass_kg = (
        nominal_plant_mass_kg
        * float(
            mass_scale
        )
    )

    actual_plant_inertia_diag = (
        nominal_plant_inertia_diag
        * float(
            inertia_scale
        )
    )

    model.body_mass[
        base_body_id
    ] = actual_plant_mass_kg

    model.body_inertia[
        base_body_id
    ] = actual_plant_inertia_diag

    # Recompute MuJoCo constants after runtime inertial changes.
    mujoco.mj_setConst(
        model,
        data,
    )

    return (
        nominal_plant_mass_kg,
        actual_plant_mass_kg,
        nominal_plant_inertia_diag,
        actual_plant_inertia_diag,
    )


def step_robust_plant(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    alpha: np.ndarray,
    thrust_command: np.ndarray,
    alpha_command: np.ndarray,
    *,
    config: RobustnessConfig,
    time_s: float,
) -> np.ndarray:
    """
    Advance the actual plant one physics step while the NMPC model
    remains nominal.
    """
    dt = float(
        model.opt.timestep
    )

    actual_tau = (
        SERVO_TIME_CONSTANT
        * config.servo_time_scale
    )

    alpha_mid = first_order_servo_step(
        alpha,
        alpha_command,
        0.5 * dt,
        time_constant=actual_tau,
        command_limit=SERVO_COMMAND_LIMIT,
        mechanical_limit=SERVO_MECHANICAL_LIMIT,
    )

    actual_thrust = (
        np.asarray(
            thrust_command,
            dtype=float,
        ).reshape(4)
        * config.thrust_effectiveness_scale
        * np.asarray(
            config.rotor_effectiveness,
            dtype=float,
        )
    )

    z = rotor_commands_to_virtual(
        actual_thrust,
        alpha_mid,
    )

    actual_A = allocation_matrix(
        arm_length=ARM_LENGTH,
        kq_over_kt=(
            KQ_OVER_KT
            * config.reaction_torque_scale
        ),
    )

    wrench_B = (
        actual_A @ z
    )

    if (
        time_s
        + 1e-12
        >= config.disturbance_start_s
    ):
        force_W = np.asarray(
            config.disturbance_force_W_N,
            dtype=float,
        )

        torque_W = np.asarray(
            config.disturbance_torque_W_Nm,
            dtype=float,
        )

        R_WB = quaternion_to_rotation_np(
            data.qpos[3:7]
        )

        wrench_B[0:3] += (
            R_WB.T @ force_W
        )

        wrench_B[3:6] += (
            R_WB.T @ torque_W
        )

    data.ctrl[:] = wrench_B

    mujoco.mj_step(
        model,
        data,
    )

    return first_order_servo_step(
        alpha,
        alpha_command,
        dt,
        time_constant=actual_tau,
        command_limit=SERVO_COMMAND_LIMIT,
        mechanical_limit=SERVO_MECHANICAL_LIMIT,
    )


def reset_vehicle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: General6DofEndEffectorTrajectory,
) -> np.ndarray:
    """
    Initialize body pose/velocity and physical tilt states from the
    first trajectory point and its refined actuator reference.
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

    stage0 = (
        make_effector_stage_reference(
            reference,
            previous_effector_quaternion=(
                reference.quaternion_WE
            ),
            previous_body_quaternion=(
                body_reference.quaternion_WB
            ),
            previous_alpha=np.zeros(
                4,
                dtype=float,
            ),
            transform=DEFAULT_END_EFFECTOR,
        )
    )

    return (
        stage0.parameters[
            P_ALPHA
        ].copy()
    )


def update_target_marker(
    viewer,
    target_position_W: np.ndarray,
) -> None:
    """
    Green sphere showing the current desired tool position.
    Slot 0 is reserved for this marker; servo_overlay starts at slot 1.
    """
    marker = (
        viewer.user_scn.geoms[
            0
        ]
    )

    mujoco.mjv_initGeom(
        marker,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array(
            [
                0.035,
                0.035,
                0.035,
            ],
            dtype=float,
        ),
        np.asarray(
            target_position_W,
            dtype=float,
        ),
        np.eye(
            3
        ).reshape(
            -1
        ),
        np.array(
            [
                0.1,
                1.0,
                0.1,
                0.70,
            ],
            dtype=float,
        ),
    )

    viewer.user_scn.ngeom = max(
        viewer.user_scn.ngeom,
        1,
    )


class Logger:
    def __init__(
        self,
        *,
        trajectory: General6DofEndEffectorTrajectory,
        robustness_config: RobustnessConfig,
        plant_mass_scale: float,
        plant_inertia_scale: float,
        nominal_plant_mass_kg: float,
        actual_plant_mass_kg: float,
        nominal_plant_inertia_diag: np.ndarray,
        actual_plant_inertia_diag: np.ndarray,
    ) -> None:
        self.trajectory = (
            trajectory
        )

        self.robustness_config = (
            robustness_config
        )

        self.plant_mass_scale = float(
            plant_mass_scale
        )

        self.plant_inertia_scale = float(
            plant_inertia_scale
        )

        self.nominal_plant_mass_kg = float(
            nominal_plant_mass_kg
        )

        self.actual_plant_mass_kg = float(
            actual_plant_mass_kg
        )

        self.nominal_plant_inertia_diag = np.asarray(
            nominal_plant_inertia_diag,
            dtype=float,
        ).reshape(
            3
        ).copy()

        self.actual_plant_inertia_diag = np.asarray(
            actual_plant_inertia_diag,
            dtype=float,
        ).reshape(
            3
        ).copy()

        LOG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        suffix_parts: list[str] = []

        if trajectory.bias_compensation:
            suffix_parts.append(
                "bias_comp"
            )

        if (
            self.robustness_config.name
            != "nominal"
        ):
            suffix_parts.append(
                self.robustness_config.name
            )

        if not np.isclose(
            self.plant_mass_scale,
            1.0,
            atol=1e-12,
        ):
            scale_label = (
                f"{self.plant_mass_scale:.3f}"
                .replace(
                    ".",
                    "p",
                )
            )

            suffix_parts.append(
                f"mass_scale_{scale_label}"
            )

        if not np.isclose(
            self.plant_inertia_scale,
            1.0,
            atol=1e-12,
        ):
            inertia_label = (
                f"{self.plant_inertia_scale:.3f}"
                .replace(
                    ".",
                    "p",
                )
            )

            suffix_parts.append(
                f"inertia_scale_{inertia_label}"
            )

        suffix = (
            "_"
            + "_".join(
                suffix_parts
            )
            if suffix_parts
            else ""
        )

        self.csv_path = (
            LOG_DIR
            / (
                "general_6dof_effector_closed_loop"
                f"{suffix}_{timestamp}.csv"
            )
        )

        self.summary_path = (
            LOG_DIR
            / (
                "general_6dof_effector_closed_loop"
                f"{suffix}_{timestamp}_summary.txt"
            )
        )

        self.file = (
            self.csv_path.open(
                "w",
                newline="",
            )
        )

        fieldnames = [
            "time_s",
            "activity_window",

            "plant_mass_scale",
            "plant_inertia_scale",
            "nominal_plant_mass_kg",
            "actual_plant_mass_kg",
            "nominal_plant_Ixx_kgm2",
            "nominal_plant_Iyy_kgm2",
            "nominal_plant_Izz_kgm2",
            "actual_plant_Ixx_kgm2",
            "actual_plant_Iyy_kgm2",
            "actual_plant_Izz_kgm2",

            "true_ref_px_m",
            "true_ref_py_m",
            "true_ref_pz_m",
            "controller_ref_px_m",
            "controller_ref_py_m",
            "controller_ref_pz_m",
            "compensation_x_m",
            "compensation_y_m",
            "compensation_z_m",
            "actual_px_m",
            "actual_py_m",
            "actual_pz_m",
            "error_px_m",
            "error_py_m",
            "error_pz_m",
            "tool_position_error_m",

            "true_ref_vx_mps",
            "true_ref_vy_mps",
            "true_ref_vz_mps",
            "controller_ref_vx_mps",
            "controller_ref_vy_mps",
            "controller_ref_vz_mps",
            "actual_vx_mps",
            "actual_vy_mps",
            "actual_vz_mps",
            "tool_velocity_error_mps",

            "ref_roll_deg",
            "ref_pitch_deg",
            "ref_yaw_deg",
            "attitude_error_deg",

            "ref_omega_x_radps",
            "ref_omega_y_radps",
            "ref_omega_z_radps",
            "actual_omega_x_radps",
            "actual_omega_y_radps",
            "actual_omega_z_radps",
            "omega_error_radps",

            "ref_qw",
            "ref_qx",
            "ref_qy",
            "ref_qz",
            "actual_qw",
            "actual_qx",
            "actual_qy",
            "actual_qz",

            "body_px_m",
            "body_py_m",
            "body_pz_m",

            "solve_time_ms",
            "total_controller_time_ms",
            "solver_status",
            "acados_deadline_missed",
            "total_deadline_missed",
        ]

        for rotor in range(
            1,
            5,
        ):
            fieldnames.extend(
                [
                    f"ref_thrust_{rotor}_N",
                    f"ref_alpha_{rotor}_deg",
                    f"actual_alpha_{rotor}_deg",
                    f"alpha_cmd_{rotor}_deg",
                    f"thrust_cmd_{rotor}_N",
                ]
            )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=fieldnames,
        )

        self.writer.writeheader()

        self.times_s: list[
            float
        ] = []

        self.position_error_vectors: list[
            np.ndarray
        ] = []

        self.position_errors: list[
            float
        ] = []

        self.controller_reference_position_errors: list[
            float
        ] = []

        self.velocity_errors: list[
            float
        ] = []

        self.attitude_errors: list[
            float
        ] = []

        self.omega_errors: list[
            float
        ] = []

        self.solve_times_ms: list[
            float
        ] = []

        self.total_times_ms: list[
            float
        ] = []

        self.thrust_commands_N: list[
            np.ndarray
        ] = []

        self.max_abs_alpha_deg = 0.0
        self.max_abs_alpha_cmd_deg = 0.0
        self.max_thrust_cmd_N = 0.0

        self.control_updates = 0
        self.solver_failures = 0
        self.acados_deadline_misses = 0
        self.total_deadline_misses = 0

        self.closed = False

    def note_solver_failure(
        self,
    ) -> None:
        self.solver_failures += 1

    def record(
        self,
        *,
        time_s: float,
        state: np.ndarray,
        effector,
        true_reference,
        controller_reference,
        compensation,
        euler,
        stage0_reference,
        command,
        total_controller_time_s: float,
        activity_window: float,
    ) -> None:
        position_error_vector = (
            effector.position_W
            - true_reference.position_W
        )

        controller_reference_position_error = float(
            np.linalg.norm(
                effector.position_W
                - controller_reference.position_W
            )
        )

        position_error = float(
            np.linalg.norm(
                position_error_vector
            )
        )

        velocity_error = float(
            np.linalg.norm(
                effector.velocity_W
                - true_reference.velocity_W
            )
        )

        attitude_error = (
            quaternion_error_deg(
                effector.quaternion_WE,
                true_reference.quaternion_WE,
            )
        )

        omega_error = float(
            np.linalg.norm(
                effector.omega_E
                - true_reference.omega_E
            )
        )

        solve_time_ms = (
            1000.0
            * float(
                command.solve_time_s
            )
        )

        total_time_ms = (
            1000.0
            * float(
                total_controller_time_s
            )
        )

        acados_miss = (
            command.solve_time_s
            > CONTROL_PERIOD
        )

        total_miss = (
            total_controller_time_s
            > CONTROL_PERIOD
        )

        ref_thrust = np.asarray(
            stage0_reference.parameters[
                P_THRUST
            ],
            dtype=float,
        ).reshape(
            4
        )

        ref_alpha = np.asarray(
            stage0_reference.parameters[
                P_ALPHA
            ],
            dtype=float,
        ).reshape(
            4
        )

        row = {
            "time_s":
                time_s,
            "activity_window":
                activity_window,

            "plant_mass_scale":
                self.plant_mass_scale,
            "plant_inertia_scale":
                self.plant_inertia_scale,
            "nominal_plant_mass_kg":
                self.nominal_plant_mass_kg,
            "actual_plant_mass_kg":
                self.actual_plant_mass_kg,

            "nominal_plant_Ixx_kgm2":
                self.nominal_plant_inertia_diag[0],
            "nominal_plant_Iyy_kgm2":
                self.nominal_plant_inertia_diag[1],
            "nominal_plant_Izz_kgm2":
                self.nominal_plant_inertia_diag[2],

            "actual_plant_Ixx_kgm2":
                self.actual_plant_inertia_diag[0],
            "actual_plant_Iyy_kgm2":
                self.actual_plant_inertia_diag[1],
            "actual_plant_Izz_kgm2":
                self.actual_plant_inertia_diag[2],

            "true_ref_px_m":
                true_reference.position_W[0],
            "true_ref_py_m":
                true_reference.position_W[1],
            "true_ref_pz_m":
                true_reference.position_W[2],

            "controller_ref_px_m":
                controller_reference.position_W[0],
            "controller_ref_py_m":
                controller_reference.position_W[1],
            "controller_ref_pz_m":
                controller_reference.position_W[2],

            "compensation_x_m":
                compensation.position_offset_W[0],
            "compensation_y_m":
                compensation.position_offset_W[1],
            "compensation_z_m":
                compensation.position_offset_W[2],

            "actual_px_m":
                effector.position_W[0],
            "actual_py_m":
                effector.position_W[1],
            "actual_pz_m":
                effector.position_W[2],

            "error_px_m":
                position_error_vector[0],
            "error_py_m":
                position_error_vector[1],
            "error_pz_m":
                position_error_vector[2],

            "tool_position_error_m":
                position_error,

            "true_ref_vx_mps":
                true_reference.velocity_W[0],
            "true_ref_vy_mps":
                true_reference.velocity_W[1],
            "true_ref_vz_mps":
                true_reference.velocity_W[2],

            "controller_ref_vx_mps":
                controller_reference.velocity_W[0],
            "controller_ref_vy_mps":
                controller_reference.velocity_W[1],
            "controller_ref_vz_mps":
                controller_reference.velocity_W[2],

            "actual_vx_mps":
                effector.velocity_W[0],
            "actual_vy_mps":
                effector.velocity_W[1],
            "actual_vz_mps":
                effector.velocity_W[2],

            "tool_velocity_error_mps":
                velocity_error,

            "ref_roll_deg":
                np.degrees(
                    euler.angles_rad[0]
                ),
            "ref_pitch_deg":
                np.degrees(
                    euler.angles_rad[1]
                ),
            "ref_yaw_deg":
                np.degrees(
                    euler.angles_rad[2]
                ),

            "attitude_error_deg":
                attitude_error,

            "ref_omega_x_radps":
                true_reference.omega_E[0],
            "ref_omega_y_radps":
                true_reference.omega_E[1],
            "ref_omega_z_radps":
                true_reference.omega_E[2],

            "actual_omega_x_radps":
                effector.omega_E[0],
            "actual_omega_y_radps":
                effector.omega_E[1],
            "actual_omega_z_radps":
                effector.omega_E[2],

            "omega_error_radps":
                omega_error,

            "ref_qw":
                true_reference.quaternion_WE[0],
            "ref_qx":
                true_reference.quaternion_WE[1],
            "ref_qy":
                true_reference.quaternion_WE[2],
            "ref_qz":
                true_reference.quaternion_WE[3],

            "actual_qw":
                effector.quaternion_WE[0],
            "actual_qx":
                effector.quaternion_WE[1],
            "actual_qy":
                effector.quaternion_WE[2],
            "actual_qz":
                effector.quaternion_WE[3],

            "body_px_m":
                state[0],
            "body_py_m":
                state[1],
            "body_pz_m":
                state[2],

            "solve_time_ms":
                solve_time_ms,

            "total_controller_time_ms":
                total_time_ms,

            "solver_status":
                command.status,

            "acados_deadline_missed":
                int(
                    acados_miss
                ),

            "total_deadline_missed":
                int(
                    total_miss
                ),
        }

        for rotor in range(
            4
        ):
            index = (
                rotor + 1
            )

            row[
                f"ref_thrust_{index}_N"
            ] = ref_thrust[
                rotor
            ]

            row[
                f"ref_alpha_{index}_deg"
            ] = np.degrees(
                ref_alpha[
                    rotor
                ]
            )

            row[
                f"actual_alpha_{index}_deg"
            ] = np.degrees(
                state[
                    13 + rotor
                ]
            )

            row[
                f"alpha_cmd_{index}_deg"
            ] = np.degrees(
                command.alpha_cmd[
                    rotor
                ]
            )

            row[
                f"thrust_cmd_{index}_N"
            ] = command.thrust[
                rotor
            ]

        self.writer.writerow(
            row
        )

        self.times_s.append(
            float(
                time_s
            )
        )

        self.position_error_vectors.append(
            position_error_vector.copy()
        )

        self.position_errors.append(
            position_error
        )

        self.controller_reference_position_errors.append(
            controller_reference_position_error
        )

        self.velocity_errors.append(
            velocity_error
        )

        self.attitude_errors.append(
            attitude_error
        )

        self.omega_errors.append(
            omega_error
        )

        self.solve_times_ms.append(
            solve_time_ms
        )

        self.total_times_ms.append(
            total_time_ms
        )

        self.thrust_commands_N.append(
            np.asarray(
                command.thrust,
                dtype=float,
            ).reshape(
                4
            ).copy()
        )

        self.max_abs_alpha_deg = max(
            self.max_abs_alpha_deg,
            float(
                np.max(
                    np.abs(
                        np.degrees(
                            state[13:17]
                        )
                    )
                )
            ),
        )

        self.max_abs_alpha_cmd_deg = max(
            self.max_abs_alpha_cmd_deg,
            float(
                np.max(
                    np.abs(
                        np.degrees(
                            command.alpha_cmd
                        )
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

        if acados_miss:
            self.acados_deadline_misses += 1

        if total_miss:
            self.total_deadline_misses += 1

    @staticmethod
    def _rms(
        values: np.ndarray,
    ) -> float:
        if values.size == 0:
            return float(
                "nan"
            )

        return float(
            np.sqrt(
                np.mean(
                    values**2
                )
            )
        )

    def close(
        self,
        *,
        completed: bool,
        final_time_s: float,
    ) -> None:
        if self.closed:
            return

        self.file.close()
        self.closed = True

        times = np.asarray(
            self.times_s,
            dtype=float,
        )

        position_vectors = np.asarray(
            self.position_error_vectors,
            dtype=float,
        )

        position = np.asarray(
            self.position_errors,
            dtype=float,
        )

        controller_position = np.asarray(
            self.controller_reference_position_errors,
            dtype=float,
        )

        velocity = np.asarray(
            self.velocity_errors,
            dtype=float,
        )

        attitude = np.asarray(
            self.attitude_errors,
            dtype=float,
        )

        omega = np.asarray(
            self.omega_errors,
            dtype=float,
        )

        solve = np.asarray(
            self.solve_times_ms,
            dtype=float,
        )

        total = np.asarray(
            self.total_times_ms,
            dtype=float,
        )

        thrust_commands = np.asarray(
            self.thrust_commands_N,
            dtype=float,
        )

        active_mask = (
            (times >= self.trajectory.active_start_s)
            & (
                times
                <= self.trajectory.active_end_s
            )
        )

        full_mask = (
            (
                times
                >= self.trajectory.full_activity_start_s
            )
            & (
                times
                <= self.trajectory.full_activity_end_s
            )
        )

        final_hold_mask = (
            times
            >= self.trajectory.active_end_s
        )

        def subset(
            values: np.ndarray,
            mask: np.ndarray,
        ) -> np.ndarray:
            if (
                values.shape[0]
                != mask.shape[0]
            ):
                return np.asarray(
                    [],
                    dtype=float,
                )

            return values[
                mask
            ]

        def write_tracking_block(
            file,
            title: str,
            mask: np.ndarray | None,
        ) -> None:
            if mask is None:
                pos = position
                pos_vec = position_vectors
                vel = velocity
                att = attitude
                omg = omega
            else:
                pos = subset(
                    position,
                    mask,
                )
                pos_vec = subset(
                    position_vectors,
                    mask,
                )
                vel = subset(
                    velocity,
                    mask,
                )
                att = subset(
                    attitude,
                    mask,
                )
                omg = subset(
                    omega,
                    mask,
                )

            file.write(
                title + "\n"
            )

            file.write(
                "  tool position RMS [m]: "
                f"{self._rms(pos):.6f}\n"
            )

            file.write(
                "  tool position max [m]: "
                f"{np.max(pos) if pos.size else float('nan'):.6f}\n"
            )

            if pos_vec.size:
                per_axis_rms = np.sqrt(
                    np.mean(
                        pos_vec**2,
                        axis=0,
                    )
                )
            else:
                per_axis_rms = np.full(
                    3,
                    np.nan,
                )

            file.write(
                "  position-axis RMS [m] X/Y/Z: "
                f"{per_axis_rms[0]:.6f}, "
                f"{per_axis_rms[1]:.6f}, "
                f"{per_axis_rms[2]:.6f}\n"
            )

            if pos_vec.size:
                per_axis_mean = np.mean(
                    pos_vec,
                    axis=0,
                )
            else:
                per_axis_mean = np.full(
                    3,
                    np.nan,
                )

            file.write(
                "  signed position mean [m] X/Y/Z: "
                f"{per_axis_mean[0]:.6f}, "
                f"{per_axis_mean[1]:.6f}, "
                f"{per_axis_mean[2]:.6f}\n"
            )

            file.write(
                "  tool velocity RMS [m/s]: "
                f"{self._rms(vel):.6f}\n"
            )

            file.write(
                "  attitude RMS [deg]: "
                f"{self._rms(att):.6f}\n"
            )

            file.write(
                "  attitude max [deg]: "
                f"{np.max(att) if att.size else float('nan'):.6f}\n"
            )

            file.write(
                "  angular-velocity RMS [rad/s]: "
                f"{self._rms(omg):.6f}\n\n"
            )

        with self.summary_path.open(
            "w"
        ) as summary:
            summary.write(
                "GENERAL 6-DOF END-EFFECTOR NMPC BENCHMARK\n"
            )
            summary.write(
                "=" * 72
                + "\n\n"
            )

            summary.write(
                "Plant / controller model\n"
            )
            summary.write(
                "  NMPC model mass [kg]: "
                f"{NOMINAL_MODEL_MASS_KG:.6f}\n"
            )
            summary.write(
                "  NMPC model inertia diag [kg m^2]: "
                f"[{NOMINAL_MODEL_INERTIA_DIAG[0]:.6f}, "
                f"{NOMINAL_MODEL_INERTIA_DIAG[1]:.6f}, "
                f"{NOMINAL_MODEL_INERTIA_DIAG[2]:.6f}]\n"
            )
            summary.write(
                "  nominal MJCF plant mass [kg]: "
                f"{self.nominal_plant_mass_kg:.6f}\n"
            )
            summary.write(
                "  plant mass scale: "
                f"{self.plant_mass_scale:.6f}\n"
            )
            summary.write(
                "  actual MuJoCo plant mass [kg]: "
                f"{self.actual_plant_mass_kg:.6f}\n"
            )
            summary.write(
                "  nominal MJCF inertia diag [kg m^2]: "
                f"[{self.nominal_plant_inertia_diag[0]:.6f}, "
                f"{self.nominal_plant_inertia_diag[1]:.6f}, "
                f"{self.nominal_plant_inertia_diag[2]:.6f}]\n"
            )
            summary.write(
                "  plant inertia scale: "
                f"{self.plant_inertia_scale:.6f}\n"
            )
            summary.write(
                "  actual MuJoCo inertia diag [kg m^2]: "
                f"[{self.actual_plant_inertia_diag[0]:.6f}, "
                f"{self.actual_plant_inertia_diag[1]:.6f}, "
                f"{self.actual_plant_inertia_diag[2]:.6f}]\n\n"
            )

            summary.write(
                "Robustness configuration\n"
            )
            summary.write(
                "  case: "
                f"{self.robustness_config.name}\n"
            )
            summary.write(
                "  servo time scale: "
                f"{self.robustness_config.servo_time_scale:.6f}\n"
            )
            summary.write(
                "  actual servo tau [s]: "
                f"{SERVO_TIME_CONSTANT * self.robustness_config.servo_time_scale:.6f}\n"
            )
            summary.write(
                "  uniform thrust effectiveness: "
                f"{self.robustness_config.thrust_effectiveness_scale:.6f}\n"
            )
            summary.write(
                "  per-rotor effectiveness R1/R2/R3/R4: "
                f"{self.robustness_config.rotor_effectiveness}\n"
            )
            summary.write(
                "  reaction torque scale: "
                f"{self.robustness_config.reaction_torque_scale:.6f}\n"
            )
            summary.write(
                "  external force W [N]: "
                f"{self.robustness_config.disturbance_force_W_N}\n"
            )
            summary.write(
                "  external torque W [N m]: "
                f"{self.robustness_config.disturbance_torque_W_Nm}\n"
            )
            summary.write(
                "  disturbance start [s]: "
                f"{self.robustness_config.disturbance_start_s:.6f}\n\n"
            )

            summary.write(
                "Trajectory\n"
            )
            summary.write(
                "  rate-bias compensation enabled: "
                f"{self.trajectory.bias_compensation}\n"
            )
            summary.write(
                "  rate-bias gain [m/(deg/s)]: "
                f"{self.trajectory.rate_bias_gain_m_per_degps:.8f}\n"
            )
            summary.write(
                "  center position [m]: "
                f"{self.trajectory.center_position_W}\n"
            )
            summary.write(
                "  XYZ amplitudes [m]: "
                f"{TRANSLATION_AMPLITUDE_M}\n"
            )
            summary.write(
                "  XYZ periods [s]: "
                f"{TRANSLATION_PERIOD_S}\n"
            )
            summary.write(
                "  roll/pitch/yaw amplitudes [deg]: "
                f"{EULER_AMPLITUDE_DEG}\n"
            )
            summary.write(
                "  roll/pitch/yaw periods [s]: "
                f"{EULER_PERIOD_S}\n"
            )
            summary.write(
                "  active interval [s]: "
                f"{self.trajectory.active_start_s:.3f} "
                "to "
                f"{self.trajectory.active_end_s:.3f}\n"
            )
            summary.write(
                "  full-activity interval [s]: "
                f"{self.trajectory.full_activity_start_s:.3f} "
                "to "
                f"{self.trajectory.full_activity_end_s:.3f}\n\n"
            )

            summary.write(
                f"Run completed normally: {completed}\n"
            )
            summary.write(
                "Final simulation time [s]: "
                f"{final_time_s:.3f}\n"
            )
            summary.write(
                f"Control updates: {self.control_updates}\n"
            )
            summary.write(
                f"Solver failures: {self.solver_failures}\n"
            )
            summary.write(
                "10 ms acados-only deadline misses: "
                f"{self.acados_deadline_misses}\n"
            )
            summary.write(
                "10 ms TOTAL-controller deadline misses: "
                f"{self.total_deadline_misses}\n\n"
            )

            def timing_block(
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

            timing_block(
                "Acados solve timing [ms]",
                solve,
            )

            timing_block(
                "TOTAL controller.solve() timing [ms]",
                total,
            )

            write_tracking_block(
                summary,
                "Tracking over complete run",
                None,
            )

            write_tracking_block(
                summary,
                "Tracking during active motion",
                active_mask,
            )

            write_tracking_block(
                summary,
                "Tracking during full-amplitude motion",
                full_mask,
            )

            write_tracking_block(
                summary,
                "Tracking during final stationary hold",
                final_hold_mask,
            )

            def write_thrust_block(
                title: str,
                mask: np.ndarray,
            ) -> None:
                thrust = subset(
                    thrust_commands,
                    mask,
                )

                summary.write(
                    title + "\n"
                )

                if (
                    thrust.size == 0
                    or thrust.ndim != 2
                ):
                    summary.write(
                        "  no data\n\n"
                    )
                    return

                mean_per_rotor = np.mean(
                    thrust,
                    axis=0,
                )

                mean_total = float(
                    np.mean(
                        np.sum(
                            thrust,
                            axis=1,
                        )
                    )
                )

                summary.write(
                    "  mean thrust per rotor [N] R1/R2/R3/R4: "
                    f"{mean_per_rotor[0]:.6f}, "
                    f"{mean_per_rotor[1]:.6f}, "
                    f"{mean_per_rotor[2]:.6f}, "
                    f"{mean_per_rotor[3]:.6f}\n"
                )
                summary.write(
                    "  mean total rotor thrust magnitude [N]: "
                    f"{mean_total:.6f}\n\n"
                )

            write_thrust_block(
                "Thrust during full-amplitude motion",
                full_mask,
            )

            write_thrust_block(
                "Thrust during final stationary hold",
                final_hold_mask,
            )

            summary.write(
                "Controller-reference position tracking\n"
            )
            summary.write(
                "  complete-run RMS [m]: "
                f"{self._rms(controller_position):.6f}\n"
            )
            summary.write(
                "  full-amplitude RMS [m]: "
                f"{self._rms(subset(controller_position, full_mask)):.6f}\n\n"
            )

            summary.write(
                "Actuator usage\n"
            )
            summary.write(
                "  max |actual alpha| [deg]: "
                f"{self.max_abs_alpha_deg:.6f}\n"
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
                "CSV telemetry:\n"
            )
            summary.write(
                str(
                    self.csv_path
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reusable general 6-DoF end-effector NMPC benchmark."
        )
    )

    parser.add_argument(
        "--robustness-case",
        choices=sorted(
            ROBUSTNESS_CASES.keys()
        ),
        default="nominal",
        help=(
            "Named robustness preset. Explicit options below override "
            "the selected preset."
        ),
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run without the MuJoCo GLFW viewer. Intended for batch "
            "robustness sweeps and automated experiments."
        ),
    )

    parser.add_argument(
        "--bias-compensation",
        action="store_true",
        help=(
            "Reuse the previously identified pitch/roll "
            "rate-bias compensation."
        ),
    )

    parser.add_argument(
        "--plant-mass-scale",
        type=float,
        default=None,
        help=(
            "Override the MuJoCo plant body-mass scale. "
            "Examples: 0.90, 1.00, 1.10."
        ),
    )

    parser.add_argument(
        "--plant-inertia-scale",
        type=float,
        default=None,
        help=(
            "Override all three MuJoCo principal-inertia scales while "
            "keeping the NMPC model unchanged. "
            "Examples: 0.90, 1.00, 1.10."
        ),
    )

    parser.add_argument(
        "--servo-time-scale",
        type=float,
        default=None,
        help="Override actual servo time-constant scale.",
    )

    parser.add_argument(
        "--thrust-effectiveness-scale",
        type=float,
        default=None,
        help="Override uniform realized-thrust effectiveness scale.",
    )

    parser.add_argument(
        "--rotor-effectiveness",
        type=str,
        default=None,
        help="Override per-rotor thrust scales as R1,R2,R3,R4.",
    )

    parser.add_argument(
        "--reaction-torque-scale",
        type=float,
        default=None,
        help="Override actual KQ/KT reaction-torque scale.",
    )

    parser.add_argument(
        "--disturbance-force-world",
        type=str,
        default=None,
        help="Override constant world-frame force Fx,Fy,Fz [N].",
    )

    parser.add_argument(
        "--disturbance-torque-world",
        type=str,
        default=None,
        help="Override constant world-frame torque Tx,Ty,Tz [N m].",
    )

    parser.add_argument(
        "--disturbance-start-s",
        type=float,
        default=None,
        help="Override disturbance start time [s].",
    )

    parser.add_argument(
        "--cad-visual",
        action="store_true",
        help=(
            "Display/render the maneuver with the CAD visual MJCF and "
            "spinning propellers. Physics and control still run on the "
            "original rigid plant; the CAD model only mirrors its state."
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

    return args


def main() -> None:
    args = parse_args()

    robustness = resolve_robustness_config(
        args
    )

    trajectory = (
        General6DofEndEffectorTrajectory(
            bias_compensation=(
                args.bias_compensation
            ),
        )
    )

    print()
    print("=" * 80)
    print(
        "GENERAL 6-DOF END-EFFECTOR NMPC BENCHMARK"
    )
    print("=" * 80)

    print()
    print(
        "Controller               : frozen baseline "
        "EffectorNmpcController"
    )
    print(
        "Predictive pre-position  : disabled"
    )
    print(
        "Viewer                   : "
        f"{'disabled (headless)' if args.headless else 'enabled'}"
    )
    print(
        "Rate-bias compensation   : "
        f"{trajectory.bias_compensation}"
    )
    print(
        "Rate-bias gain           : "
        f"{RATE_BIAS_GAIN_M_PER_DEGPS:.8f} m/(deg/s)"
    )
    print(
        "Center tool position     : "
        f"{CENTER_POSITION_W} m"
    )
    print(
        "XYZ amplitudes           : "
        f"{TRANSLATION_AMPLITUDE_M} m"
    )
    print(
        "RPY amplitudes           : "
        f"{EULER_AMPLITUDE_DEG} deg"
    )
    print(
        "XYZ periods              : "
        f"{TRANSLATION_PERIOD_S} s"
    )
    print(
        "RPY periods              : "
        f"{EULER_PERIOD_S} s"
    )
    print(
        "Robustness case          : "
        f"{robustness.name}"
    )
    print(
        "Plant mass scale         : "
        f"{robustness.mass_scale:.3f}"
    )
    print(
        "Plant inertia scale      : "
        f"{robustness.inertia_scale:.3f}"
    )
    print(
        "Servo time scale         : "
        f"{robustness.servo_time_scale:.3f}"
    )
    print(
        "Thrust effectiveness     : "
        f"{robustness.thrust_effectiveness_scale:.3f}"
    )
    print(
        "Rotor effectiveness      : "
        f"{robustness.rotor_effectiveness}"
    )
    print(
        "Reaction torque scale    : "
        f"{robustness.reaction_torque_scale:.3f}"
    )
    print(
        "External force W [N]     : "
        f"{robustness.disturbance_force_W_N}"
    )
    print(
        "External torque W [Nm]   : "
        f"{robustness.disturbance_torque_W_Nm}"
    )

    print()
    print(
        "Trajectory schedule:"
    )

    for line in (
        trajectory.schedule_lines()
    ):
        print(
            "  "
            + line
        )

    if not JSON_FILE.exists():
        raise FileNotFoundError(
            "Built effector NMPC solver was not found:\n"
            f"{JSON_FILE}"
        )

    model = (
        mujoco.MjModel.from_xml_path(
            str(
                XML_PATH
            )
        )
    )

    data = mujoco.MjData(
        model
    )

    (
        nominal_plant_mass_kg,
        actual_plant_mass_kg,
        nominal_plant_inertia_diag,
        actual_plant_inertia_diag,
    ) = apply_plant_parameter_scales(
        model,
        data,
        mass_scale=(
            robustness.mass_scale
        ),
        inertia_scale=(
            robustness.inertia_scale
        ),
    )

    print()
    print(
        "NMPC model mass          : "
        f"{NOMINAL_MODEL_MASS_KG:.4f} kg"
    )
    print(
        "Nominal MJCF mass        : "
        f"{nominal_plant_mass_kg:.4f} kg"
    )
    print(
        "Actual MuJoCo mass       : "
        f"{actual_plant_mass_kg:.4f} kg"
    )
    print(
        "NMPC model inertia       : "
        f"{NOMINAL_MODEL_INERTIA_DIAG} kg m^2"
    )
    print(
        "Nominal MJCF inertia     : "
        f"{nominal_plant_inertia_diag} kg m^2"
    )
    print(
        "Actual MuJoCo inertia    : "
        f"{actual_plant_inertia_diag} kg m^2"
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

    controller = (
        EffectorNmpcController(
            build_directory=BUILD_DIR,
            json_file=JSON_FILE,
            generate=False,
            build=False,
            verbose=False,
        )
    )

    alpha = reset_vehicle(
        model,
        data,
        trajectory,
    )

    print()
    print(
        "Initial physical alpha   : "
        f"{np.round(np.degrees(alpha), 2)} deg"
    )

    logger = Logger(
        trajectory=trajectory,
        robustness_config=robustness,
        plant_mass_scale=(
            robustness.mass_scale
        ),
        plant_inertia_scale=(
            robustness.inertia_scale
        ),
        nominal_plant_mass_kg=(
            nominal_plant_mass_kg
        ),
        actual_plant_mass_kg=(
            actual_plant_mass_kg
        ),
        nominal_plant_inertia_diag=(
            nominal_plant_inertia_diag
        ),
        actual_plant_inertia_diag=(
            actual_plant_inertia_diag
        ),
    )

    completed = False
    control_steps = 0

    visual = None
    if args.cad_visual:
        visual = CadManeuverVisual()

        if args.render_mp4 is not None:
            visual.open_video(
                args.render_mp4,
                fps=args.video_fps,
                size=tuple(args.video_size),
            )

    def mirror_to_visual(
        display_state: np.ndarray,
        time_s: float,
    ) -> np.ndarray:
        display_reference = trajectory.true_sample(
            time_s
        )
        display_effector = body_to_end_effector_kinematics(
            position_W=display_state[0:3],
            velocity_W=display_state[3:6],
            quaternion_WB=display_state[6:10],
            omega_B=display_state[10:13],
            transform=DEFAULT_END_EFFECTOR,
        )
        visual.set_state(
            display_state,
            target_W=display_reference.position_W,
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
            float(data.time),
        )
        visual.write_frame(
            float(data.time),
            tool_W,
        )

    next_viewer_time = 0.0
    next_terminal_time = 0.0

    def run_control_loop(
        viewer=None,
    ) -> bool:
        nonlocal alpha
        nonlocal control_steps

        next_terminal_time = 0.0
        next_viewer_time = 0.0

        while (
            data.time
            < trajectory.end_time_s
            and (
                viewer is None
                or viewer.is_running()
            )
        ):
            wall_start = time.perf_counter()
            current_time = float(data.time)

            state = full_state_from_mujoco(
                data,
                alpha,
            )

            true_reference = trajectory.true_sample(
                current_time
            )
            controller_reference = trajectory.sample(
                current_time
            )
            compensation = trajectory.bias_compensation_state(
                current_time
            )
            euler = trajectory.euler_state(
                current_time
            )

            (
                activity_window,
                _,
                _,
                _,
            ) = trajectory.activity_window(
                current_time
            )

            effector = body_to_end_effector_kinematics(
                position_W=state[0:3],
                velocity_W=state[3:6],
                quaternion_WB=state[6:10],
                omega_B=state[10:13],
                transform=DEFAULT_END_EFFECTOR,
            )

            stage0_reference = make_effector_stage_reference(
                controller_reference,
                previous_effector_quaternion=(
                    effector.quaternion_WE
                ),
                previous_body_quaternion=(
                    state[6:10]
                ),
                previous_alpha=(
                    state[13:17]
                ),
                transform=DEFAULT_END_EFFECTOR,
            )

            try:
                controller_start = time.perf_counter()

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
                true_reference=true_reference,
                controller_reference=controller_reference,
                compensation=compensation,
                euler=euler,
                stage0_reference=stage0_reference,
                command=command,
                total_controller_time_s=(
                    total_controller_time_s
                ),
                activity_window=activity_window,
            )

            if (
                current_time
                + 1e-12
                >= next_terminal_time
            ):
                position_error_mm = (
                    1000.0
                    * np.linalg.norm(
                        effector.position_W
                        - true_reference.position_W
                    )
                )

                attitude_error = quaternion_error_deg(
                    effector.quaternion_WE,
                    true_reference.quaternion_WE,
                )

                print(
                    f"t={current_time:6.2f}s | "
                    f"p_ref={np.round(true_reference.position_W, 2)}m | "
                    f"rpy_ref={np.round(np.degrees(euler.angles_rad), 1)}deg | "
                    f"pos_err={position_error_mm:6.1f}mm | "
                    f"att_err={attitude_error:5.2f}deg | "
                    f"thrust_max={np.max(command.thrust):5.2f}N | "
                    f"ctrl={1000.0 * total_controller_time_s:5.2f}ms"
                )

                next_terminal_time += (
                    1.0 / TERMINAL_HZ
                )

            for _ in range(
                physics_steps_per_control
            ):
                alpha = step_robust_plant(
                    model,
                    data,
                    alpha,
                    command.thrust,
                    command.alpha_cmd,
                    config=robustness,
                    time_s=float(data.time),
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
                display_state = full_state_from_mujoco(
                    data,
                    alpha,
                )
                display_reference = trajectory.true_sample(
                    float(data.time)
                )

                with viewer.lock():
                    if visual is not None:
                        mirror_to_visual(
                            display_state,
                            float(data.time),
                        )
                        viewer.user_scn.ngeom = 0
                        visual.decorate(
                            viewer.user_scn
                        )
                    else:
                        update_target_marker(
                            viewer,
                            display_reference.position_W,
                        )

                        update_servo_overlay(
                            viewer,
                            body_position_W=(
                                display_state[0:3]
                            ),
                            body_quaternion_WB=(
                                display_state[6:10]
                            ),
                            alpha=(
                                display_state[13:17]
                            ),
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
                viewer.cam.distance = 2.4 if visual is None else 2.0
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -18.0 if visual is None else -15.0
                viewer.cam.lookat[:] = (
                    CENTER_POSITION_W
                    if visual is None
                    else visual.cam_main.lookat
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
            "GENERAL 6-DOF BENCHMARK FINISHED"
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
        print(
            "CSV:"
        )
        print(
            logger.csv_path
        )

        print()
        print(
            "Summary:"
        )
        print(
            logger.summary_path
        )


if __name__ == "__main__":
    main()
