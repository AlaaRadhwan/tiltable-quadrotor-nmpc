from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    thrust_allocation_matrix,
)
from tiltable_quadrotor_rl.control.dynamic_allocation import (
    allocate_dynamic_horizon,
)
from tiltable_quadrotor_rl.robot.tilt_servo import (
    first_order_servo_step,
    write_tilt_angles,
)


# ============================================================
# Vehicle / timing
# ============================================================

MASS = 3.039
GRAVITY = 9.81

WEIGHT = MASS * GRAVITY
HOVER_THRUST = WEIGHT / 4.0

PHYSICS_DT = 0.002       # 500 Hz
CONTROL_DT = 0.010       # 100 Hz
HORIZON_STEPS = 3        # 3 x 10 ms = 30 ms

TOTAL_TIME = 2.0

PHYSICS_STEPS_PER_CONTROL = int(
    round(
        CONTROL_DT / PHYSICS_DT
    )
)

assert np.isclose(
    PHYSICS_STEPS_PER_CONTROL
    * PHYSICS_DT,
    CONTROL_DT,
)


# ============================================================
# MuJoCo model
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

XML_PATH = (
    ROOT
    / "src"
    / "tiltable_quadrotor_rl"
    / "robot"
    / "mjcf"
    / "tiltable_quadrotor.xml"
)

model = mujoco.MjModel.from_xml_path(
    str(XML_PATH)
)

data = mujoco.MjData(model)

mujoco.mj_resetData(
    model,
    data,
)

# Fixed vehicle body.
#
# We still want to isolate the actuator/allocation layer
# before allowing the complete vehicle to move.
data.qpos[0:3] = [
    0.0,
    0.0,
    1.0,
]

data.qpos[3:7] = [
    1.0,
    0.0,
    0.0,
    0.0,
]

data.qvel[:] = 0.0


# ============================================================
# Desired wrench schedule
# ============================================================

HOVER_WRENCH = np.array([
    0.0,
    0.0,
    WEIGHT,
    0.0,
    0.0,
    0.0,
])

FORWARD_WRENCH = np.array([
    20.0,
    0.0,
    WEIGHT,
    0.0,
    0.0,
    0.0,
])


def desired_wrench_at_time(
    time: float,
) -> np.ndarray:

    if time < 0.5:
        return HOVER_WRENCH

    if time < 1.5:
        return FORWARD_WRENCH

    return HOVER_WRENCH


# ============================================================
# Initial actuator state
# ============================================================

alpha = np.zeros(
    4,
    dtype=float,
)

alpha_cmd = np.zeros(
    4,
    dtype=float,
)

thrust_cmd = np.full(
    4,
    HOVER_THRUST,
    dtype=float,
)


# Diagnostics from most recent optimization.
planned_first_start_wrench = (
    HOVER_WRENCH.copy()
)

planned_first_end_wrench = (
    HOVER_WRENCH.copy()
)

planned_terminal_wrench = (
    HOVER_WRENCH.copy()
)

optimizer_nfev = 0
optimizer_cost = 0.0


# ============================================================
# Logs
# ============================================================

logs = []

max_model_error = 0.0
max_function_evaluations = 0

num_steps = int(
    round(
        TOTAL_TIME / PHYSICS_DT
    )
)


# ============================================================
# Simulation
# ============================================================

for step in range(
    num_steps + 1
):

    time = (
        step * PHYSICS_DT
    )

    desired_wrench = (
        desired_wrench_at_time(
            time
        )
    )

    # --------------------------------------------------------
    # Receding-horizon allocator: 100 Hz
    # --------------------------------------------------------

    if (
        step
        % PHYSICS_STEPS_PER_CONTROL
        == 0
    ):

        previous_thrust = (
            thrust_cmd.copy()
        )

        previous_alpha_cmd = (
            alpha_cmd.copy()
        )

        (
            thrust_cmd,
            alpha_cmd,
            info,
        ) = allocate_dynamic_horizon(
            desired_wrench=desired_wrench,
            current_alpha=alpha,
            previous_thrust=previous_thrust,
            previous_alpha_cmd=previous_alpha_cmd,
            horizon_steps=HORIZON_STEPS,
            control_dt=CONTROL_DT,
        )

        if not info["success"]:
            raise RuntimeError(
                "Dynamic horizon allocator "
                "failed at "
                f"t = {time:.3f} s"
            )

        optimizer_nfev = int(
            info[
                "function_evaluations"
            ]
        )

        optimizer_cost = float(
            info["cost"]
        )

        max_function_evaluations = max(
            max_function_evaluations,
            optimizer_nfev,
        )

        # ----------------------------------------------------
        # Store what the optimizer currently predicts.
        #
        # Interval 0 START:
        #     what the first command produces immediately.
        #
        # Interval 0 END:
        #     predicted wrench after 10 ms.
        #
        # Final interval END:
        #     predicted wrench at the end of the 30 ms horizon.
        # ----------------------------------------------------

        planned_first_start_wrench = (
            info[
                "wrench_start_trajectory"
            ][0].copy()
        )

        planned_first_end_wrench = (
            info[
                "wrench_end_trajectory"
            ][0].copy()
        )

        planned_terminal_wrench = (
            info[
                "wrench_end_trajectory"
            ][-1].copy()
        )

    # --------------------------------------------------------
    # Write CURRENT physical servo angles to MuJoCo.
    # --------------------------------------------------------

    write_tilt_angles(
        model,
        data,
        alpha,
    )

    # Fixed base pose.
    data.qpos[0:3] = [
        0.0,
        0.0,
        1.0,
    ]

    data.qpos[3:7] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]

    data.qvel[:] = 0.0

    # Current applied thrust command.
    data.ctrl[:] = thrust_cmd

    mujoco.mj_forward(
        model,
        data,
    )

    actual_wrench = (
        data.qfrc_actuator[
            :6
        ].copy()
    )

    # --------------------------------------------------------
    # Analytical-vs-MuJoCo consistency check
    # --------------------------------------------------------

    B = thrust_allocation_matrix(
        alpha
    )

    analytical_wrench = (
        B @ thrust_cmd
    )

    model_error = float(
        np.max(
            np.abs(
                actual_wrench
                - analytical_wrench
            )
        )
    )

    max_model_error = max(
        max_model_error,
        model_error,
    )

    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------

    logs.append({
        "time": time,

        "desired_wrench": (
            desired_wrench.copy()
        ),

        "actual_wrench": (
            actual_wrench.copy()
        ),

        "alpha": (
            alpha.copy()
        ),

        "alpha_cmd": (
            alpha_cmd.copy()
        ),

        "thrust": (
            thrust_cmd.copy()
        ),

        "planned_first_start_wrench": (
            planned_first_start_wrench.copy()
        ),

        "planned_first_end_wrench": (
            planned_first_end_wrench.copy()
        ),

        "planned_terminal_wrench": (
            planned_terminal_wrench.copy()
        ),

        "nfev": (
            optimizer_nfev
        ),

        "cost": (
            optimizer_cost
        ),
    })

    # --------------------------------------------------------
    # Advance physical servo model by 2 ms
    # --------------------------------------------------------

    if step < num_steps:

        alpha = (
            first_order_servo_step(
                alpha,
                alpha_cmd,
                PHYSICS_DT,
            )
        )


# ============================================================
# Helper
# ============================================================

def row_at_time(
    wanted_time: float,
):

    index = int(
        round(
            wanted_time
            / PHYSICS_DT
        )
    )

    return logs[index]


# ============================================================
# General summary
# ============================================================

print()
print("=" * 96)
print(
    "3-STEP / 30 ms "
    "RECEDING-HORIZON DYNAMIC ALLOCATOR"
)
print("=" * 96)

print()

print(
    f"Servo simulation rate : "
    f"{1.0 / PHYSICS_DT:.0f} Hz"
)

print(
    f"Allocator rate        : "
    f"{1.0 / CONTROL_DT:.0f} Hz"
)

print(
    f"Horizon steps         : "
    f"{HORIZON_STEPS}"
)

print(
    f"Total horizon         : "
    f"{1000.0 * CONTROL_DT * HORIZON_STEPS:.0f} ms"
)

print()

print(
    "Maximum B(alpha) vs MuJoCo error:",
    f"{max_model_error:.3e}",
)

print(
    "Maximum optimizer function evaluations:",
    max_function_evaluations,
)


# ============================================================
# Actual wrench snapshots
# ============================================================

snapshot_times = [
    0.490,
    0.500,
    0.510,
    0.520,
    0.530,
    0.550,
    0.600,
    0.750,
    1.000,
    1.490,
    1.500,
    1.510,
    1.520,
    1.530,
    1.550,
    1.600,
    1.750,
    1.990,
]

print()
print("-" * 96)
print("ACTUAL WRENCH THROUGH TIME")
print("-" * 96)

print()

print(
    f"{'t [s]':>7} "
    f"{'Fx_des':>8} "
    f"{'Fx':>9} "
    f"{'Fz':>9} "
    f"{'Tx':>9} "
    f"{'Ty':>9} "
    f"{'Tz':>9} "
    f"{'|a|max':>9} "
    f"{'|cmd|max':>10}"
)

print("-" * 96)

for time in snapshot_times:

    row = row_at_time(
        time
    )

    desired = (
        row[
            "desired_wrench"
        ]
    )

    wrench = (
        row[
            "actual_wrench"
        ]
    )

    max_alpha_deg = float(
        np.max(
            np.abs(
                np.rad2deg(
                    row["alpha"]
                )
            )
        )
    )

    max_cmd_deg = float(
        np.max(
            np.abs(
                np.rad2deg(
                    row[
                        "alpha_cmd"
                    ]
                )
            )
        )
    )

    print(
        f"{time:7.3f} "
        f"{desired[0]:8.3f} "
        f"{wrench[0]:9.3f} "
        f"{wrench[2]:9.3f} "
        f"{wrench[3]:9.4f} "
        f"{wrench[4]:9.4f} "
        f"{wrench[5]:9.4f} "
        f"{max_alpha_deg:9.2f} "
        f"{max_cmd_deg:10.2f}"
    )


# ============================================================
# Detailed controller snapshots
# ============================================================

detailed_times = [
    0.500,
    0.510,
    0.520,
    0.530,
    0.550,
    1.500,
    1.510,
    1.520,
    1.530,
    1.550,
]

print()
print("-" * 96)
print("DETAILED RECEDING-HORIZON SNAPSHOTS")
print("-" * 96)

for time in detailed_times:

    row = row_at_time(
        time
    )

    print()
    print(
        f"t = {time:.3f} s"
    )

    print(
        "  actual alpha [deg]     :",
        np.round(
            np.rad2deg(
                row["alpha"]
            ),
            3,
        ),
    )

    print(
        "  command alpha [deg]    :",
        np.round(
            np.rad2deg(
                row["alpha_cmd"]
            ),
            3,
        ),
    )

    print(
        "  thrust [N]             :",
        np.round(
            row["thrust"],
            4,
        ),
    )

    print(
        "  actual wrench          :",
        np.round(
            row[
                "actual_wrench"
            ],
            5,
        ),
    )

    print(
        "  planned immediate      :",
        np.round(
            row[
                "planned_first_start_wrench"
            ],
            5,
        ),
    )

    print(
        "  planned +10 ms         :",
        np.round(
            row[
                "planned_first_end_wrench"
            ],
            5,
        ),
    )

    print(
        "  planned horizon end    :",
        np.round(
            row[
                "planned_terminal_wrench"
            ],
            5,
        ),
    )

    print(
        "  optimizer cost         :",
        f"{row['cost']:.8f}",
    )

    print(
        "  optimizer nfev         :",
        row["nfev"],
    )


# ============================================================
# Quantitative metrics
# ============================================================

times = np.array([
    row["time"]
    for row in logs
])

desired_array = np.stack([
    row["desired_wrench"]
    for row in logs
])

actual_array = np.stack([
    row["actual_wrench"]
    for row in logs
])

error_array = (
    actual_array
    - desired_array
)


forward_mask = (
    (times >= 0.5)
    & (times < 1.5)
)

settled_forward_mask = (
    (times >= 0.6)
    & (times < 1.5)
)

return_mask = (
    times >= 1.5
)


fx_rms_settled = float(
    np.sqrt(
        np.mean(
            error_array[
                settled_forward_mask,
                0,
            ] ** 2
        )
    )
)

fz_rms_settled = float(
    np.sqrt(
        np.mean(
            error_array[
                settled_forward_mask,
                2,
            ] ** 2
        )
    )
)


peak_fx_error_forward = float(
    np.max(
        np.abs(
            error_array[
                forward_mask,
                0,
            ]
        )
    )
)

peak_fz_error_forward = float(
    np.max(
        np.abs(
            error_array[
                forward_mask,
                2,
            ]
        )
    )
)


moment_norm = np.linalg.norm(
    actual_array[:, 3:6],
    axis=1,
)

peak_moment_forward = float(
    np.max(
        moment_norm[
            forward_mask
        ]
    )
)

peak_moment_return = float(
    np.max(
        moment_norm[
            return_mask
        ]
    )
)


# Return-to-hover force errors.
peak_fx_error_return = float(
    np.max(
        np.abs(
            error_array[
                return_mask,
                0,
            ]
        )
    )
)

peak_fz_error_return = float(
    np.max(
        np.abs(
            error_array[
                return_mask,
                2,
            ]
        )
    )
)


# ============================================================
# Compare against previous endpoint-preview experiment
# ============================================================

OLD_PREVIEW_PEAK_FZ_ERROR = (
    7.900419
)

OLD_PREVIEW_PEAK_MOMENT_FORWARD = (
    0.362649
)

OLD_PREVIEW_PEAK_MOMENT_RETURN = (
    0.117867
)


fz_improvement_percent = (
    100.0
    * (
        OLD_PREVIEW_PEAK_FZ_ERROR
        - peak_fz_error_forward
    )
    / OLD_PREVIEW_PEAK_FZ_ERROR
)

forward_moment_improvement_percent = (
    100.0
    * (
        OLD_PREVIEW_PEAK_MOMENT_FORWARD
        - peak_moment_forward
    )
    / OLD_PREVIEW_PEAK_MOMENT_FORWARD
)

return_moment_improvement_percent = (
    100.0
    * (
        OLD_PREVIEW_PEAK_MOMENT_RETURN
        - peak_moment_return
    )
    / OLD_PREVIEW_PEAK_MOMENT_RETURN
)


print()
print("-" * 96)
print("SUMMARY METRICS")
print("-" * 96)

print()

print(
    "Settled forward Fx RMS error "
    "(0.6–1.5 s):",
    f"{fx_rms_settled:.6f} N",
)

print(
    "Settled forward Fz RMS error "
    "(0.6–1.5 s):",
    f"{fz_rms_settled:.6f} N",
)

print()

print(
    "Peak forward-transition Fx error:",
    f"{peak_fx_error_forward:.6f} N",
)

print(
    "Peak forward-transition Fz error:",
    f"{peak_fz_error_forward:.6f} N",
)

print(
    "Peak moment norm during forward segment:",
    f"{peak_moment_forward:.6f} N m",
)

print()

print(
    "Peak return-to-hover Fx error:",
    f"{peak_fx_error_return:.6f} N",
)

print(
    "Peak return-to-hover Fz error:",
    f"{peak_fz_error_return:.6f} N",
)

print(
    "Peak moment norm after return-to-hover:",
    f"{peak_moment_return:.6f} N m",
)

print()
print("-" * 96)
print("CHANGE VS 30 ms ENDPOINT-PREVIEW BASELINE")
print("-" * 96)

print()

print(
    "Forward Fz peak-error improvement:",
    f"{fz_improvement_percent:.2f} %",
)

print(
    "Forward peak-moment improvement:",
    f"{forward_moment_improvement_percent:.2f} %",
)

print(
    "Return peak-moment improvement:",
    f"{return_moment_improvement_percent:.2f} %",
)

print()
print("=" * 96)
print("SIMULATION COMPLETE")
print("=" * 96)