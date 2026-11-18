from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    thrust_allocation_matrix,
)
from tiltable_quadrotor_rl.control.dynamic_allocation import (
    allocate_dynamic_one_step,
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

PHYSICS_DT = 0.002       # 500 Hz servo simulation
CONTROL_DT = 0.010       # 100 Hz allocator
PREVIEW_DT = 0.030       # allocator looks 30 ms ahead

TOTAL_TIME = 2.0

CONTROL_STEPS = int(
    round(
        CONTROL_DT / PHYSICS_DT
    )
)

assert np.isclose(
    CONTROL_STEPS * PHYSICS_DT,
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

# Keep the body at a fixed identity pose.
#
# We are deliberately NOT integrating the vehicle rigid-body
# dynamics in this experiment. We only want to test actuator
# allocation + servo dynamics.
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

predicted_wrench = (
    HOVER_WRENCH.copy()
)

optimizer_success = True
optimizer_nfev = 0


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
    # High-level allocator: 100 Hz
    # --------------------------------------------------------

    if step % CONTROL_STEPS == 0:

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
        ) = allocate_dynamic_one_step(
            desired_wrench=desired_wrench,
            current_alpha=alpha,
            previous_thrust=previous_thrust,
            previous_alpha_cmd=previous_alpha_cmd,

            # IMPORTANT:
            # Controller still runs every 10 ms.
            # This is only the look-ahead time.
            dt=PREVIEW_DT,
        )

        optimizer_success = (
            bool(info["success"])
        )

        optimizer_nfev = int(
            info[
                "function_evaluations"
            ]
        )

        predicted_wrench = (
            info[
                "predicted_wrench"
            ].copy()
        )

        max_function_evaluations = max(
            max_function_evaluations,
            optimizer_nfev,
        )

        if not optimizer_success:
            raise RuntimeError(
                "Dynamic allocator failed at "
                f"t = {time:.3f} s"
            )

    # --------------------------------------------------------
    # Write current physical servo angles into MuJoCo
    # --------------------------------------------------------

    write_tilt_angles(
        model,
        data,
        alpha,
    )

    # Keep base fixed.
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

    # Rotor thrust is assumed instantaneous for now.
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
    # Analytical instantaneous wrench
    #
    # This is only a model-consistency check:
    #
    #     B(alpha_actual) @ thrust
    #
    # should match MuJoCo.
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
    # Log current state
    # --------------------------------------------------------

    logs.append({
        "time": time,

        "desired_wrench": (
            desired_wrench.copy()
        ),

        "actual_wrench": (
            actual_wrench.copy()
        ),

        # This is the allocator's prediction PREVIEW_DT
        # into the future if the selected command were held.
        "predicted_wrench": (
            predicted_wrench.copy()
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

        "nfev": (
            optimizer_nfev
        ),
    })

    # --------------------------------------------------------
    # Advance explicit servo by one 2 ms physics step
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
# Helpers
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
# Summary
# ============================================================

print()
print("=" * 92)
print("30 ms RECEDING-PREVIEW DYNAMIC ALLOCATOR")
print("=" * 92)

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
    f"Preview horizon       : "
    f"{1000.0 * PREVIEW_DT:.0f} ms"
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
# Time-history snapshots
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
print("-" * 92)
print("ACTUAL WRENCH THROUGH TIME")
print("-" * 92)

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

print("-" * 92)

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
# Detailed control snapshots
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
print("-" * 92)
print("DETAILED CONTROLLER SNAPSHOTS")
print("-" * 92)

for time in detailed_times:

    row = row_at_time(
        time
    )

    print()
    print(
        f"t = {time:.3f} s"
    )

    print(
        "  actual alpha [deg] :",
        np.round(
            np.rad2deg(
                row["alpha"]
            ),
            3,
        ),
    )

    print(
        "  command alpha [deg]:",
        np.round(
            np.rad2deg(
                row["alpha_cmd"]
            ),
            3,
        ),
    )

    print(
        "  thrust [N]         :",
        np.round(
            row["thrust"],
            4,
        ),
    )

    print(
        "  actual wrench      :",
        np.round(
            row[
                "actual_wrench"
            ],
            5,
        ),
    )

    print(
        "  predicted +30ms    :",
        np.round(
            row[
                "predicted_wrench"
            ],
            5,
        ),
    )

    print(
        "  optimizer nfev     :",
        row["nfev"],
    )


# ============================================================
# Quantitative transition metrics
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


forward_error = (
    actual_array
    - desired_array
)


fx_rms_settled = float(
    np.sqrt(
        np.mean(
            forward_error[
                settled_forward_mask,
                0,
            ] ** 2
        )
    )
)

fz_rms_settled = float(
    np.sqrt(
        np.mean(
            forward_error[
                settled_forward_mask,
                2,
            ] ** 2
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

peak_fx_error_forward = float(
    np.max(
        np.abs(
            forward_error[
                forward_mask,
                0,
            ]
        )
    )
)

peak_fz_error_forward = float(
    np.max(
        np.abs(
            forward_error[
                forward_mask,
                2,
            ]
        )
    )
)

print()
print("-" * 92)
print("SUMMARY METRICS")
print("-" * 92)

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

print(
    "Peak moment norm after return-to-hover:",
    f"{peak_moment_return:.6f} N m",
)

print()

print("=" * 92)
print("SIMULATION COMPLETE")
print("=" * 92)