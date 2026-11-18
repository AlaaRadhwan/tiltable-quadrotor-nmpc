from __future__ import annotations

import numpy as np

from tiltable_quadrotor_rl.control.dynamic_allocation import (
    allocate_dynamic_one_step,
)


MASS = 3.039
G = 9.81

WEIGHT = MASS * G
HOVER_THRUST = WEIGHT / 4.0


desired_wrench = np.array([
    20.0,
    0.0,
    WEIGHT,
    0.0,
    0.0,
    0.0,
])

current_alpha = np.zeros(4)

previous_thrust = np.full(
    4,
    HOVER_THRUST,
)

previous_alpha_cmd = np.zeros(4)


horizons = [
    0.010,
    0.020,
    0.030,
    0.050,
]


print()
print("=" * 90)
print("DYNAMIC ALLOCATOR — PREDICTION HORIZON SWEEP")
print("=" * 90)


for horizon in horizons:

    (
        thrust,
        alpha_cmd,
        info,
    ) = allocate_dynamic_one_step(
        desired_wrench=desired_wrench,
        current_alpha=current_alpha,
        previous_thrust=previous_thrust,
        previous_alpha_cmd=previous_alpha_cmd,
        dt=horizon,
    )

    print()
    print("-" * 90)
    print(
        f"Prediction horizon = "
        f"{1000*horizon:.0f} ms"
    )
    print("-" * 90)

    print(
        "Thrust [N]:"
    )
    print(
        np.round(
            thrust,
            6,
        )
    )

    print(
        "Tilt command [deg]:"
    )
    print(
        np.round(
            np.rad2deg(
                alpha_cmd
            ),
            3,
        )
    )

    print(
        "Predicted actual tilt [deg]:"
    )
    print(
        np.round(
            np.rad2deg(
                info["predicted_alpha"]
            ),
            3,
        )
    )

    print(
        "Predicted wrench:"
    )
    print(
        np.round(
            info["predicted_wrench"],
            6,
        )
    )

    print(
        "Wrench error:"
    )
    print(
        np.round(
            info["wrench_error"],
            6,
        )
    )

    print(
        "Error norm:",
        f"{info['wrench_error_norm']:.6f}",
    )

    print(
        "Function evaluations:",
        info["function_evaluations"],
    )


print()
print("=" * 90)