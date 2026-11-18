from __future__ import annotations

import numpy as np

from tiltable_quadrotor_rl.control.dynamic_allocation import (
    allocate_dynamic_horizon,
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


(
    thrust,
    alpha_cmd,
    info,
) = allocate_dynamic_horizon(
    desired_wrench=desired_wrench,
    current_alpha=current_alpha,
    previous_thrust=previous_thrust,
    previous_alpha_cmd=previous_alpha_cmd,
)


print()
print("=" * 92)
print("3-STEP DYNAMIC HORIZON ALLOCATOR")
print("=" * 92)

print()
print(
    "Optimizer success:",
    info["success"],
)

print(
    "Function evaluations:",
    info[
        "function_evaluations"
    ],
)

print(
    "Cost:",
    info["cost"],
)

print()
print(
    "FIRST COMMAND TO APPLY"
)

print(
    "Thrust [N]:",
    np.round(
        thrust,
        5,
    ),
)

print(
    "Tilt command [deg]:",
    np.round(
        np.rad2deg(
            alpha_cmd
        ),
        3,
    ),
)


for step in range(3):

    print()
    print("-" * 92)

    print(
        f"PLANNED INTERVAL {step}"
    )

    print(
        f"Time:"
        f" {10 * step} -> "
        f"{10 * (step + 1)} ms"
    )

    print(
        "Thrust [N]:",
        np.round(
            info[
                "thrust_plan"
            ][step],
            5,
        ),
    )

    print(
        "Tilt command [deg]:",
        np.round(
            np.rad2deg(
                info[
                    "alpha_cmd_plan"
                ][step]
            ),
            3,
        ),
    )

    print(
        "Physical alpha START [deg]:",
        np.round(
            np.rad2deg(
                info[
                    "alpha_trajectory"
                ][step]
            ),
            3,
        ),
    )

    print(
        "Physical alpha END [deg]:",
        np.round(
            np.rad2deg(
                info[
                    "alpha_trajectory"
                ][step + 1]
            ),
            3,
        ),
    )

    print(
        "Wrench START:",
        np.round(
            info[
                "wrench_start_trajectory"
            ][step],
            5,
        ),
    )

    print(
        "Wrench END:",
        np.round(
            info[
                "wrench_end_trajectory"
            ][step],
            5,
        ),
    )


print()
print("=" * 92)
