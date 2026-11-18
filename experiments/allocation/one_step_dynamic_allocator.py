from __future__ import annotations

import numpy as np

from tiltable_quadrotor_rl.control.dynamic_allocation import (
    allocate_dynamic_one_step,
)


MASS = 3.039
GRAVITY = 9.81

WEIGHT = MASS * GRAVITY
HOVER_THRUST = WEIGHT / 4.0


# ============================================================
# Initial actuator state
# ============================================================

current_alpha = np.zeros(
    4,
    dtype=float,
)

previous_alpha_cmd = np.zeros(
    4,
    dtype=float,
)

previous_thrust = np.full(
    4,
    HOVER_THRUST,
    dtype=float,
)


# ============================================================
# Desired wrench
#
# +20 N body-X force while maintaining hover support
# and zero body moments.
# ============================================================

desired_wrench = np.array([
    20.0,
    0.0,
    WEIGHT,
    0.0,
    0.0,
    0.0,
])


# ============================================================
# Run one-step dynamic allocator
# ============================================================

(
    thrust,
    alpha_cmd,
    info,
) = allocate_dynamic_one_step(
    desired_wrench=desired_wrench,
    current_alpha=current_alpha,
    previous_thrust=previous_thrust,
    previous_alpha_cmd=previous_alpha_cmd,
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 72)
print("ONE-STEP DYNAMIC ALLOCATOR")
print("=" * 72)

print()
print("Desired wrench:")
print(desired_wrench)

print()
print("Current rotor angles [deg]:")
print(
    np.rad2deg(
        current_alpha
    )
)

print()
print("Previous thrust [N]:")
print(previous_thrust)

print()
print("Previous tilt commands [deg]:")
print(
    np.rad2deg(
        previous_alpha_cmd
    )
)

print()
print("-" * 72)

print()
print("New thrust commands [N]:")
print(thrust)

print()
print("New tilt commands [deg]:")
print(
    np.rad2deg(
        alpha_cmd
    )
)

print()
print("Predicted rotor angles after 10 ms [deg]:")
print(
    np.rad2deg(
        info["predicted_alpha"]
    )
)

print()
print("Predicted wrench after 10 ms:")
print(
    info["predicted_wrench"]
)

print()
print("Wrench error:")
print(
    info["wrench_error"]
)

print()
print(
    "Wrench error norm:",
    info["wrench_error_norm"],
)

print()
print(
    "Optimizer success:",
    info["success"],
)

print(
    "Function evaluations:",
    info["function_evaluations"],
)

print(
    "Optimizer message:",
    info["message"],
)

print()
print("=" * 72)