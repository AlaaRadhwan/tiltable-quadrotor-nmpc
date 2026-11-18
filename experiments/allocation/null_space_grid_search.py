from __future__ import annotations

import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    allocation_matrix,
    wrench_to_virtual,
    virtual_to_rotor_commands,
    minimize_peak_thrust_grid,
)


MASS = 3.039
G = 9.81
MAX_THRUST = 23.0


A = allocation_matrix()


wrench = np.array(
    [
        45.0,
        0.0,
        MASS * G,
        0.0,
        0.0,
        0.0,
    ]
)


# ============================================================
# Plain pseudoinverse
# ============================================================

z0 = wrench_to_virtual(
    wrench
)

thrust0, alpha0 = virtual_to_rotor_commands(
    z0
)


# ============================================================
# Null-space optimized
# ============================================================

(
    z_best,
    c_best,
    thrust_best,
    alpha_best,
) = minimize_peak_thrust_grid(
    wrench,
)


print()
print("=" * 72)
print("NULL-SPACE THRUST OPTIMIZATION")
print("=" * 72)

print()
print("Desired wrench:")
print(wrench)


print()
print("-" * 72)
print("PSEUDOINVERSE SOLUTION")
print("-" * 72)

for i in range(4):
    print(
        f"Rotor {i + 1}: "
        f"{thrust0[i]:8.3f} N, "
        f"{np.rad2deg(alpha0[i]):8.3f} deg"
    )

print()
print(
    "Peak thrust:",
    f"{np.max(thrust0):.3f} N"
)

print(
    "Feasible:",
    bool(np.all(thrust0 <= MAX_THRUST))
)


print()
print("-" * 72)
print("NULL-SPACE OPTIMIZED SOLUTION")
print("-" * 72)

print()
print("Best c:")
print(c_best)

for i in range(4):
    print(
        f"Rotor {i + 1}: "
        f"{thrust_best[i]:8.3f} N, "
        f"{np.rad2deg(alpha_best[i]):8.3f} deg"
    )

print()
print(
    "Peak thrust:",
    f"{np.max(thrust_best):.3f} N"
)

print(
    "Feasible:",
    bool(np.all(thrust_best <= MAX_THRUST))
)


# ============================================================
# Make sure wrench did not change
# ============================================================

w_original = A @ z0
w_optimized = A @ z_best

print()
print("-" * 72)
print("WRENCH CHECK")
print("-" * 72)

print()
print("Pseudoinverse wrench:")
print(w_original)

print()
print("Optimized wrench:")
print(w_optimized)

print()
print("Difference:")
print(
    w_optimized - w_original
)

assert np.allclose(
    w_optimized,
    wrench,
    atol=1e-10,
)

print()
print(
    "[PASS] Null-space optimization changed rotor loading "
    "without changing the desired wrench."
)