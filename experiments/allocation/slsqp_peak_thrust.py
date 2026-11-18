from __future__ import annotations

import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    allocation_matrix,
    wrench_to_virtual,
    virtual_to_rotor_commands,
    minimize_peak_thrust_grid,
    minimize_peak_thrust_slsqp,
)


MASS = 3.039
G = 9.81
MAX_THRUST = 23.0


wrench = np.array(
    [
        500.0,
        0.0,
        MASS * G,
        0.0,
        0.0,
        0.0,
    ]
)

z0 = wrench_to_virtual(wrench)

thrust0, alpha0 = virtual_to_rotor_commands(z0)

# Grid search

(
    z_grid,
    c_grid,
    thrust_grid,
    alpha_grid,
) = minimize_peak_thrust_grid(wrench)

# SLSQP

(
    z_slsqp,
    c_slsqp,
    thrust_slsqp,
    alpha_slsqp,
    result,
) = minimize_peak_thrust_slsqp(wrench)

# ============================================================
# Print results
# ============================================================

print()
print("=" * 72)
print("PSEUDOINVERSE")
print("=" * 72)

print("c = [0, 0]")

for i in range(4):
    print(
        f"Rotor {i + 1}: "
        f"{thrust0[i]:8.6f} N, "
        f"{np.rad2deg(alpha0[i]):9.4f} deg"
    )

print(
    f"Peak thrust: {np.max(thrust0):.6f} N"
)


print()
print("=" * 72)
print("GRID SEARCH")
print("=" * 72)

print("c:")
print(c_grid)

for i in range(4):
    print(
        f"Rotor {i + 1}: "
        f"{thrust_grid[i]:8.6f} N, "
        f"{np.rad2deg(alpha_grid[i]):9.4f} deg"
    )

print(
    f"Peak thrust: {np.max(thrust_grid):.6f} N"
)


print()
print("=" * 72)
print("SLSQP")
print("=" * 72)

print("c:")
print(c_slsqp)

for i in range(4):
    print(
        f"Rotor {i + 1}: "
        f"{thrust_slsqp[i]:8.6f} N, "
        f"{np.rad2deg(alpha_slsqp[i]):9.4f} deg"
    )

print(
    f"Peak thrust: {np.max(thrust_slsqp):.6f} N"
)

print()
print("Optimizer result:")
print("success    :", result.success)
print("message    :", result.message)
print("iterations :", result.nit)
print("evaluations:", result.nfev)
print("t          :", result.x[2])


# ============================================================
# Wrench validation
# ============================================================

A = allocation_matrix()

w_slsqp = A @ z_slsqp

print()
print("=" * 72)
print("WRENCH CHECK")
print("=" * 72)

print()
print("Desired:")
print(wrench)

print()
print("SLSQP:")
print(w_slsqp)

print()
print("Error:")
print(w_slsqp - wrench)


assert result.success

assert np.allclose(
    w_slsqp,
    wrench,
    atol=1e-10,
)

print()
print("[PASS]")