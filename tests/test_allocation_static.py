from __future__ import annotations

import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    MAX_THRUST,
    allocate_exact_wrench,
    allocate_thrust_at_fixed_angles,
    allocate_wrench,
    allocation_matrix,
    null_space_basis,
    rotor_commands_to_virtual,
    virtual_to_wrench,
    wrench_to_virtual,
)


MASS = 3.039
G = 9.81

HOVER = np.array([
    0.0,
    0.0,
    MASS * G,
    0.0,
    0.0,
    0.0,
])


def assert_wrench_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = 1e-8,
) -> None:
    assert np.allclose(
        actual,
        expected,
        atol=atol,
    ), (
        "\nWrench mismatch:\n"
        f"actual   = {actual}\n"
        f"expected = {expected}\n"
        f"error    = {actual - expected}"
    )


# ============================================================
# Allocation matrix / null space
# ============================================================

print()
print("=" * 72)
print("STATIC ALLOCATION TESTS")
print("=" * 72)


A = allocation_matrix()

assert A.shape == (6, 8)
assert np.linalg.matrix_rank(A) == 6

print("[PASS] Allocation matrix has shape (6, 8) and rank 6.")


N = null_space_basis(A)

assert N.shape == (8, 2)
assert np.allclose(
    A @ N,
    0.0,
    atol=1e-10,
)

print("[PASS] Null space has dimension 2 and A @ N = 0.")


desired = np.array([
    5.0,
    0.0,
    MASS * G,
    0.0,
    0.0,
    0.0,
])

z0 = wrench_to_virtual(desired)

c = np.array([
    3.0,
    -2.0,
])

z_modified = (
    z0 + N @ c
)

assert not np.allclose(
    z_modified,
    z0,
)

assert_wrench_close(
    A @ z_modified,
    A @ z0,
)

print("[PASS] Null-space motion preserves the requested wrench.")


# ============================================================
# Exact minimum-peak allocation
# ============================================================

thrust, alpha, info = allocate_exact_wrench(
    HOVER
)

assert info["optimization_success"]
assert info["feasible"]
assert info["peak_thrust"] < MAX_THRUST

achieved = virtual_to_wrench(
    rotor_commands_to_virtual(
        thrust,
        alpha,
    )
)

assert_wrench_close(
    achieved,
    HOVER,
)

print("[PASS] Exact hover allocation is feasible and wrench-exact.")


FORWARD_45 = np.array([
    45.0,
    0.0,
    MASS * G,
    0.0,
    0.0,
    0.0,
])

thrust, alpha, info = allocate_exact_wrench(
    FORWARD_45
)

assert info["optimization_success"]
assert info["feasible"]
assert info["peak_thrust"] <= MAX_THRUST + 1e-6

achieved = virtual_to_wrench(
    rotor_commands_to_virtual(
        thrust,
        alpha,
    )
)

assert_wrench_close(
    achieved,
    FORWARD_45,
)

print(
    "[PASS] 45 N forward wrench is feasible "
    f"(peak = {info['peak_thrust']:.6f} N)."
)


FORWARD_500 = np.array([
    500.0,
    0.0,
    MASS * G,
    0.0,
    0.0,
    0.0,
])

_, _, info = allocate_exact_wrench(
    FORWARD_500
)

assert info["optimization_success"]
assert not info["feasible"]
assert info["peak_thrust"] > MAX_THRUST

print(
    "[PASS] 500 N exact forward wrench is correctly "
    f"classified infeasible "
    f"(required peak = {info['peak_thrust']:.3f} N)."
)


# ============================================================
# Hierarchical feasibility handling
# ============================================================

thrust, alpha, info = allocate_wrench(
    FORWARD_45,
    HOVER,
)

assert not info["scaled"]
assert np.isclose(
    info["scale"],
    1.0,
)
assert_wrench_close(
    info["achieved_wrench"],
    FORWARD_45,
)

assert np.max(thrust) <= (
    MAX_THRUST + 1e-6
)

print("[PASS] Feasible hierarchical request is not scaled.")


thrust, alpha, info = allocate_wrench(
    FORWARD_500,
    HOVER,
)

assert info["scaled"]
assert 0.0 < info["scale"] < 1.0
assert np.max(thrust) <= (
    MAX_THRUST + 1e-6
)

# The chosen fallback should preserve hover Fz.
assert np.isclose(
    info["achieved_wrench"][2],
    HOVER[2],
    atol=1e-8,
)

achieved_from_commands = virtual_to_wrench(
    rotor_commands_to_virtual(
        thrust,
        alpha,
    )
)

assert_wrench_close(
    achieved_from_commands,
    info["achieved_wrench"],
)

print(
    "[PASS] Infeasible hierarchical request is scaled "
    f"to s = {info['scale']:.9f}."
)


# ============================================================
# Fixed-angle thrust allocation
# ============================================================

zero_alpha = np.zeros(4)

thrust, achieved, info = (
    allocate_thrust_at_fixed_angles(
        zero_alpha,
        HOVER,
    )
)

assert info["success"]
assert_wrench_close(
    achieved,
    HOVER,
)

print("[PASS] Fixed-angle allocator reproduces hover at zero tilt.")


FORWARD_20 = np.array([
    20.0,
    0.0,
    MASS * G,
    0.0,
    0.0,
    0.0,
])

thrust, achieved, info = (
    allocate_thrust_at_fixed_angles(
        zero_alpha,
        FORWARD_20,
    )
)

# Fx is impossible at zero tilt.
assert np.isclose(
    achieved[0],
    0.0,
    atol=1e-8,
)

# But hover support should remain essentially exact.
assert np.isclose(
    achieved[2],
    MASS * G,
    atol=1e-8,
)

print(
    "[PASS] Zero-tilt allocator correctly accepts "
    "unavoidable forward-force error."
)


forward_alpha = np.deg2rad(
    np.array([
        -3.43363036,
        71.14559351,
        -3.43363036,
        -64.27833278,
    ])
)

thrust, achieved, info = (
    allocate_thrust_at_fixed_angles(
        forward_alpha,
        FORWARD_20,
    )
)

assert info["success"]
assert_wrench_close(
    achieved,
    FORWARD_20,
    atol=1e-7,
)

print(
    "[PASS] Forward rotor geometry reproduces the "
    "20 N forward wrench."
)

print()
print("=" * 72)
print("ALL STATIC ALLOCATION TESTS PASSED")
print("=" * 72)
