import numpy as np

from tiltable_quadrotor_rl.control.end_effector import (
    quaternion_conjugate_np,
    quaternion_product_np,
)
from tiltable_quadrotor_rl.control.nmpc_effector_showcase_trajectory import (
    FixedEffectorRotationShowcaseTrajectory,
)


def quaternion_physical_angle_deg(
    left: np.ndarray,
    right: np.ndarray,
) -> float:

    dot = abs(
        float(
            np.dot(
                left,
                right,
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


print()
print("=" * 72)
print("FIXED END-EFFECTOR SHOWCASE TRAJECTORY VALIDATION")
print("=" * 72)

trajectory = (
    FixedEffectorRotationShowcaseTrajectory(
        position_W=np.array(
            [
                0.0,
                0.0,
                1.5,
            ]
        )
    )
)


# ============================================================
# TEST 1 — fixed tool position
# ============================================================

print()
print("-" * 72)
print("TEST 1 — TOOL POSITION IS EXACTLY FIXED")
print("-" * 72)

times = np.linspace(
    0.0,
    trajectory.end_time_s,
    1001,
)

max_position_error = 0.0
max_velocity = 0.0
max_acceleration = 0.0

target = trajectory.position_W

for time_s in times:
    point = trajectory.sample(
        float(
            time_s
        )
    )

    max_position_error = max(
        max_position_error,
        float(
            np.linalg.norm(
                point.position_W
                - target
            )
        ),
    )

    max_velocity = max(
        max_velocity,
        float(
            np.linalg.norm(
                point.velocity_W
            )
        ),
    )

    max_acceleration = max(
        max_acceleration,
        float(
            np.linalg.norm(
                point.acceleration_W
            )
        ),
    )

assert max_position_error < 1e-15
assert max_velocity < 1e-15
assert max_acceleration < 1e-15

print(
    "Maximum tool position error : "
    f"{max_position_error:.3e} m"
)

print()
print("[PASS] Tool origin is fixed in world.")


# ============================================================
# TEST 2 — each maneuver is physically one complete turn
# ============================================================

print()
print("-" * 72)
print("TEST 2 — FULL 360-DEG ROTATIONS")
print("-" * 72)

for index, maneuver in enumerate(
    trajectory.maneuvers
):
    before = trajectory.sample(
        maneuver.start_time_s
    )

    after = trajectory.sample(
        maneuver.end_time_s
    )

    physical_angle = (
        quaternion_physical_angle_deg(
            before.quaternion_WE,
            after.quaternion_WE,
        )
    )

    # After a full turn the physical orientation is identical, even
    # though the continuous quaternion representation may have changed
    # sign.
    assert physical_angle < 1e-10

    midpoint = trajectory.sample(
        0.5
        * (
            maneuver.start_time_s
            + maneuver.end_time_s
        )
    )

    # At s=0.5, theta=pi for one-turn maneuvers. The physical
    # orientation should therefore be 180 degrees from the start.
    midpoint_angle = (
        quaternion_physical_angle_deg(
            before.quaternion_WE,
            midpoint.quaternion_WE,
        )
    )

    assert np.isclose(
        midpoint_angle,
        180.0,
        atol=1e-9,
    )

    print(
        f"Maneuver {index + 1}: "
        f"midpoint={midpoint_angle:.6f} deg, "
        f"endpoint physical error={physical_angle:.3e} deg"
    )

print()
print("[PASS] Unwrapped full rotations are preserved.")


# ============================================================
# TEST 3 — quaternion derivative agrees with omega_E
# ============================================================

print()
print("-" * 72)
print("TEST 3 — QUATERNION / ANGULAR-RATE CONSISTENCY")
print("-" * 72)

dt = 1e-6
max_qdot_error = 0.0
max_omega_dot_error = 0.0

for maneuver in trajectory.maneuvers:
    test_times = np.linspace(
        maneuver.start_time_s + 0.1,
        maneuver.end_time_s - 0.1,
        9,
    )

    for time_s in test_times:
        minus = trajectory.sample(
            float(
                time_s - dt
            )
        )

        center = trajectory.sample(
            float(
                time_s
            )
        )

        plus = trajectory.sample(
            float(
                time_s + dt
            )
        )

        # Align finite-difference quaternion signs to the center
        # representation before differentiating.
        q_minus = minus.quaternion_WE.copy()
        q_plus = plus.quaternion_WE.copy()

        if np.dot(
            q_minus,
            center.quaternion_WE,
        ) < 0.0:
            q_minus = -q_minus

        if np.dot(
            q_plus,
            center.quaternion_WE,
        ) < 0.0:
            q_plus = -q_plus

        qdot_fd = (
            q_plus
            - q_minus
        ) / (
            2.0 * dt
        )

        omega_quaternion = np.concatenate(
            [
                np.zeros(
                    1,
                    dtype=float,
                ),
                center.omega_E,
            ]
        )

        qdot_expected = (
            0.5
            * quaternion_product_np(
                center.quaternion_WE,
                omega_quaternion,
            )
        )

        # quaternion_product_np normalizes, so it is not suitable for
        # multiplying by a non-unit pure-rate quaternion. Construct the
        # Hamilton product explicitly for the derivative check.
        qw = center.quaternion_WE[0]
        qv = center.quaternion_WE[1:4]

        omega = center.omega_E

        qdot_expected = 0.5 * np.concatenate(
            [
                np.array(
                    [
                        -np.dot(
                            qv,
                            omega,
                        )
                    ]
                ),
                (
                    qw * omega
                    + np.cross(
                        qv,
                        omega,
                    )
                ),
            ]
        )

        qdot_error = float(
            np.max(
                np.abs(
                    qdot_fd
                    - qdot_expected
                )
            )
        )

        omega_dot_fd = (
            plus.omega_E
            - minus.omega_E
        ) / (
            2.0 * dt
        )

        omega_dot_error = float(
            np.max(
                np.abs(
                    omega_dot_fd
                    - center.omega_dot_E
                )
            )
        )

        max_qdot_error = max(
            max_qdot_error,
            qdot_error,
        )

        max_omega_dot_error = max(
            max_omega_dot_error,
            omega_dot_error,
        )

print(
    "Max qdot error       : "
    f"{max_qdot_error:.3e}"
)

print(
    "Max omega_dot error  : "
    f"{max_omega_dot_error:.3e}"
)

assert max_qdot_error < 1e-7
assert max_omega_dot_error < 1e-7

print()
print("[PASS] Orientation trajectory derivatives are consistent.")


# ============================================================
# TEST 4 — endpoint rates are zero
# ============================================================

print()
print("-" * 72)
print("TEST 4 — MANEUVER ENDPOINT SMOOTHNESS")
print("-" * 72)

for maneuver in trajectory.maneuvers:
    for time_s in (
        maneuver.start_time_s,
        maneuver.end_time_s,
    ):
        point = trajectory.sample(
            time_s
        )

        assert np.linalg.norm(
            point.omega_E
        ) < 1e-12

        assert np.linalg.norm(
            point.omega_dot_E
        ) < 1e-12

print()
print("[PASS] Every maneuver starts and ends at zero angular rate.")


print()
print("=" * 72)
print("ALL FIXED-EFFECTOR SHOWCASE TRAJECTORY TESTS PASSED")
print("=" * 72)
