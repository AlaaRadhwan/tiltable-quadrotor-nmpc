import numpy as np

from tiltable_quadrotor_rl.control.nmpc_effector_paper_pitch_trajectory import (
    PaperStyleFixedEffectorPitchTrajectory,
)


trajectory = PaperStyleFixedEffectorPitchTrajectory(
    position_W=np.array(
        [
            0.0,
            0.0,
            1.5,
        ],
        dtype=float,
    )
)


print()
print("=" * 78)
print("BOUNDED-RATE PAPER-STYLE PITCH TRAJECTORY TEST")
print("=" * 78)


# ============================================================
# TEST 1 — exact maneuver endpoints
# ============================================================

print()
print("-" * 78)
print("TEST 1 — MANEUVER ENDPOINTS")
print("-" * 78)

expected = {
    0.0: 0.0,
    2.0: 0.0,
    7.0: 180.0,
    10.0: 180.0,
    20.0: -180.0,
    23.0: -180.0,
    28.0: 0.0,
    31.0: 0.0,
}

for time_s, expected_deg in expected.items():

    angle, rate, acceleration = (
        trajectory.pitch_state(
            time_s
        )
    )

    angle_deg = float(
        np.degrees(
            angle
        )
    )

    print(
        f"t={time_s:5.1f}s | "
        f"pitch={angle_deg:9.4f}deg | "
        f"rate={np.degrees(rate):9.4f}deg/s | "
        f"acc={np.degrees(acceleration):9.4f}deg/s^2"
    )

    assert np.isclose(
        angle_deg,
        expected_deg,
        atol=1e-10,
    )

    assert abs(
        rate
    ) < 1e-12

    assert abs(
        acceleration
    ) < 1e-12

print()
print("[PASS] Endpoints and hold transitions are exact.")


# ============================================================
# TEST 2 — plateau rate is bounded at 40 deg/s
# ============================================================

print()
print("-" * 78)
print("TEST 2 — RATE LIMIT")
print("-" * 78)

times = np.linspace(
    0.0,
    trajectory.end_time_s,
    20001,
)

rates = np.array(
    [
        trajectory.pitch_state(
            float(
                time_s
            )
        )[1]
        for time_s in times
    ],
    dtype=float,
)

max_rate_degps = float(
    np.max(
        np.abs(
            np.degrees(
                rates
            )
        )
    )
)

print(
    "Maximum |pitch rate|: "
    f"{max_rate_degps:.6f} deg/s"
)

assert np.isclose(
    max_rate_degps,
    40.0,
    atol=1e-6,
)

print()
print("[PASS] Peak pitch rate is exactly 40 deg/s.")


# ============================================================
# TEST 3 — expected cruise sections
# ============================================================

print()
print("-" * 78)
print("TEST 3 — CONSTANT-RATE CRUISE")
print("-" * 78)

checks = [
    # entry: accel 2.0-2.5, cruise 2.5-6.5, decel 6.5-7.0
    (3.0, 40.0),
    (5.0, 40.0),
    (6.0, 40.0),

    # full cycle: accel 10-11, cruise 11-19, decel 19-20
    (12.0, -40.0),
    (15.0, -40.0),
    (18.0, -40.0),

    # exit: accel 23-23.5, cruise 23.5-27.5, decel 27.5-28
    (24.0, 40.0),
    (26.0, 40.0),
    (27.0, 40.0),
]

for time_s, expected_rate_degps in checks:

    _, rate, acceleration = (
        trajectory.pitch_state(
            time_s
        )
    )

    rate_degps = float(
        np.degrees(
            rate
        )
    )

    acceleration_degps2 = float(
        np.degrees(
            acceleration
        )
    )

    print(
        f"t={time_s:5.1f}s | "
        f"rate={rate_degps:8.3f}deg/s | "
        f"acc={acceleration_degps2:8.3f}deg/s^2"
    )

    assert np.isclose(
        rate_degps,
        expected_rate_degps,
        atol=1e-10,
    )

    assert abs(
        acceleration_degps2
    ) < 1e-10

print()
print("[PASS] Cruise regions are constant-rate.")


# ============================================================
# TEST 4 — finite-difference derivative consistency
# ============================================================

print()
print("-" * 78)
print("TEST 4 — DERIVATIVE CONSISTENCY")
print("-" * 78)

dt = 1e-5

test_times = [
    2.20,
    2.80,
    6.80,
    10.35,
    12.0,
    19.65,
    23.20,
    24.5,
    27.80,
]

max_rate_error = 0.0
max_acc_error = 0.0

for time_s in test_times:

    angle_minus, rate_minus, _ = (
        trajectory.pitch_state(
            time_s - dt
        )
    )

    _, rate, acceleration = (
        trajectory.pitch_state(
            time_s
        )
    )

    angle_plus, rate_plus, _ = (
        trajectory.pitch_state(
            time_s + dt
        )
    )

    rate_fd = (
        angle_plus
        - angle_minus
    ) / (
        2.0 * dt
    )

    acceleration_fd = (
        rate_plus
        - rate_minus
    ) / (
        2.0 * dt
    )

    max_rate_error = max(
        max_rate_error,
        abs(
            rate_fd
            - rate
        ),
    )

    max_acc_error = max(
        max_acc_error,
        abs(
            acceleration_fd
            - acceleration
        ),
    )

print(
    "Max rate error : "
    f"{max_rate_error:.3e} rad/s"
)

print(
    "Max accel error: "
    f"{max_acc_error:.3e} rad/s^2"
)

assert max_rate_error < 1e-8
assert max_acc_error < 1e-7

print()
print("[PASS] Angle/rate/acceleration are mutually consistent.")


# ============================================================
# TEST 5 — fixed end-effector origin
# ============================================================

print()
print("-" * 78)
print("TEST 5 — FIXED END-EFFECTOR ORIGIN")
print("-" * 78)

target = np.array(
    [
        0.0,
        0.0,
        1.5,
    ],
    dtype=float,
)

max_position_error = 0.0

for time_s in np.linspace(
    0.0,
    trajectory.end_time_s,
    1001,
):

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

    assert np.linalg.norm(
        point.velocity_W
    ) == 0.0

    assert np.linalg.norm(
        point.acceleration_W
    ) == 0.0

assert max_position_error < 1e-15

print(
    "Maximum position-reference error: "
    f"{max_position_error:.3e} m"
)

print()
print("[PASS] Tool origin remains exactly fixed in world.")


print()
print("=" * 78)
print("ALL BOUNDED-RATE TRAJECTORY TESTS PASSED")
print("=" * 78)
