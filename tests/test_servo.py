from __future__ import annotations

import math

import numpy as np

from tiltable_quadrotor_rl.robot.tilt_servo import (
    DEFAULT_TIME_CONSTANT,
    first_order_servo_step,
)


DT = 0.002
TARGET = 1.0

alpha = np.zeros(4)
alpha_cmd = np.full(4, TARGET)

time = 0.0

threshold = 1.0 - math.exp(-1.0)

samples = []

while time < 0.30:

    alpha = first_order_servo_step(
        alpha,
        alpha_cmd,
        DT,
    )

    time += DT

    exact = (
        TARGET
        * (
            1.0
            - math.exp(
                -time / DEFAULT_TIME_CONSTANT
            )
        )
    )

    error = alpha[0] - exact

    samples.append(
        (
            time,
            alpha[0],
            exact,
            error,
        )
    )


print()
print("=" * 68)
print("EXPLICIT FIRST-ORDER SERVO TEST")
print("=" * 68)

print()
print(
    f"dt                 = {DT:.6f} s"
)

print(
    f"time constant      = "
    f"{DEFAULT_TIME_CONSTANT:.6f} s"
)

print(
    f"63.2% target       = "
    f"{threshold:.9f} rad"
)

print()
print(
    f"{'time':>10} "
    f"{'simulated':>14} "
    f"{'analytical':>14} "
    f"{'error':>14}"
)

print("-" * 56)

# Print useful checkpoints.
for t, simulated, exact, error in samples:

    if any(
        abs(t - x) < 1e-9
        for x in (
            0.012,
            0.024,
            0.048,
            0.096,
            0.192,
        )
    ):
        print(
            f"{t:10.3f} "
            f"{simulated:14.9f} "
            f"{exact:14.9f} "
            f"{error:14.3e}"
        )


errors = np.array(
    [
        row[3]
        for row in samples
    ]
)

max_error = np.max(
    np.abs(errors)
)

value_at_tau = next(
    simulated
    for t, simulated, _, _ in samples
    if abs(
        t - DEFAULT_TIME_CONSTANT
    ) < 1e-9
)

print()
print(
    f"alpha(0.048 s)     = "
    f"{value_at_tau:.9f} rad"
)

print(
    f"Expected           = "
    f"{threshold:.9f} rad"
)

print(
    f"Maximum curve error = "
    f"{max_error:.3e} rad"
)

assert np.isclose(
    value_at_tau,
    threshold,
    atol=1e-12,
)

assert max_error < 1e-12

print()
print("[PASS] Explicit servo exactly matches first-order model.")
print()