import numpy as np

from tiltable_quadrotor_rl.robot.parameters import (
    MASS,
    GRAVITY,
    WEIGHT,
    HOVER_THRUST,
    INERTIA_DIAG,
    ARM_LENGTH,
    MAX_THRUST,
    KQ_OVER_KT,
    SERVO_TIME_CONSTANT,
    SERVO_COMMAND_LIMIT,
)


print()
print("=" * 72)
print("VEHICLE PARAMETER CONSISTENCY")
print("=" * 72)


assert np.isclose(
    WEIGHT,
    MASS * GRAVITY,
)

assert np.isclose(
    HOVER_THRUST,
    MASS * GRAVITY / 4.0,
)

assert INERTIA_DIAG.shape == (3,)

assert np.all(
    INERTIA_DIAG > 0.0
)

assert ARM_LENGTH > 0.0
assert MAX_THRUST > 0.0
assert KQ_OVER_KT > 0.0
assert SERVO_TIME_CONSTANT > 0.0
assert SERVO_COMMAND_LIMIT > 0.0


print(f"Mass               : {MASS:.6f} kg")
print(f"Weight             : {WEIGHT:.6f} N")
print(f"Hover thrust/rotor : {HOVER_THRUST:.9f} N")
print(f"Inertia diagonal   : {INERTIA_DIAG}")
print(f"Arm length         : {ARM_LENGTH:.6f} m")
print(f"Max thrust         : {MAX_THRUST:.6f} N")
print(f"kq/kt              : {KQ_OVER_KT:.6f} m")
print(
    f"Servo tau          : "
    f"{1e3 * SERVO_TIME_CONSTANT:.3f} ms"
)

print()
print("[PASS] Central vehicle parameters are internally consistent.")
