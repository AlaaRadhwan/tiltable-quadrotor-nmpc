import numpy as np

from tiltable_quadrotor_rl.control.nmpc_model import (
    MASS,
    GRAVITY,
    NX,
    NU,
    build_dynamics_function,
    build_wrench_function,
)

from tiltable_quadrotor_rl.control.allocation import (
    rotor_commands_to_virtual,
    virtual_to_wrench,
)

from tiltable_quadrotor_rl.robot.tilt_servo import (
    DEFAULT_TIME_CONSTANT,
)


print()
print("=" * 72)
print("NMPC CONTINUOUS MODEL VALIDATION")
print("=" * 72)


dynamics = build_dynamics_function()
wrench_function = build_wrench_function()


# ------------------------------------------------------------------
# TEST 1 — Dimensions
# ------------------------------------------------------------------

assert NX == 17
assert NU == 8

print("[PASS] State dimension = 17, input dimension = 8.")


# ------------------------------------------------------------------
# TEST 2 — Static hover must be an equilibrium
# ------------------------------------------------------------------

x = np.zeros(NX)

# Position is arbitrary for the dynamics.
x[2] = 1.0

# Unit quaternion: no body rotation.
x[6] = 1.0

u = np.zeros(NU)

hover_thrust = (
    MASS * GRAVITY / 4.0
)

u[0:4] = hover_thrust

x_dot = np.asarray(
    dynamics(x, u)
).reshape(-1)

max_hover_derivative = np.max(
    np.abs(x_dot)
)

assert max_hover_derivative < 1e-10

print(
    "[PASS] Level equal-thrust hover is a continuous-time equilibrium."
)


# ------------------------------------------------------------------
# TEST 3 — Servo dynamics
# ------------------------------------------------------------------

x = np.zeros(NX)
x[6] = 1.0

alpha = np.array(
    [0.1, -0.2, 0.3, -0.4]
)

alpha_cmd = np.array(
    [0.4, 0.1, -0.1, -0.2]
)

x[13:17] = alpha

u = np.zeros(NU)
u[4:8] = alpha_cmd

x_dot = np.asarray(
    dynamics(x, u)
).reshape(-1)

expected_alpha_dot = (
    alpha_cmd - alpha
) / DEFAULT_TIME_CONSTANT

assert np.allclose(
    x_dot[13:17],
    expected_alpha_dot,
    atol=1e-12,
)

print("[PASS] Servo derivative matches the first-order model.")


# ------------------------------------------------------------------
# TEST 4 — Quaternion kinematics sanity check
# ------------------------------------------------------------------

x = np.zeros(NX)
x[6] = 1.0

# 1 rad/s yaw rate
x[12] = 1.0

u = np.zeros(NU)

x_dot = np.asarray(
    dynamics(x, u)
).reshape(-1)

expected_q_dot = np.array(
    [0.0, 0.0, 0.0, 0.5]
)

assert np.allclose(
    x_dot[6:10],
    expected_q_dot,
    atol=1e-12,
)

print("[PASS] Quaternion derivative has the expected yaw-rate behavior.")


# ------------------------------------------------------------------
# TEST 5 — CasADi wrench model must exactly reproduce
#          our already validated allocation model.
# ------------------------------------------------------------------

rng = np.random.default_rng(7)

max_error = 0.0

for _ in range(100):

    thrust = rng.uniform(
        0.0,
        23.0,
        size=4,
    )

    alpha = rng.uniform(
        -3.0,
        3.0,
        size=4,
    )

    z = rotor_commands_to_virtual(
        thrust,
        alpha,
    )

    expected = virtual_to_wrench(z)

    actual = np.asarray(
        wrench_function(
            thrust,
            alpha,
        )
    ).reshape(-1)

    error = np.max(
        np.abs(actual - expected)
    )

    max_error = max(
        max_error,
        error,
    )


assert max_error < 1e-12

print(
    f"[PASS] CasADi wrench model matches allocation model "
    f"(max error {max_error:.3e})."
)


print()
print("=" * 72)
print("ALL NMPC CONTINUOUS MODEL TESTS PASSED")
print("=" * 72)
