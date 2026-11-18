from __future__ import annotations

import math

import numpy as np

from tiltable_quadrotor_rl.control.dynamic_allocation import (
    predict_tilt_angles,
)

from tiltable_quadrotor_rl.control.dynamic_allocation import (
    allocate_dynamic_one_step,
    predict_tilt_angles,
)


MASS = 3.039
G = 9.81

hover_thrust = MASS * G / 4.0

desired_wrench = np.array([
    20.0,
    0.0,
    MASS * G,
    0.0,
    0.0,
    0.0,
])

thrust, alpha_cmd, info = (
    allocate_dynamic_one_step(
        desired_wrench=desired_wrench,
        current_alpha=np.zeros(4),
        previous_thrust=np.full(
            4,
            hover_thrust,
        ),
        previous_alpha_cmd=np.zeros(4),
    )
)

assert info["success"]

assert thrust.shape == (4,)
assert alpha_cmd.shape == (4,)

assert np.all(
    thrust >= 0.0
)

assert np.all(
    thrust <= 23.0 + 1e-10
)

assert np.all(
    np.abs(alpha_cmd)
    <= 3.15 + 1e-10
)

assert np.all(
    np.isfinite(
        info["predicted_wrench"]
    )
)

print(
    "[PASS] Dynamic allocator produces "
    "valid bounded actuator commands."
)

DT = 0.01
TAU = 0.048

alpha = np.zeros(4)

alpha_cmd = np.ones(4)

predicted = predict_tilt_angles(
    alpha,
    alpha_cmd,
    dt=DT,
    time_constant=TAU,
)

expected = (
    1.0
    - math.exp(
        -DT / TAU
    )
)

assert np.allclose(
    predicted,
    expected,
    atol=1e-12,
)

print()
print(
    "[PASS] One-step dynamic allocator "
    "servo prediction is exact."
)