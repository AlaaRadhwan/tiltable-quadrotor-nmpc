from __future__ import annotations

import math

import numpy as np


# ============================================================
# Rigid-body parameters
# ============================================================

MASS = 3.039
GRAVITY = 9.81

INERTIA_DIAG = np.array(
    [
        0.063,
        0.062,
        0.095,
    ],
    dtype=float,
)

ARM_LENGTH = 0.275


# ============================================================
# Rotor / propulsion parameters
# ============================================================

MAX_THRUST = 23.0

KQ_OVER_KT = 0.0165

ROTOR_RADIUS = 0.114


# ============================================================
# Tilt-servo parameters
# ============================================================

# Current nominal value from the reference platform.
#
# This is deliberately configurable and must NOT be treated as
# a universal servo constant. Replace it with identified hardware
# data when available.
SERVO_TIME_CONSTANT = 0.048

SERVO_COMMAND_LIMIT = 3.15

SERVO_MECHANICAL_LIMIT = math.radians(
    225.0
)


# ============================================================
# Fixed end-effector transform
# ============================================================
#
# B = vehicle body / CoG frame.
# E = end-effector frame.
#
# Current design:
#   - tool origin is 0.15 m above the CoG along +Z_B
#   - tool axes are aligned with body axes
#
# These are the authoritative controller-side values. The MJCF
# contains the same fixed transform and tests verify that the two
# definitions remain consistent.

END_EFFECTOR_POSITION_B = np.array(
    [
        0.0,
        0.0,
        0.15,
    ],
    dtype=float,
)

END_EFFECTOR_QUATERNION_BE = np.array(
    [
        1.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=float,
)


# ============================================================
# Simulation / controller timing
# ============================================================

PHYSICS_DT = 0.002       # 500 Hz
CONTROL_DT = 0.010       # 100 Hz


# ============================================================
# Derived quantities
# ============================================================

WEIGHT = MASS * GRAVITY

HOVER_THRUST = WEIGHT / 4.0
