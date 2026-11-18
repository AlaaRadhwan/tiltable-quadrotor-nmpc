from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    allocate_wrench,
)
from tiltable_quadrotor_rl.robot.tilt_servo import (
    first_order_servo_step,
    write_tilt_angles,
)


ROOT = Path(__file__).resolve().parents[1]

XML_PATH = (
    ROOT
    / "src"
    / "tiltable_quadrotor_rl"
    / "robot"
    / "mjcf"
    / "tiltable_quadrotor.xml"
)


MASS = 3.039
G = 9.81

PHYSICS_DT = 0.002

# Allocator/controller update:
# 100 Hz
CONTROL_DT = 0.01

STEPS_PER_CONTROL = int(
    CONTROL_DT / PHYSICS_DT
)


hover = np.array([
    0.0,
    0.0,
    MASS * G,
    0.0,
    0.0,
    0.0,
])


forward = np.array([
    20.0,
    0.0,
    MASS * G,
    0.0,
    0.0,
    0.0,
])


# ============================================================
# Model
# ============================================================

model = mujoco.MjModel.from_xml_path(
    str(XML_PATH)
)

data = mujoco.MjData(model)

# We want qfrc_actuator only.
# Gravity is irrelevant to the actuator-wrench measurement.
model.opt.gravity[:] = 0.0


# ============================================================
# Fixed base state
# ============================================================

def reset_base():
    data.qpos[0:3] = [
        0.0,
        0.0,
        1.0,
    ]

    data.qpos[3:7] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]

    data.qvel[:6] = 0.0


# ============================================================
# Current physical servo state
# ============================================================

alpha = np.zeros(4)


# ============================================================
# Simulation
#
# 0.0–0.5 s : hover
# 0.5–1.5 s : +20 N X
# 1.5–2.0 s : hover
# ============================================================

TOTAL_TIME = 2.0

physics_steps = int(
    TOTAL_TIME / PHYSICS_DT
)


thrust_cmd = np.zeros(4)
alpha_cmd = np.zeros(4)

desired_wrench = hover.copy()


print()
print("=" * 72)
print("DYNAMIC ALLOCATOR / SERVO TEST")
print("=" * 72)


for step in range(physics_steps):

    t = step * PHYSICS_DT

    # --------------------------------------------------------
    # Select requested wrench
    # --------------------------------------------------------

    if t < 0.5:
        requested = hover

    elif t < 1.5:
        requested = forward

    else:
        requested = hover

    # --------------------------------------------------------
    # Allocator runs at 100 Hz
    # --------------------------------------------------------

    if step % STEPS_PER_CONTROL == 0:

        (
            thrust_cmd,
            alpha_cmd,
            info,
        ) = allocate_wrench(
            requested,
            hover,
        )

        desired_wrench = (
            info["achieved_wrench"]
        )

    # --------------------------------------------------------
    # Servo updates at 500 Hz
    # --------------------------------------------------------

    alpha = first_order_servo_step(
        alpha,
        alpha_cmd,
        PHYSICS_DT,
    )

    write_tilt_angles(
        model,
        data,
        alpha,
    )

    data.ctrl[:] = thrust_cmd

    reset_base()

    mujoco.mj_forward(
        model,
        data,
    )

    actual_wrench = (
        data.qfrc_actuator[:6].copy()
    )

    # --------------------------------------------------------
    # Print selected timestamps
    # --------------------------------------------------------

    sample_times = [
        0.0,
        0.5,
        0.51,
        0.52,
        0.55,
        0.60,
        0.70,
        1.0,
        1.5,
        1.51,
        1.55,
        1.60,
        1.80,
    ]

    if any(
        abs(t - ts) < PHYSICS_DT / 2
        for ts in sample_times
    ):
        print()
        print(
            f"t = {t:.3f} s"
        )

        print(
            "alpha cmd [deg] :",
            np.rad2deg(alpha_cmd),
        )

        print(
            "alpha     [deg] :",
            np.rad2deg(alpha),
        )

        print(
            "desired wrench  :",
            desired_wrench,
        )

        print(
            "actual wrench   :",
            actual_wrench,
        )

        print(
            "wrench error    :",
            actual_wrench
            - desired_wrench,
        )


print()
print("=" * 72)
print("TEST COMPLETE")
print("=" * 72)