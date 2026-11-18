from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    allocate_wrench,
    allocate_thrust_at_fixed_angles,
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

model.opt.gravity[:] = 0.0


# ============================================================
# Fixed base helper
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
# Servo state
# ============================================================

alpha = np.zeros(4)

alpha_cmd = np.zeros(4)

desired_wrench = hover.copy()


# ============================================================
# Simulation
# ============================================================

TOTAL_TIME = 2.0

physics_steps = int(
    TOTAL_TIME / PHYSICS_DT
)


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


print()
print("=" * 72)
print("DYNAMIC SERVO-AWARE ALLOCATOR TEST")
print("=" * 72)


for step in range(physics_steps):

    t = step * PHYSICS_DT

    # --------------------------------------------------------
    # Requested wrench
    # --------------------------------------------------------

    if t < 0.5:

        requested = hover

    elif t < 1.5:

        requested = forward

    else:

        requested = hover

    # --------------------------------------------------------
    # High-level allocator:
    #
    # Runs at 100 Hz.
    #
    # Its main job here is to determine the desired
    # rotor tilt configuration.
    # --------------------------------------------------------

    if step % STEPS_PER_CONTROL == 0:

        (
            _,
            alpha_cmd,
            allocation_info,
        ) = allocate_wrench(
            requested,
            hover,
        )

        desired_wrench = (
            allocation_info["achieved_wrench"]
        )

    # --------------------------------------------------------
    # Actual servo dynamics:
    #
    # 500 Hz
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

    # --------------------------------------------------------
    # NEW:
    #
    # Given the CURRENT ACTUAL servo angles, find the
    # best physically valid thrusts for the desired wrench.
    # --------------------------------------------------------

    (
        thrust_cmd,
        predicted_wrench,
        thrust_info,
    ) = allocate_thrust_at_fixed_angles(
        alpha,
        desired_wrench,
    )

    data.ctrl[:] = thrust_cmd

    # --------------------------------------------------------
    # Evaluate MuJoCo actuator wrench
    # --------------------------------------------------------

    reset_base()

    mujoco.mj_forward(
        model,
        data,
    )

    actual_wrench = (
        data.qfrc_actuator[:6].copy()
    )

    # --------------------------------------------------------
    # Selected output samples
    # --------------------------------------------------------

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
            "thrust cmd [N]  :",
            thrust_cmd,
        )

        print(
            "desired wrench  :",
            desired_wrench,
        )

        print(
            "predicted wrench:",
            predicted_wrench,
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

        print(
            "weighted error  :",
            thrust_info[
                "weighted_error_norm"
            ],
        )


print()
print("=" * 72)
print("TEST COMPLETE")
print("=" * 72)