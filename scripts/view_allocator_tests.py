from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
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

DT = 0.002

# How long to hold the drone frozen so we can inspect
# the rotor configuration before releasing it.
PREVIEW_TIME = 1.5

# How long to let the actual dynamics run.
DYNAMIC_TIME = 0.8


# ============================================================
# Helpers
# ============================================================

def reset_vehicle(model, data):
    mujoco.mj_resetData(
        model,
        data,
    )

    # Position
    data.qpos[0:3] = [
        0.0,
        0.0,
        1.0,
    ]

    # Quaternion: identity orientation
    data.qpos[3:7] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


def print_commands(
    name,
    wrench,
    thrust,
    alpha_cmd,
    info,
):
    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print()
    print("Desired wrench:")
    print(wrench)

    print()
    print("Rotor commands:")

    for i in range(4):
        print(
            f"  Rotor {i + 1}: "
            f"thrust = {thrust[i]:8.4f} N, "
            f"alpha_cmd = {np.rad2deg(alpha_cmd[i]):8.3f} deg"
        )

    print()
    print("Feasible :", info["feasible"])

    if not info["feasible"]:
        print("This case will NOT be simulated.")
        print("It violates actuator limits.")


def preview_configuration(
    model,
    data,
    viewer,
    alpha_cmd,
):
    """
    Smoothly move the visual rotor angles into the allocator's
    requested configuration while holding the vehicle fixed.

    This lets us inspect the geometry before the physics starts.
    """

    alpha = np.zeros(4)

    steps = int(
        PREVIEW_TIME / DT
    )

    for _ in range(steps):

        if not viewer.is_running():
            return False

        alpha = first_order_servo_step(
            alpha,
            alpha_cmd,
            DT,
        )

        write_tilt_angles(
            model,
            data,
            alpha,
        )

        # Freeze the base during preview.
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

        mujoco.mj_forward(
            model,
            data,
        )

        viewer.sync()
        time.sleep(DT)

    return True


def run_dynamic_test(
    model,
    data,
    viewer,
    thrust,
    alpha_cmd,
):
    """
    Run the actual vehicle physics.

    The tilt servos follow the explicit first-order model.
    Thrust commands are applied directly in Newtons.
    """

    alpha = np.zeros(4)

    steps = int(
        DYNAMIC_TIME / DT
    )

    for _ in range(steps):

        if not viewer.is_running():
            return False

        # Explicit servo dynamics.
        alpha = first_order_servo_step(
            alpha,
            alpha_cmd,
            DT,
        )

        write_tilt_angles(
            model,
            data,
            alpha,
        )

        # MuJoCo only receives the four rotor thrusts.
        data.ctrl[:] = thrust

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()
        time.sleep(DT)

    return True


def run_case(
    model,
    data,
    viewer,
    name,
    wrench,
):
    thrust, alpha_cmd, info = allocate_wrench(
        wrench
    )

    print_commands(
        name,
        wrench,
        thrust,
        alpha_cmd,
        info,
    )

    if not info["feasible"]:
        return True

    # -----------------------------------------------
    # Stage A — inspect rotor configuration
    # -----------------------------------------------

    reset_vehicle(
        model,
        data,
    )

    print()
    print("Previewing rotor configuration...")

    if not preview_configuration(
        model,
        data,
        viewer,
        alpha_cmd,
    ):
        return False

    time.sleep(0.5)

    # -----------------------------------------------
    # Stage B — reset and release the vehicle
    # -----------------------------------------------

    reset_vehicle(
        model,
        data,
    )

    print("Running dynamics...")

    if not run_dynamic_test(
        model,
        data,
        viewer,
        thrust,
        alpha_cmd,
    ):
        return False

    time.sleep(1.0)

    return True


# ============================================================
# Test cases
# ============================================================

cases = [

    (
        "TEST 1 — HOVER",
        np.array([
            0.0,
            0.0,
            MASS * G,
            0.0,
            0.0,
            0.0,
        ]),
    ),

    (
        "TEST 2 — +X FORCE",
        np.array([
            5.42544768,
            0.0,
            MASS * G,
            0.0,
            0.0,
            0.0,
        ]),
    ),

    (
        "TEST 3 — +Y FORCE",
        np.array([
            0.0,
            4.0,
            MASS * G,
            0.0,
            0.0,
            0.0,
        ]),
    ),

    (
        "TEST 4 — +ROLL TORQUE",
        np.array([
            0.0,
            0.0,
            MASS * G,
            0.5,
            0.0,
            0.0,
        ]),
    ),

    (
        "TEST 5 — +PITCH TORQUE",
        np.array([
            0.0,
            0.0,
            MASS * G,
            0.0,
            0.5,
            0.0,
        ]),
    ),

    (
        "TEST 6 — +YAW TORQUE",
        np.array([
            0.0,
            0.0,
            MASS * G,
            0.0,
            0.0,
            0.5,
        ]),
    ),

]


# ============================================================
# Main
# ============================================================

model = mujoco.MjModel.from_xml_path(
    str(XML_PATH)
)

data = mujoco.MjData(model)

base_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base",
)


with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    viewer.cam.type = (
        mujoco.mjtCamera.mjCAMERA_TRACKING
    )

    viewer.cam.trackbodyid = base_id

    viewer.cam.distance = 1.6
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25

    # Show rotor-site coordinate frames.
    viewer.opt.frame = (
        mujoco.mjtFrame.mjFRAME_SITE
    )

    print()
    print("=" * 72)
    print("VISUAL ALLOCATOR TESTS")
    print("=" * 72)
    print()
    print(
        "Each case has two stages:"
    )
    print(
        "  1. rotor configuration preview"
    )
    print(
        "  2. actual free-flight dynamics"
    )

    for name, wrench in cases:

        if not viewer.is_running():
            break

        if not run_case(
            model,
            data,
            viewer,
            name,
            wrench,
        ):
            break

    print()
    print("=" * 72)
    print("VISUAL TESTS COMPLETE")
    print("=" * 72)

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)