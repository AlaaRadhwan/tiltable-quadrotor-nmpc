from __future__ import annotations

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


XML_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tiltable_quadrotor_rl"
    / "robot"
    / "mjcf"
    / "tiltable_quadrotor.xml"
)

MASS = 3.039
G = 9.81

HOVER_THRUST = MASS * G / 4.0


def joint_qpos_address(model, joint_name):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if jid < 0:
        raise RuntimeError(f"Joint not found: {joint_name}")

    return model.jnt_qposadr[jid]


def set_tilt_positions(model, data, angles):
    for i, angle in enumerate(angles, start=1):
        adr = joint_qpos_address(
            model,
            f"tilt_{i}",
        )
        data.qpos[adr] = angle


def visual_tilt_motion(
    model,
    data,
    viewer,
    start_deg,
    end_deg,
    duration=2.0,
):
    """
    Kinematic demonstration only.

    We directly move the four tilt joints so that we can clearly
    inspect their geometry without the vehicle flying away.
    """

    start = math.radians(start_deg)
    end = math.radians(end_deg)

    fps = 60
    steps = int(duration * fps)

    for k in range(steps + 1):

        if not viewer.is_running():
            return False

        s = k / steps

        # Smooth interpolation.
        s = 0.5 - 0.5 * math.cos(math.pi * s)

        alpha = start + s * (end - start)

        set_tilt_positions(
            model,
            data,
            [alpha, alpha, alpha, alpha],
        )

        # Keep base fixed during this visual geometry inspection.
        data.qpos[0:3] = [0.0, 0.0, 1.0]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        data.qvel[:] = 0.0

        mujoco.mj_forward(model, data)

        viewer.sync()
        time.sleep(1.0 / fps)

    return True


def run_physics_for(model, data, viewer, seconds):
    steps = int(seconds / model.opt.timestep)

    for _ in range(steps):

        if not viewer.is_running():
            return False

        mujoco.mj_step(model, data)

        viewer.sync()

        # Approximately real-time.
        time.sleep(model.opt.timestep)

    return True


# ============================================================
# Load model
# ============================================================

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

base_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base",
)


with mujoco.viewer.launch_passive(model, data) as viewer:

    # Track the vehicle with the camera.
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = base_id
    viewer.cam.distance = 1.6
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25

    # Show site coordinate frames.
    # The blue/local-Z axis of each rotor site is its thrust axis.
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    print()
    print("=" * 60)
    print("PHASE 1 — ROTOR TILT GEOMETRY")
    print("=" * 60)
    print()
    print("The drone is intentionally frozen.")
    print("Watch the four rotor discs and their site coordinate frames.")
    print()
    print("We will move:")
    print("    0 deg -> 30 deg -> 90 deg -> 0 deg")
    print()

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    if not visual_tilt_motion(
        model, data, viewer,
        0, 30, 2.0
    ):
        raise SystemExit

    time.sleep(0.5)

    if not visual_tilt_motion(
        model, data, viewer,
        30, 90, 2.0
    ):
        raise SystemExit

    time.sleep(1.0)

    if not visual_tilt_motion(
        model, data, viewer,
        90, 0, 2.0
    ):
        raise SystemExit

    time.sleep(1.0)


    # ========================================================
    # Actual dynamic hover
    # ========================================================

    print()
    print("=" * 60)
    print("PHASE 2 — REAL FREE-FLIGHT HOVER")
    print("=" * 60)
    print()
    print("Now the base is free.")
    print(
        f"Each rotor receives {HOVER_THRUST:.4f} N."
    )
    print("No position controller is running.")
    print()

    mujoco.mj_resetData(model, data)

    data.ctrl[:] = 0.0

    data.ctrl[0:4] = HOVER_THRUST
    data.ctrl[4:8] = 0.0

    mujoco.mj_forward(model, data)

    if not run_physics_for(
        model, data, viewer, 2.0
    ):
        raise SystemExit


    # ========================================================
    # Omnidirectional translation demonstration
    # ========================================================

    print()
    print("=" * 60)
    print("PHASE 3 — PURE +X FORCE WITH TORQUE COMPENSATION")
    print("=" * 60)
    print()
    print("We will generate +X force while compensating")
    print("the propeller reaction torque.")
    print()

    mujoco.mj_resetData(model, data)

    data.ctrl[:] = 0.0

    # --------------------------------------------------------
    # Desired horizontal component
    # --------------------------------------------------------

    nominal_alpha = math.radians(20.0)

    H = HOVER_THRUST * math.tan(nominal_alpha)

    # Desired net X force:
    #
    # rotor 2 contributes +H
    # rotor 4 contributes +H
    #
    # therefore Fx = 2H

    desired_fx = 2.0 * H

    # --------------------------------------------------------
    # Compensate roll reaction torque
    # --------------------------------------------------------

    arm_length = 0.275
    kq_over_kt = 0.0165

    delta_v = (
        kq_over_kt
        * H
        / arm_length
    )

    v1 = HOVER_THRUST
    v3 = HOVER_THRUST

    v2 = HOVER_THRUST - delta_v
    v4 = HOVER_THRUST + delta_v

    # --------------------------------------------------------
    # Convert horizontal/vertical components to:
    #
    #     thrust magnitude
    #     servo angle
    #
    # f = sqrt(h^2 + v^2)
    # alpha = atan2(h, v)
    # --------------------------------------------------------

    f1 = v1
    a1 = 0.0

    f2 = math.sqrt(
        H * H + v2 * v2
    )
    a2 = math.atan2(
        H,
        v2,
    )

    f3 = v3
    a3 = 0.0

    # Rotor 4 needs negative local horizontal component
    # to produce +X in the body/world frame.
    f4 = math.sqrt(
        H * H + v4 * v4
    )
    a4 = math.atan2(
        -H,
        v4,
    )

    print(f"H               = {H:.4f} N")
    print(f"Desired Fx      = {desired_fx:.4f} N")
    print(f"delta_v         = {delta_v:.4f} N")
    print()

    print(
        f"Rotor 1: f={f1:.4f} N, "
        f"alpha={math.degrees(a1):.3f} deg"
    )

    print(
        f"Rotor 2: f={f2:.4f} N, "
        f"alpha={math.degrees(a2):.3f} deg"
    )

    print(
        f"Rotor 3: f={f3:.4f} N, "
        f"alpha={math.degrees(a3):.3f} deg"
    )

    print(
        f"Rotor 4: f={f4:.4f} N, "
        f"alpha={math.degrees(a4):.3f} deg"
    )

    # --------------------------------------------------------
    # Start directly from the desired tilt configuration.
    #
    # This isolates the steady-state wrench geometry from
    # servo-transition dynamics for this demonstration.
    # --------------------------------------------------------

    set_tilt_positions(
        model,
        data,
        [a1, a2, a3, a4],
    )

    # Servo activation states correspond to the four
    # filtered position actuators.
    if model.na == 4:
        data.act[:] = [
            a1,
            a2,
            a3,
            a4,
        ]

    data.ctrl[0:4] = [
        f1,
        f2,
        f3,
        f4,
    ]

    data.ctrl[4:8] = [
        a1,
        a2,
        a3,
        a4,
    ]

    mujoco.mj_forward(model, data)

    print()
    print("Initial generalized actuator wrench:")
    print(data.qfrc_actuator[:6])

    print()
    print(
        "Expected approximately:"
    )
    print(
        f"[{desired_fx:.4f}, 0, "
        f"{MASS * G:.4f}, 0, 0, 0]"
    )
    print()

    if not run_physics_for(
        model,
        data,
        viewer,
        5.0,
    ):
        raise SystemExit


    print()
    print("=" * 60)
    print("VISUAL DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("Close the MuJoCo viewer to exit.")

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)