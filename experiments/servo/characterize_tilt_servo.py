from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

XML_PATH = (
    PROJECT_ROOT
    / "src"
    / "tiltable_quadrotor_rl"
    / "robot"
    / "mjcf"
    / "tiltable_quadrotor.xml"
)


SERVO_TAU = 0.048
COMMAND_STEP = 1.0

SIM_DT = 0.002
DURATION = 0.30

KP_VALUES = [
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
]


def build_test_model(kp: float):
    """
    Load the production MJCF and add four filtered
    position servos IN MEMORY.

    Production XML is not modified.
    """

    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    actuator = root.find("actuator")

    if actuator is None:
        raise RuntimeError("No <actuator> block found.")

    for i in range(1, 5):
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"tilt_servo_{i}",
                "joint": f"tilt_{i}",
                "kp": str(kp),
                "dampratio": "1",
                "timeconst": str(SERVO_TAU),
                "ctrllimited": "true",
                "ctrlrange": "-3.15 3.15",
            },
        )

    # Existing keyframes only contain the four thrust controls.
    # Extend them with four zero servo commands.
    keyframe = root.find("keyframe")

    if keyframe is not None:
        for key in keyframe.findall("key"):
            ctrl_text = key.get("ctrl")

            if ctrl_text is not None:
                ctrl = [
                    float(v)
                    for v in ctrl_text.split()
                ]

                if len(ctrl) == 4:
                    ctrl += [0.0] * 4

                key.set(
                    "ctrl",
                    " ".join(str(v) for v in ctrl),
                )

    xml_string = ET.tostring(
        root,
        encoding="unicode",
    )

    model = mujoco.MjModel.from_xml_string(
        xml_string
    )

    return model


def joint_qpos_address(model, name):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    return model.jnt_qposadr[jid]


def first_crossing_time(
    time,
    signal,
    level,
):
    indices = np.flatnonzero(
        signal >= level
    )

    if len(indices) == 0:
        return math.nan

    return float(
        time[indices[0]]
    )


def run_case(kp: float):
    model = build_test_model(kp)
    data = mujoco.MjData(model)

    # Isolate the servo mechanics.
    # No gravity and no rotor thrust.
    model.opt.gravity[:] = 0.0

    mujoco.mj_resetData(
        model,
        data,
    )

    # Start with a level body.
    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    # Four thrust controls.
    data.ctrl[0:4] = 0.0

    # Simultaneous equal servo command.
    #
    # The four servo axes are arranged symmetrically, so
    # their reaction torques should largely cancel at the body.
    data.ctrl[4:8] = COMMAND_STEP

    mujoco.mj_forward(
        model,
        data,
    )

    qpos_addr = [
        joint_qpos_address(
            model,
            f"tilt_{i}",
        )
        for i in range(1, 5)
    ]

    time_history = []
    alpha_history = []
    activation_history = []
    force_history = []
    base_omega_history = []

    while data.time < DURATION:

        mujoco.mj_step(
            model,
            data,
        )

        alpha = np.array(
            [
                data.qpos[address]
                for address in qpos_addr
            ]
        )

        time_history.append(
            data.time
        )

        alpha_history.append(
            alpha
        )

        # Four filtered servo activation states.
        activation_history.append(
            data.act.copy()
        )

        # Actuator order:
        #   0:4 -> thrust
        #   4:8 -> tilt servos
        force_history.append(
            data.actuator_force[4:8].copy()
        )

        base_omega_history.append(
            data.qvel[3:6].copy()
        )

    time_history = np.asarray(
        time_history
    )

    alpha_history = np.asarray(
        alpha_history
    )

    activation_history = np.asarray(
        activation_history
    )

    force_history = np.asarray(
        force_history
    )

    base_omega_history = np.asarray(
        base_omega_history
    )

    # Average all four servos.
    alpha_mean = np.mean(
        alpha_history,
        axis=1,
    )

    activation_mean = np.mean(
        activation_history,
        axis=1,
    )

    # Desired paper-style first-order response.
    target = (
        COMMAND_STEP
        * (
            1.0
            - np.exp(
                -time_history / SERVO_TAU
            )
        )
    )

    error = (
        alpha_mean - target
    )

    rms_error = float(
        np.sqrt(
            np.mean(error**2)
        )
    )

    max_error = float(
        np.max(
            np.abs(error)
        )
    )

    t63 = first_crossing_time(
        time_history,
        alpha_mean,
        0.6321205588 * COMMAND_STEP,
    )

    t90 = first_crossing_time(
        time_history,
        alpha_mean,
        0.9 * COMMAND_STEP,
    )

    peak_torque = float(
        np.max(
            np.abs(force_history)
        )
    )

    peak_base_omega = float(
        np.max(
            np.linalg.norm(
                base_omega_history,
                axis=1,
            )
        )
    )

    final_alpha = float(
        alpha_mean[-1]
    )

    final_activation = float(
        activation_mean[-1]
    )

    return {
        "kp": kp,
        "rms_error": rms_error,
        "max_error": max_error,
        "t63": t63,
        "t90": t90,
        "peak_torque": peak_torque,
        "peak_base_omega": peak_base_omega,
        "final_alpha": final_alpha,
        "final_activation": final_activation,
        "nu": model.nu,
        "na": model.na,
    }


def main():
    print()
    print("=" * 110)
    print("MUJOCO PHYSICAL TILT-SERVO CHARACTERIZATION")
    print("=" * 110)

    print()
    print(f"Target servo time constant : {SERVO_TAU:.3f} s")
    print(f"Command step              : {COMMAND_STEP:.3f} rad")
    print(f"Simulation timestep        : {SIM_DT:.3f} s")

    print()
    print(
        f"{'kp':>8s}"
        f"{'RMS err':>12s}"
        f"{'max err':>12s}"
        f"{'t63 [ms]':>12s}"
        f"{'t90 [ms]':>12s}"
        f"{'peak tau':>12s}"
        f"{'base |w|':>12s}"
        f"{'alpha end':>12s}"
    )

    print("-" * 110)

    for kp in KP_VALUES:

        result = run_case(
            kp
        )

        print(
            f"{result['kp']:8.3f}"
            f"{result['rms_error']:12.5f}"
            f"{result['max_error']:12.5f}"
            f"{1e3 * result['t63']:12.2f}"
            f"{1e3 * result['t90']:12.2f}"
            f"{result['peak_torque']:12.5f}"
            f"{result['peak_base_omega']:12.5e}"
            f"{result['final_alpha']:12.5f}"
        )

    # Show compiled dimensions for one representative case.
    model = build_test_model(
        KP_VALUES[0]
    )

    print()
    print(
        f"Experimental model dimensions: "
        f"nq={model.nq}, nv={model.nv}, "
        f"nu={model.nu}, na={model.na}"
    )

    print()
    print(
        "Expected:"
        "\n  nq = 11"
        "\n  nv = 10"
        "\n  nu = 8  (4 thrust + 4 servo commands)"
        "\n  na = 4  (4 filtered servo activation states)"
    )


if __name__ == "__main__":
    main()
