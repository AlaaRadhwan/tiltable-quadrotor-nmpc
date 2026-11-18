from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np


# ============================================================
# Expected physical parameters
# ============================================================

XML_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tiltable_quadrotor_rl"
    / "robot"
    / "mjcf"
    / "tiltable_quadrotor.xml"
)

MASS_EXPECTED = 3.039

INERTIA_EXPECTED = np.diag(
    [
        0.063,
        0.062,
        0.095,
    ]
)

ARM_LENGTH = 0.275
KQ_OVER_KT = 0.0165

MAX_THRUST = 23.0
HOVER_THRUST = MASS_EXPECTED * 9.81 / 4.0

SERVO_MECHANICAL_LIMIT = math.radians(225.0)
SERVO_COMMAND_LIMIT = 3.15


# ============================================================
# Helpers
# ============================================================

def obj_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"Could not find MuJoCo object: {name}")
    return idx


def actuator_names(model):
    return [
        mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            i,
        )
        for i in range(model.nu)
    ]


def set_joint_qpos(model, data, joint_name, value):
    jid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )
    qadr = model.jnt_qposadr[jid]
    data.qpos[qadr] = value


def site_z_axis(model, data, site_name):
    sid = obj_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    R = data.site_xmat[sid].reshape(3, 3)

    # Third column is the site's local +Z axis expressed in world.
    return R[:, 2].copy()


def print_vector(name, x):
    print(f"{name}: {np.array2string(np.asarray(x), precision=8)}")


# ============================================================
# Load model
# ============================================================

print()
print("=" * 72)
print("TILTABLE QUADROTOR PLANT VALIDATION")
print("=" * 72)

print(f"\nLoading:\n{XML_PATH}\n")

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

print("[PASS] XML compiled successfully.")


# ============================================================
# TEST 1 — Compiled structure
# ============================================================

print()
print("-" * 72)
print("TEST 1 — MODEL STRUCTURE")
print("-" * 72)

print(f"nq    = {model.nq}")
print(f"nv    = {model.nv}")
print(f"nu    = {model.nu}")
print(f"na    = {model.na}")
print(f"njnt  = {model.njnt}")
print(f"nbody = {model.nbody}")

assert model.nq == 11, f"Expected nq=11, found {model.nq}"
assert model.nv == 10, f"Expected nv=10, found {model.nv}"
assert model.nu == 4, f"Expected nu=8, found {model.nu}"
assert model.njnt == 5, f"Expected njnt=5, found {model.njnt}"

expected_actuator_names = [
    "thrust_1",
    "thrust_2",
    "thrust_3",
    "thrust_4",
]

actual_actuator_names = actuator_names(model)

print("\nActuator order:")
for i, name in enumerate(actual_actuator_names):
    print(f"  ctrl[{i}] -> {name}")

assert actual_actuator_names == expected_actuator_names

print("\n[PASS] Model dimensions and actuator ordering are correct.")


# ============================================================
# TEST 2 — Total mass and assembled inertia
# ============================================================

print()
print("-" * 72)
print("TEST 2 — TOTAL MASS AND INERTIA")
print("-" * 72)

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

total_mass = mujoco.mj_getTotalmass(model)

print(f"Expected total mass : {MASS_EXPECTED:.9f} kg")
print(f"Compiled total mass : {total_mass:.9f} kg")

assert np.isclose(
    total_mass,
    MASS_EXPECTED,
    atol=1e-10,
)

# Compute total COM from all compiled body inertias.
masses = model.body_mass[1:].copy()
positions = data.xipos[1:].copy()

total_com = np.sum(
    masses[:, None] * positions,
    axis=0,
) / total_mass

# Assemble inertia about the total COM.
I_total = np.zeros((3, 3))

for body_id in range(1, model.nbody):

    m = model.body_mass[body_id]

    p = data.xipos[body_id]

    R = data.ximat[body_id].reshape(3, 3)

    I_local = np.diag(
        model.body_inertia[body_id]
    )

    I_world = R @ I_local @ R.T

    r = p - total_com

    I_parallel = m * (
        np.dot(r, r) * np.eye(3)
        - np.outer(r, r)
    )

    I_total += I_world + I_parallel

print()
print_vector("Total COM", total_com)

print("\nExpected inertia:")
print(INERTIA_EXPECTED)

print("\nCompiled assembled inertia:")
print(I_total)

print("\nDifference:")
print(I_total - INERTIA_EXPECTED)

assert np.allclose(
    I_total,
    INERTIA_EXPECTED,
    atol=1e-8,
)

print("\n[PASS] Assembled mass and inertia match the target values.")


# ============================================================
# TEST 3 — Tilt geometry / thrust directions
# ============================================================

print()
print("-" * 72)
print("TEST 3 — ROTOR FRAME AND TILT GEOMETRY")
print("-" * 72)

test_angles_deg = [
    0.0,
    30.0,
    90.0,
]

for angle_deg in test_angles_deg:

    angle = math.radians(angle_deg)

    mujoco.mj_resetData(model, data)

    if model.na:
        data.act[:] = 0.0

    data.ctrl[:] = 0.0

    for i in range(1, 5):
        set_joint_qpos(
            model,
            data,
            f"tilt_{i}",
            angle,
        )

    mujoco.mj_forward(model, data)

    print(f"\nalpha = {angle_deg:.1f} deg")

    for i in range(1, 5):
        z = site_z_axis(
            model,
            data,
            f"rotor_{i}_thrust",
        )

        print_vector(f"  rotor {i} +Z", z)

    if angle_deg == 0.0:

        expected = np.array(
            [0.0, 0.0, 1.0]
        )

        for i in range(1, 5):
            z = site_z_axis(
                model,
                data,
                f"rotor_{i}_thrust",
            )

            assert np.allclose(
                z,
                expected,
                atol=1e-8,
            )

    if angle_deg == 90.0:

        expected_dirs = {
            1: np.array([0.0, -1.0, 0.0]),
            2: np.array([1.0, 0.0, 0.0]),
            3: np.array([0.0, 1.0, 0.0]),
            4: np.array([-1.0, 0.0, 0.0]),
        }

        for i, expected in expected_dirs.items():

            z = site_z_axis(
                model,
                data,
                f"rotor_{i}_thrust",
            )

            assert np.allclose(
                z,
                expected,
                atol=1e-7,
            )

print("\n[PASS] Rotor tilt frames produce the expected thrust directions.")


# ============================================================
# TEST 4 — Individual rotor wrench
# ============================================================

print()
print("-" * 72)
print("TEST 4 — INDIVIDUAL ROTOR WRENCH")
print("-" * 72)

# Temporarily remove gravity so we inspect actuator wrench only.
gravity_original = model.opt.gravity.copy()
model.opt.gravity[:] = 0.0

expected_wrenches = {

    1: np.array(
        [
            0.0,
            0.0,
            1.0,
            0.0,
            -ARM_LENGTH,
            -KQ_OVER_KT,
        ]
    ),

    2: np.array(
        [
            0.0,
            0.0,
            1.0,
            ARM_LENGTH,
            0.0,
            KQ_OVER_KT,
        ]
    ),

    3: np.array(
        [
            0.0,
            0.0,
            1.0,
            0.0,
            ARM_LENGTH,
            -KQ_OVER_KT,
        ]
    ),

    4: np.array(
        [
            0.0,
            0.0,
            1.0,
            -ARM_LENGTH,
            0.0,
            KQ_OVER_KT,
        ]
    ),
}

for rotor in range(1, 5):

    mujoco.mj_resetData(model, data)

    if model.na:
        data.act[:] = 0.0

    data.ctrl[:] = 0.0

    # Apply exactly 1 N thrust to one rotor.
    data.ctrl[rotor - 1] = 1.0

    mujoco.mj_forward(model, data)

    wrench = data.qfrc_actuator[:6].copy()

    expected = expected_wrenches[rotor]

    print(f"\nRotor {rotor}, f = 1 N")
    print_vector("  measured", wrench)
    print_vector("  expected", expected)
    print_vector("  error   ", wrench - expected)

    assert np.allclose(
        wrench,
        expected,
        atol=1e-7,
    )

model.opt.gravity[:] = gravity_original

print("\n[PASS] Individual rotor forces, moment arms and reaction torques are correct.")


# ============================================================
# TEST 5 — Hover equilibrium
# ============================================================

print()
print("-" * 72)
print("TEST 5 — STATIC HOVER EQUILIBRIUM")
print("-" * 72)

mujoco.mj_resetData(model, data)

if model.na:
    data.act[:] = 0.0

data.ctrl[:] = 0.0

data.ctrl[0:4] = HOVER_THRUST

mujoco.mj_forward(model, data)

base_qacc = data.qacc[:6].copy()
actuator_wrench = data.qfrc_actuator[:6].copy()

print(f"Hover thrust per rotor = {HOVER_THRUST:.9f} N")

print()
print_vector("Actuator generalized wrench", actuator_wrench)
print_vector("Base acceleration", base_qacc)

print(
    f"\nTotal upward thrust = "
    f"{4 * HOVER_THRUST:.9f} N"
)

print(
    f"Vehicle weight      = "
    f"{MASS_EXPECTED * 9.81:.9f} N"
)

assert np.linalg.norm(base_qacc[:3]) < 1e-7
assert np.linalg.norm(base_qacc[3:6]) < 1e-7

print("\n[PASS] Equal thrust gives static hover equilibrium.")


# ============================================================
# TEST 6 — Rotor actuators do not drive tilt DOFs
# ============================================================

print()
print("-" * 72)
print("TEST 6 — NO DIRECT THRUST TORQUE ON TILT JOINTS")
print("-" * 72)

mujoco.mj_resetData(model, data)

gravity_original = model.opt.gravity.copy()
model.opt.gravity[:] = 0.0

# Give all four rotors arbitrary nonzero thrust.
data.ctrl[:] = [
    5.0,
    8.0,
    11.0,
    14.0,
]

# Give the rotors arbitrary tilt angles.
angles = [
    math.radians(20),
    math.radians(-30),
    math.radians(45),
    math.radians(-15),
]

for i, angle in enumerate(angles, start=1):
    set_joint_qpos(
        model,
        data,
        f"tilt_{i}",
        angle,
    )

mujoco.mj_forward(
    model,
    data,
)

tilt_generalized_forces = (
    data.qfrc_actuator[6:10].copy()
)

print_vector(
    "Tilt generalized forces",
    tilt_generalized_forces,
)

assert np.allclose(
    tilt_generalized_forces,
    0.0,
    atol=1e-10,
)

model.opt.gravity[:] = gravity_original

print()
print(
    "[PASS] Rotor thrust does not directly drive "
    "the tilt hinge DOFs."
)


# ============================================================
# TEST 7 — Two-second free-flight hover
# ============================================================

print()
print("-" * 72)
print("TEST 7 — FREE-FLIGHT HOVER")
print("-" * 72)

expected_gravity = np.array([0.0, 0.0, -9.81])

assert np.allclose(
    model.opt.gravity,
    expected_gravity,
), (
    f"Unexpected gravity before hover test: "
    f"{model.opt.gravity}"
)

mujoco.mj_resetData(model, data)

if model.na:
    data.act[:] = 0.0

data.ctrl[:] = 0.0
data.ctrl[0:4] = HOVER_THRUST
data.ctrl[4:8] = 0.0

mujoco.mj_forward(model, data)

initial_pos = data.qpos[0:3].copy()
initial_quat = data.qpos[3:7].copy()

duration = 2.0

while data.time < duration:
    mujoco.mj_step(model, data)

final_pos = data.qpos[0:3].copy()
final_quat = data.qpos[3:7].copy()

position_drift = final_pos - initial_pos

# Quaternion relative to initial identity-like orientation.
dot = abs(
    float(
        np.dot(
            initial_quat,
            final_quat,
        )
    )
)

dot = np.clip(dot, -1.0, 1.0)

attitude_drift = 2.0 * math.acos(dot)

print_vector("Initial position", initial_pos)
print_vector("Final position  ", final_pos)
print_vector("Position drift  ", position_drift)

print(
    f"Attitude drift = "
    f"{math.degrees(attitude_drift):.9f} deg"
)

print(
    f"Final base linear speed = "
    f"{np.linalg.norm(data.qvel[:3]):.12f} m/s"
)

print(
    f"Final base angular speed = "
    f"{np.linalg.norm(data.qvel[3:6]):.12f} rad/s"
)

assert np.linalg.norm(position_drift) < 1e-5
assert attitude_drift < math.radians(0.001)

print("\n[PASS] Vehicle remains in free-flight hover.")


# ============================================================
# Final result
# ============================================================

print()
print("=" * 72)
print("ALL CORE PLANT TESTS COMPLETED")
print("=" * 72)
print()