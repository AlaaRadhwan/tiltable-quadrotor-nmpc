from pathlib import Path

import mujoco
import numpy as np

from tiltable_quadrotor_rl.robot.parameters import (
    MASS,
    GRAVITY,
    INERTIA_DIAG,
    WEIGHT,
)


XML_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tiltable_quadrotor_rl"
    / "robot"
    / "mjcf"
    / "tiltable_quadrotor_rigid.xml"
)


print()
print("=" * 72)
print("RIGID TILTABLE-QUADROTOR PLANT VALIDATION")
print("=" * 72)

print(f"\nLoading:\n{XML_PATH}\n")


model = mujoco.MjModel.from_xml_path(
    str(XML_PATH)
)

data = mujoco.MjData(model)


# ============================================================
# TEST 1 — compiled structure
# ============================================================

print("-" * 72)
print("TEST 1 — MODEL STRUCTURE")
print("-" * 72)

print(f"nq    = {model.nq}")
print(f"nv    = {model.nv}")
print(f"nu    = {model.nu}")
print(f"na    = {model.na}")
print(f"njnt  = {model.njnt}")
print(f"nbody = {model.nbody}")

assert model.nq == 7
assert model.nv == 6

# Actuation is intentionally external to the nominal MJCF.
assert model.nu == 6
assert model.na == 0

# World + one free rigid body.
assert model.nbody == 2

# Only the free joint.
assert model.njnt == 1

print()
print("[PASS] Nominal plant has exactly six mechanical DoFs.")


# ============================================================
# TEST 2 — total mass
# ============================================================

print()
print("-" * 72)
print("TEST 2 — MASS")
print("-" * 72)

mujoco.mj_resetData(
    model,
    data,
)

mujoco.mj_forward(
    model,
    data,
)

compiled_mass = mujoco.mj_getTotalmass(
    model
)

print(f"Expected : {MASS:.9f} kg")
print(f"Compiled : {compiled_mass:.9f} kg")

assert np.isclose(
    compiled_mass,
    MASS,
    atol=1e-12,
)

print()
print("[PASS] Total mass is correct.")


# ============================================================
# TEST 3 — inertia
# ============================================================

print()
print("-" * 72)
print("TEST 3 — BODY INERTIA")
print("-" * 72)

base_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base",
)

assert base_id >= 0

compiled_inertia = (
    model.body_inertia[base_id].copy()
)

print(f"Expected : {INERTIA_DIAG}")
print(f"Compiled : {compiled_inertia}")

assert np.allclose(
    compiled_inertia,
    INERTIA_DIAG,
    atol=1e-12,
)

print()
print("[PASS] Body inertia is correct.")


# ============================================================
# TEST 4 — gravity-only acceleration
# ============================================================

print()
print("-" * 72)
print("TEST 4 — FREE FALL")
print("-" * 72)

mujoco.mj_resetData(
    model,
    data,
)

mujoco.mj_forward(
    model,
    data,
)

expected = np.array(
    [0.0, 0.0, -GRAVITY]
)

actual = data.qacc[0:3].copy()

print(f"Expected linear qacc : {expected}")
print(f"Actual linear qacc   : {actual}")

assert np.allclose(
    actual,
    expected,
    atol=1e-10,
)

assert np.linalg.norm(
    data.qacc[3:6]
) < 1e-12

print()
print("[PASS] Gravity produces the expected free-fall acceleration.")


# ============================================================
# TEST 5 — hover through Cartesian applied wrench
# ============================================================

print()
print("-" * 72)
print("TEST 5 — APPLIED-WRENCH HOVER")
print("-" * 72)

mujoco.mj_resetData(
    model,
    data,
)


# [Fx, Fy, Fz, Tx, Ty, Tz]
#
# applied to the body's center of mass.
data.ctrl[:] = np.array(
    [
        0.0,
        0.0,
        WEIGHT,
        0.0,
        0.0,
        0.0,
    ]
)

mujoco.mj_forward(
    model,
    data,
)

print(
    "Base acceleration:",
    data.qacc[0:6],
)

assert np.linalg.norm(
    data.qacc[0:3]
) < 1e-9

assert np.linalg.norm(
    data.qacc[3:6]
) < 1e-9

print()
print(
    "[PASS] A Cartesian upward force equal to weight "
    "produces static hover."
)


# ============================================================
# TEST 6 — known torque response
# ============================================================

print()
print("-" * 72)
print("TEST 6 — KNOWN BODY TORQUE")
print("-" * 72)

mujoco.mj_resetData(
    model,
    data,
)

# Remove gravity so this test isolates rotation.
gravity_original = model.opt.gravity.copy()

model.opt.gravity[:] = 0.0

# Identity orientation means body and world axes coincide.
test_torque = 1.0

data.ctrl[:] = np.array(
    [
        0.0,
        0.0,
        0.0,
        test_torque,
        0.0,
        0.0,
    ]
)

mujoco.mj_forward(
    model,
    data,
)
expected_alpha_x = (
    test_torque
    / INERTIA_DIAG[0]
)

actual_alpha_x = data.qacc[3]

print(
    f"Expected angular acceleration X : "
    f"{expected_alpha_x:.9f} rad/s^2"
)

print(
    f"Actual angular acceleration X   : "
    f"{actual_alpha_x:.9f} rad/s^2"
)

assert np.isclose(
    actual_alpha_x,
    expected_alpha_x,
    atol=1e-9,
)

assert np.linalg.norm(
    data.qacc[4:6]
) < 1e-10

model.opt.gravity[:] = (
    gravity_original
)

print()
print("[PASS] Applied torque follows tau = I * alpha.")


print()
print("=" * 72)
print("ALL RIGID PLANT TESTS PASSED")
print("=" * 72)
