from pathlib import Path

import casadi as ca
import mujoco
import numpy as np

from tiltable_quadrotor_rl.control.end_effector import (
    DEFAULT_END_EFFECTOR,
    body_to_end_effector_kinematics,
    body_to_end_effector_kinematics_ca,
    end_effector_motion_to_body_motion,
    quaternion_to_rotation_np,
)
from tiltable_quadrotor_rl.control.nmpc_effector_trajectory import (
    EffectorPoseWaypoint,
    QuinticEffectorPoseTrajectory,
    quaternion_from_rpy,
)
from tiltable_quadrotor_rl.robot.parameters import (
    END_EFFECTOR_POSITION_B,
    END_EFFECTOR_QUATERNION_BE,
    INERTIA_DIAG,
    MASS,
)


XML_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tiltable_quadrotor_rl"
    / "robot"
    / "mjcf"
    / "tiltable_quadrotor_rigid.xml"
)


def quaternion_physical_error(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    left = np.asarray(
        left,
        dtype=float,
    ).reshape(4)

    right = np.asarray(
        right,
        dtype=float,
    ).reshape(4)

    return float(
        min(
            np.max(
                np.abs(
                    left - right
                )
            ),
            np.max(
                np.abs(
                    left + right
                )
            ),
        )
    )


print()
print("=" * 72)
print("END-EFFECTOR MODEL VALIDATION")
print("=" * 72)


# ============================================================
# TEST 1 — centralized transform
# ============================================================

print()
print("-" * 72)
print("TEST 1 — CENTRALIZED TOOL TRANSFORM")
print("-" * 72)

assert np.allclose(
    DEFAULT_END_EFFECTOR.position_B,
    END_EFFECTOR_POSITION_B,
    atol=0.0,
)

assert np.allclose(
    DEFAULT_END_EFFECTOR.quaternion_BE,
    END_EFFECTOR_QUATERNION_BE,
    atol=0.0,
)

print(
    "position_B     :",
    DEFAULT_END_EFFECTOR.position_B,
)

print(
    "quaternion_BE  :",
    DEFAULT_END_EFFECTOR.quaternion_BE,
)

print()
print("[PASS] Controller-side tool transform is centralized.")


# ============================================================
# TEST 2 — MJCF structure and site definition
# ============================================================

print()
print("-" * 72)
print("TEST 2 — MUJOCO TOOL SITE / RIGID STRUCTURE")
print("-" * 72)

model = mujoco.MjModel.from_xml_path(
    str(
        XML_PATH
    )
)

data = mujoco.MjData(
    model
)

assert model.nq == 7
assert model.nv == 6
assert model.nu == 6
assert model.njnt == 1
assert model.nbody == 2

compiled_mass = mujoco.mj_getTotalmass(
    model
)

assert np.isclose(
    compiled_mass,
    MASS,
    atol=1e-12,
)

base_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base",
)

assert base_id >= 0

assert np.allclose(
    model.body_inertia[base_id],
    INERTIA_DIAG,
    atol=1e-12,
)

site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "end_effector",
)

assert site_id >= 0

assert np.allclose(
    model.site_pos[site_id],
    END_EFFECTOR_POSITION_B,
    atol=1e-12,
)

# q_BE is identity in the current design, so the site's local
# orientation matrix must also be identity.
assert np.allclose(
    model.site_quat[site_id],
    END_EFFECTOR_QUATERNION_BE,
    atol=1e-12,
)

tip_geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "end_effector_tip",
)

assert tip_geom_id >= 0

assert model.geom_contype[tip_geom_id] != 0
assert model.geom_conaffinity[tip_geom_id] != 0

print()
print("[PASS] Tool is fixed, collidable, and adds no mechanical DoF.")


# ============================================================
# TEST 3 — analytical pose versus MuJoCo site pose
# ============================================================

print()
print("-" * 72)
print("TEST 3 — ANALYTICAL TOOL POSE VS MUJOCO SITE")
print("-" * 72)

rng = np.random.default_rng(
    7
)

max_position_error = 0.0
max_rotation_error = 0.0

for _ in range(200):
    position_B_W = rng.uniform(
        low=-1.0,
        high=1.0,
        size=3,
    )

    raw_quaternion = rng.normal(
        size=4
    )

    quaternion_WB = (
        raw_quaternion
        / np.linalg.norm(
            raw_quaternion
        )
    )

    data.qpos[0:3] = (
        position_B_W
    )

    data.qpos[3:7] = (
        quaternion_WB
    )

    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    analytical = (
        body_to_end_effector_kinematics(
            position_W=position_B_W,
            velocity_W=np.zeros(3),
            quaternion_WB=quaternion_WB,
            omega_B=np.zeros(3),
        )
    )

    mujoco_position = (
        data.site_xpos[
            site_id
        ].copy()
    )

    mujoco_rotation = (
        data.site_xmat[
            site_id
        ].reshape(
            3,
            3,
        ).copy()
    )

    analytical_rotation = (
        quaternion_to_rotation_np(
            analytical.quaternion_WE
        )
    )

    max_position_error = max(
        max_position_error,
        float(
            np.max(
                np.abs(
                    analytical.position_W
                    - mujoco_position
                )
            )
        ),
    )

    max_rotation_error = max(
        max_rotation_error,
        float(
            np.max(
                np.abs(
                    analytical_rotation
                    - mujoco_rotation
                )
            )
        ),
    )

print(
    "Max position error : "
    f"{max_position_error:.3e} m"
)

print(
    "Max rotation error : "
    f"{max_rotation_error:.3e}"
)

assert max_position_error < 1e-12
assert max_rotation_error < 1e-12

print()
print("[PASS] Analytical end-effector pose matches MuJoCo.")


# ============================================================
# TEST 4 — body -> tool -> body round trip
# ============================================================

print()
print("-" * 72)
print("TEST 4 — RIGID KINEMATICS ROUND TRIP")
print("-" * 72)

max_error = 0.0

for _ in range(500):
    position_B_W = rng.uniform(
        low=-1.0,
        high=1.0,
        size=3,
    )

    velocity_B_W = rng.uniform(
        low=-0.8,
        high=0.8,
        size=3,
    )

    acceleration_B_W = rng.uniform(
        low=-1.0,
        high=1.0,
        size=3,
    )

    raw_quaternion = rng.normal(
        size=4
    )

    quaternion_WB = (
        raw_quaternion
        / np.linalg.norm(
            raw_quaternion
        )
    )

    omega_B = rng.uniform(
        low=-1.0,
        high=1.0,
        size=3,
    )

    omega_dot_B = rng.uniform(
        low=-1.0,
        high=1.0,
        size=3,
    )

    tool = (
        body_to_end_effector_kinematics(
            position_W=position_B_W,
            velocity_W=velocity_B_W,
            quaternion_WB=quaternion_WB,
            omega_B=omega_B,
        )
    )

    R_WB = quaternion_to_rotation_np(
        quaternion_WB
    )

    r = (
        DEFAULT_END_EFFECTOR.position_B
    )

    acceleration_E_W = (
        acceleration_B_W
        + R_WB
        @ (
            np.cross(
                omega_dot_B,
                r,
            )
            + np.cross(
                omega_B,
                np.cross(
                    omega_B,
                    r,
                ),
            )
        )
    )

    R_BE = quaternion_to_rotation_np(
        DEFAULT_END_EFFECTOR.quaternion_BE
    )

    omega_dot_E = (
        R_BE.T
        @ omega_dot_B
    )

    reconstructed = (
        end_effector_motion_to_body_motion(
            position_E_W=tool.position_W,
            velocity_E_W=tool.velocity_W,
            acceleration_E_W=acceleration_E_W,
            quaternion_WE=tool.quaternion_WE,
            omega_E=tool.omega_E,
            omega_dot_E=omega_dot_E,
        )
    )

    errors = [
        np.max(
            np.abs(
                reconstructed.position_W
                - position_B_W
            )
        ),
        np.max(
            np.abs(
                reconstructed.velocity_W
                - velocity_B_W
            )
        ),
        np.max(
            np.abs(
                reconstructed.acceleration_W
                - acceleration_B_W
            )
        ),
        quaternion_physical_error(
            reconstructed.quaternion_WB,
            quaternion_WB,
        ),
        np.max(
            np.abs(
                reconstructed.omega_B
                - omega_B
            )
        ),
        np.max(
            np.abs(
                reconstructed.omega_dot_B
                - omega_dot_B
            )
        ),
    ]

    max_error = max(
        max_error,
        float(
            max(
                errors
            )
        ),
    )

print(
    "Maximum round-trip error : "
    f"{max_error:.3e}"
)

assert max_error < 1e-12

print()
print("[PASS] Rigid tool/body inverse kinematics are consistent.")


# ============================================================
# TEST 5 — NumPy versus CasADi end-effector transform
# ============================================================

print()
print("-" * 72)
print("TEST 5 — NUMPY VS CASADI TOOL TRANSFORM")
print("-" * 72)

p_symbol = ca.SX.sym(
    "p",
    3,
)

v_symbol = ca.SX.sym(
    "v",
    3,
)

q_symbol = ca.SX.sym(
    "q",
    4,
)

omega_symbol = ca.SX.sym(
    "omega",
    3,
)

(
    p_E_symbol,
    v_E_symbol,
    q_E_symbol,
    omega_E_symbol,
) = body_to_end_effector_kinematics_ca(
    p_symbol,
    v_symbol,
    q_symbol,
    omega_symbol,
)

tool_function = ca.Function(
    "tool_kinematics_check",
    [
        p_symbol,
        v_symbol,
        q_symbol,
        omega_symbol,
    ],
    [
        p_E_symbol,
        v_E_symbol,
        q_E_symbol,
        omega_E_symbol,
    ],
)

max_symbolic_error = 0.0

for _ in range(200):
    position = rng.uniform(
        -1.0,
        1.0,
        3,
    )

    velocity = rng.uniform(
        -1.0,
        1.0,
        3,
    )

    raw_quaternion = rng.normal(
        size=4
    )

    quaternion = (
        raw_quaternion
        / np.linalg.norm(
            raw_quaternion
        )
    )

    omega = rng.uniform(
        -1.0,
        1.0,
        3,
    )

    numerical = (
        body_to_end_effector_kinematics(
            position_W=position,
            velocity_W=velocity,
            quaternion_WB=quaternion,
            omega_B=omega,
        )
    )

    symbolic = tool_function(
        position,
        velocity,
        quaternion,
        omega,
    )

    symbolic_p = np.asarray(
        symbolic[0]
    ).reshape(3)

    symbolic_v = np.asarray(
        symbolic[1]
    ).reshape(3)

    symbolic_q = np.asarray(
        symbolic[2]
    ).reshape(4)

    symbolic_omega = np.asarray(
        symbolic[3]
    ).reshape(3)

    errors = [
        np.max(
            np.abs(
                symbolic_p
                - numerical.position_W
            )
        ),
        np.max(
            np.abs(
                symbolic_v
                - numerical.velocity_W
            )
        ),
        quaternion_physical_error(
            symbolic_q,
            numerical.quaternion_WE,
        ),
        np.max(
            np.abs(
                symbolic_omega
                - numerical.omega_E
            )
        ),
    ]

    max_symbolic_error = max(
        max_symbolic_error,
        float(
            max(
                errors
            )
        ),
    )

print(
    "Maximum NumPy/CasADi error : "
    f"{max_symbolic_error:.3e}"
)

assert max_symbolic_error < 1e-12

print()
print("[PASS] NumPy and CasADi tool transforms agree.")


# ============================================================
# TEST 6 — trajectory derivative consistency
# ============================================================

print()
print("-" * 72)
print("TEST 6 — 6-DOF TOOL TRAJECTORY DERIVATIVES")
print("-" * 72)

trajectory = QuinticEffectorPoseTrajectory(
    [
        EffectorPoseWaypoint(
            time_s=0.0,
            position_W=np.array(
                [
                    0.0,
                    0.0,
                    1.5,
                ]
            ),
            quaternion_WE=quaternion_from_rpy(
                0.0,
                0.0,
                0.0,
            ),
        ),
        EffectorPoseWaypoint(
            time_s=5.0,
            position_W=np.array(
                [
                    0.7,
                    -0.4,
                    1.8,
                ]
            ),
            quaternion_WE=quaternion_from_rpy(
                np.deg2rad(
                    14.0
                ),
                np.deg2rad(
                    -11.0
                ),
                np.deg2rad(
                    28.0
                ),
            ),
        ),
    ]
)

dt = 1e-5
test_time = 2.4

minus = trajectory.sample(
    test_time - dt
)

center = trajectory.sample(
    test_time
)

plus = trajectory.sample(
    test_time + dt
)

velocity_fd = (
    plus.position_W
    - minus.position_W
) / (
    2.0 * dt
)

acceleration_fd = (
    plus.velocity_W
    - minus.velocity_W
) / (
    2.0 * dt
)

omega_dot_fd = (
    plus.omega_E
    - minus.omega_E
) / (
    2.0 * dt
)

assert np.max(
    np.abs(
        velocity_fd
        - center.velocity_W
    )
) < 1e-8

assert np.max(
    np.abs(
        acceleration_fd
        - center.acceleration_W
    )
) < 1e-7

assert np.max(
    np.abs(
        omega_dot_fd
        - center.omega_dot_E
    )
) < 1e-7

print()
print("[PASS] Tool trajectory derivatives are self-consistent.")


print()
print("=" * 72)
print("ALL END-EFFECTOR TESTS PASSED")
print("=" * 72)
