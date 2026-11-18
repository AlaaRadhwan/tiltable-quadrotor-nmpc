from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    allocation_matrix,
    rotor_commands_to_virtual,
    thrust_allocation_matrix,
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


NUM_TESTS = 1000
MAX_THRUST = 23.0

RNG = np.random.default_rng(42)


def joint_qpos_address(
    model,
    joint_name: str,
) -> int:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"Joint not found: {joint_name}"
        )

    return int(
        model.jnt_qposadr[joint_id]
    )


print()
print("=" * 72)
print("ANALYTICAL ALLOCATION vs MUJOCO")
print("=" * 72)


model = mujoco.MjModel.from_xml_path(
    str(XML_PATH)
)

data = mujoco.MjData(model)

A = allocation_matrix()

assert A.shape == (6, 8)
assert np.linalg.matrix_rank(A) == 6

print("[PASS] A has full 6-D wrench rank.")


tilt_qpos_addresses = [
    joint_qpos_address(
        model,
        f"tilt_{i + 1}",
    )
    for i in range(4)
]


gravity_original = (
    model.opt.gravity.copy()
)

model.opt.gravity[:] = 0.0


max_A_vs_B = 0.0
max_A_vs_mujoco = 0.0
max_B_vs_mujoco = 0.0


for _ in range(NUM_TESTS):

    thrust = RNG.uniform(
        0.0,
        MAX_THRUST,
        size=4,
    )

    alpha = RNG.uniform(
        -3.0,
        3.0,
        size=4,
    )

    # --------------------------------------------------------
    # Analytical formulation A @ z
    # --------------------------------------------------------

    z = rotor_commands_to_virtual(
        thrust,
        alpha,
    )

    wrench_A = A @ z

    # --------------------------------------------------------
    # Instantaneous formulation B(alpha) @ f
    # --------------------------------------------------------

    B = thrust_allocation_matrix(
        alpha
    )

    wrench_B = (
        B @ thrust
    )

    # --------------------------------------------------------
    # MuJoCo
    # --------------------------------------------------------

    mujoco.mj_resetData(
        model,
        data,
    )

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

    for i in range(4):
        data.qpos[
            tilt_qpos_addresses[i]
        ] = alpha[i]

    data.ctrl[:] = thrust

    mujoco.mj_forward(
        model,
        data,
    )

    wrench_mujoco = (
        data.qfrc_actuator[:6].copy()
    )

    max_A_vs_B = max(
        max_A_vs_B,
        float(
            np.max(
                np.abs(
                    wrench_A - wrench_B
                )
            )
        ),
    )

    max_A_vs_mujoco = max(
        max_A_vs_mujoco,
        float(
            np.max(
                np.abs(
                    wrench_A
                    - wrench_mujoco
                )
            )
        ),
    )

    max_B_vs_mujoco = max(
        max_B_vs_mujoco,
        float(
            np.max(
                np.abs(
                    wrench_B
                    - wrench_mujoco
                )
            )
        ),
    )


model.opt.gravity[:] = (
    gravity_original
)


print()
print(
    "Maximum A vs B error      :",
    f"{max_A_vs_B:.3e}",
)

print(
    "Maximum A vs MuJoCo error :",
    f"{max_A_vs_mujoco:.3e}",
)

print(
    "Maximum B vs MuJoCo error :",
    f"{max_B_vs_mujoco:.3e}",
)


assert max_A_vs_B < 1e-10
assert max_A_vs_mujoco < 1e-7
assert max_B_vs_mujoco < 1e-7

print()
print(
    "[PASS] A @ z, B(alpha) @ f, and MuJoCo "
    "agree over 1000 random configurations."
)


# ============================================================
# Instantaneous B-rank regression checks
# ============================================================

expected_ranks = {
    0.0: 4,
    20.0: 4,
    45.0: 4,
    90.0: 3,
}

for angle_deg, expected_rank in (
    expected_ranks.items()
):

    alpha = np.deg2rad(
        np.full(
            4,
            angle_deg,
        )
    )

    B = thrust_allocation_matrix(
        alpha
    )

    rank = np.linalg.matrix_rank(
        B
    )

    assert rank == expected_rank, (
        f"At {angle_deg} deg, "
        f"expected rank {expected_rank}, "
        f"got {rank}"
    )


print(
    "[PASS] Instantaneous B(alpha) rank "
    "regression checks passed."
)

print()
print("=" * 72)
print("ALL MUJOCO ALLOCATION TESTS PASSED")
print("=" * 72)