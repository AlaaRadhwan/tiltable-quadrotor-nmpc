"""Verify the CAD visual model mirrors the rigid plant, and relate the CAD tip to the controller tool point.

    tools/python scripts/verify_cad_visual_mapping.py
"""
from __future__ import annotations

import mujoco
import numpy as np

from tiltable_quadrotor_rl.control.end_effector import DEFAULT_END_EFFECTOR
from tiltable_quadrotor_rl.visualization.cad_maneuver import DEFAULT_VISUAL_XML
from tiltable_quadrotor_rl.visualization.servo_overlay import ROTOR_OFFSETS_B, ROTOR_TILT_DIRS_B

m = mujoco.MjModel.from_xml_path(str(DEFAULT_VISUAL_XML))
d = mujoco.MjData(m)
print(f"visual nq={m.nq} nv={m.nv}; free joint qpos[0:7]=p,q_WB; qpos[7:11]=tilt_1..4 (= alpha_i)")

alpha = np.radians([25.0, -40.0, 10.0, 60.0])
d.qpos[0:3] = [0, 0, 1]; d.qpos[3] = 1; d.qpos[7:11] = alpha
mujoco.mj_forward(m, d)
ok = True
for i in range(4):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"rotor_{i + 1}")
    z = d.xmat[b].reshape(3, 3)[:, 2]                      # CAD rotor thrust axis (world == body at identity)
    expect = np.sin(alpha[i]) * ROTOR_TILT_DIRS_B[i] + np.cos(alpha[i]) * np.array([0, 0, 1.0])  # plant convention
    pos = d.xpos[b] - np.array([0, 0, 1.0])
    e, p = np.linalg.norm(z - expect), np.linalg.norm(pos - ROTOR_OFFSETS_B[i])
    ok &= e < 1e-6 and p < 1e-6
    print(f"rotor {i + 1}: thrust-axis err {e:.1e}, position err {p:.1e}")
print("tilt mapping matches plant convention:", ok)

d.qpos[7:11] = 0; mujoco.mj_forward(m, d)
g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "chassis_end_effector")
mid = m.geom_dataid[g]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
V = (d.geom_xmat[g].reshape(3, 3) @ m.mesh_vert[a:a + n].T).T + d.geom_xpos[g] - d.xpos[1]
tip = V[:, 2].max()
tool = DEFAULT_END_EFFECTOR.position_B
print(f"\ncontroller tool point (body frame): {tool}")
print(f"CAD end-effector mesh: z in [{V[:, 2].min():.4f}, {tip:.4f}] m, xy half-width {np.abs(V[:, :2]).max():.4f} m")
print(f"CAD tip extreme is {1000 * (tool[2] - tip):.1f} mm short of the controller point along body z "
      f"(lateral offset {1000 * np.linalg.norm(V[V[:, 2] > tip - 0.003][:, :2].mean(0) - tool[:2]):.2f} mm)")
