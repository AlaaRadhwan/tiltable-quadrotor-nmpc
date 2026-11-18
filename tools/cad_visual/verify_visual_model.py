"""Verify the visual MJCF leaves the validated dynamic model unchanged."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORIG = ROOT / "src/tiltable_quadrotor_rl/robot/mjcf/tiltable_quadrotor.xml"
VIS = ROOT / "assets/cad_visual/tiltable_quadrotor_visual.xml"


def name(m, obj, i):
    return mujoco.mj_id2name(m, obj, i)


def main() -> int:
    a, b = mujoco.MjModel.from_xml_path(str(ORIG)), mujoco.MjModel.from_xml_path(str(VIS))
    fails = []

    def check(label, ok, detail=""):
        print(f"[{'OK' if ok else 'FAIL'}] {label} {detail}")
        if not ok:
            fails.append(label)

    for k in ("nq", "nv", "nu", "na", "nbody", "njnt", "nsite", "nsensor", "nkey", "nmocap", "neq"):
        check(k, getattr(a, k) == getattr(b, k), f"{getattr(a, k)} vs {getattr(b, k)}")
    check("opt.timestep/gravity/integrator", a.opt.timestep == b.opt.timestep and np.array_equal(a.opt.gravity, b.opt.gravity)
          and a.opt.integrator == b.opt.integrator)
    eq = lambda x, y: np.array_equal(x, y)
    for f in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat", "body_parentid",
              "jnt_type", "jnt_axis", "jnt_pos", "jnt_range", "jnt_limited", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr",
              "dof_armature", "dof_damping", "dof_frictionloss",
              "site_pos", "site_quat", "site_bodyid", "actuator_gear", "actuator_ctrlrange", "actuator_ctrllimited",
              "actuator_trntype", "actuator_trnid", "actuator_gainprm", "actuator_biasprm", "actuator_dyntype",
              "sensor_type", "sensor_objid", "sensor_dim", "key_qpos", "key_ctrl"):
        check(f, eq(getattr(a, f), getattr(b, f)))
    for obj, n in ((mujoco.mjtObj.mjOBJ_BODY, "nbody"), (mujoco.mjtObj.mjOBJ_JOINT, "njnt"), (mujoco.mjtObj.mjOBJ_SITE, "nsite"),
                   (mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"), (mujoco.mjtObj.mjOBJ_SENSOR, "nsensor")):
        check(f"names {obj.name}", [name(a, obj, i) for i in range(getattr(a, n))] == [name(b, obj, i) for i in range(getattr(b, n))])
    check("total mass", a.body_subtreemass[1] == b.body_subtreemass[1], f"{a.body_subtreemass[1]:.6f}")

    # Added geoms must be visual-only; pre-existing geoms must keep collision settings.
    na_geom = {name(a, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(a.ngeom)}
    added = [i for i in range(b.ngeom) if name(b, mujoco.mjtObj.mjOBJ_GEOM, i) not in na_geom]
    check("added mesh geoms non-colliding", all(b.geom_contype[i] == 0 and b.geom_conaffinity[i] == 0 for i in added), f"({len(added)} added)")
    allowed = (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_ELLIPSOID)
    check("added geoms are meshes or generic primitives", all(b.geom_type[i] in allowed for i in added))
    for n, i in na_geom.items():
        j = mujoco.mj_name2id(b, mujoco.mjtObj.mjOBJ_GEOM, n)
        same = (a.geom_contype[i], a.geom_conaffinity[i], a.geom_type[i]) == (b.geom_contype[j], b.geom_conaffinity[j], b.geom_type[j]) \
            and eq(a.geom_pos[i], b.geom_pos[j]) and (n == "ground_visual" or eq(a.geom_size[i], b.geom_size[j]))  # ground is deliberately enlarged (visual only)
        check(f"original geom {n}", same)

    # Dynamic equivalence: identical rollout under the same controls.
    da, db = mujoco.MjData(a), mujoco.MjData(b)
    mujoco.mj_resetDataKeyframe(a, da, 0); mujoco.mj_resetDataKeyframe(b, db, 0)
    rng = np.random.default_rng(0)
    for _ in range(500):
        u = rng.uniform(0, 23, 4)
        da.ctrl[:], db.ctrl[:] = u, u
        mujoco.mj_step(a, da); mujoco.mj_step(b, db)
    check("500-step rollout identical (qpos)", np.array_equal(da.qpos, db.qpos), f"max|dq|={np.abs(da.qpos - db.qpos).max():.3e}")
    check("500-step rollout identical (sensordata)", np.array_equal(da.sensordata, db.sensordata))
    # Tilt-follow test: rotor geoms must rigidly rotate about each tilt axis; chassis geoms must not move.
    m = b
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    gname = [name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
    x0, R0 = d.geom_xpos.copy(), d.geom_xmat.copy()
    ang = 0.7
    for r in range(1, 5):
        d2 = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d2, 0)
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"tilt_{r}")
        d2.qpos[m.jnt_qposadr[jid]] = ang; mujoco.mj_forward(m, d2)
        bid = m.jnt_bodyid[jid]
        axis = d2.xaxis[jid]; origin = d.xanchor[jid] if False else d2.xanchor[jid]
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        Rot = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
        moved_ok = static_ok = True; n_moved = 0
        for i, n in enumerate(gname):
            if n is None: continue
            if n.startswith(f"rotor_{r}_") and not n.endswith("_visual"):
                n_moved += 1
                moved_ok &= np.allclose(d2.geom_xpos[i], origin + Rot @ (x0[i] - origin), atol=1e-9)
                moved_ok &= np.allclose(d2.geom_xmat[i].reshape(3, 3), Rot @ R0[i].reshape(3, 3), atol=1e-9)
            else:
                static_ok &= np.allclose(d2.geom_xpos[i], x0[i], atol=1e-12) if not n.startswith("rotor_") else True
        check(f"tilt_{r} drives {n_moved} rotor_{r}_* mesh geoms rigidly about CAD tilt axis; chassis static", moved_ok and static_ok)
    print("\nRESULT:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
