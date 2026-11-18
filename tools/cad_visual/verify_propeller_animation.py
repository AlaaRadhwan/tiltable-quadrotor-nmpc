"""Numeric checks for the scene-only propeller animation (Phase 2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tiltable_quadrotor_rl.visualization.propeller_spin import DEFAULT_AXES_JSON, N_ROTORS, PropellerSpinAnimator, SpinConfig  # noqa: E402

XML = ROOT / "assets/cad_visual/tiltable_quadrotor_visual.xml"
fails = []


def check(label, ok, detail=""):
    print(f"[{'OK' if ok else 'FAIL'}] {label} {detail}")
    if not ok:
        fails.append(label)


def rot_axis_angle(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return ax / np.linalg.norm(ax) if np.linalg.norm(ax) > 1e-12 else np.zeros(3), ang


def pose(m, base_euler, tilts):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    q = np.zeros(4); mujoco.mju_euler2Quat(q, np.array(base_euler, float), "xyz")
    d.qpos[3:7] = q; d.qpos[7:11] = tilts
    mujoco.mj_forward(m, d)
    return d


def main():
    m = mujoco.MjModel.from_xml_path(str(XML))
    axes = json.loads(Path(DEFAULT_AXES_JSON).read_text())["rotors"]
    cfg = SpinConfig(blur_ghosts=0)
    a = PropellerSpinAnimator(m, cfg)

    # --- axis: vertical in the rotor frame and through the (generic) shaft and blade centres
    for i in range(1, 5):
        ax = np.array(axes[f"rotor_{i}"]["axis_local_unit"]); pt = np.array(axes[f"rotor_{i}"]["point_local_m"])
        check(f"rotor_{i} spin axis parallel to rotor local Z", abs(abs(ax[2]) - 1) < 1e-6)
        err = []
        for name in (f"rotor_{i}_prop", f"rotor_{i}_motor_shaft"):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            err.append(np.linalg.norm(m.geom_pos[g][:2] - pt[:2]) if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_CYLINDER
                       else np.linalg.norm((m.geom_pos[g][:2]) - pt[:2]))
        check(f"rotor_{i} axis passes through blade/shaft centres (lateral err mm)", max(err) < 1e-3, f"{1e3 * max(err):.3f}")

    # --- direction convention
    gear_z = np.array([m.actuator_gear[j, 5] for j in a.ctrl_idx])
    check("spin sign = -sign(reaction-torque gear z)", np.array_equal(a.spin_sign, -np.sign(gear_z)), str(a.spin_sign))
    check("adjacent rotors counter-rotate", all(a.spin_sign[i] == -a.spin_sign[(i + 1) % 4] for i in range(4)))

    # --- thrust -> speed law, simulation-time integration, determinism
    s1, s2 = a.visual_speed([cfg.hover_thrust] * 4), a.visual_speed([2 * cfg.hover_thrust] * 4)
    check("speed(2f)/speed(f) = sqrt(2)", np.allclose(s2 / s1, np.sqrt(2)), f"{(s2 / s1)[0]:.6f}")
    check("zero thrust -> zero speed; cap respected", np.all(a.visual_speed([0] * 4) == 0) and np.all(a.visual_speed([1e3] * 4) <= cfg.max_speed + 1e-12))
    f = [3, 6, 9, 12.0]
    a.reset(); a.advance(0.1, f); one = a.angles.copy()
    a.reset(); [a.advance(0.01, f) for _ in range(10)]
    check("angle integral independent of dt splitting (sim time only)", np.allclose(one, a.angles, atol=1e-9))
    b = PropellerSpinAnimator(m, cfg); [b.advance(0.0333, f) for _ in range(50)]
    c = PropellerSpinAnimator(m, cfg); [c.advance(0.0333, f) for _ in range(50)]
    check("repeat runs bit-identical", np.array_equal(b.angles, c.angles))

    # --- geometry in scene: default view has no static duplicates; spun copies about the right axis
    d0 = pose(m, [0, 0, 0], [0] * 4)
    opt = mujoco.MjvOption(); scn = mujoco.MjvScene(m, maxgeom=2000)
    mujoco.mjv_updateScene(m, d0, opt, None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scn)
    static_props = [k for k in range(scn.ngeom) if scn.geoms[k].objtype == mujoco.mjtObj.mjOBJ_GEOM and m.geom_group[scn.geoms[k].objid] == 3]
    check("static propeller geoms are not drawn by default", len(static_props) == 0)
    n0 = scn.ngeom
    n_add = a.add_to_scene(d0, scn)
    check("animator adds exactly blade+shaft per rotor (8 geoms, ghosts off)", n_add == 8 and scn.ngeom == n0 + 8, f"added {n_add}")
    ga = PropellerSpinAnimator(m, SpinConfig()); ga.advance(0.01, [cfg.hover_thrust] * 4)
    scn.ngeom = n0; n_g = ga.add_to_scene(d0, scn)
    check("ghost blur adds 3 ghost blades per rotor", n_g == 8 + 4 * 3, f"added {n_g}")

    rng = np.random.default_rng(1)
    cases = [([0, 0, 0], [0] * 4), ([0, 0, 0], list(np.radians([90, -90, 90, -90]))), ([0.7, -0.4, 2.0], list(np.radians([45, -60, 90, -90]))),
             ([np.pi, 0.3, 1.0], list(np.radians([-90, 90, 30, 0]))), ([1.2, 1.1, -2.5], list(rng.uniform(-np.pi / 2, np.pi / 2, 4)))]
    ok_axis = ok_piv = ok_ang = True
    for eul, tl in cases:
        d = pose(m, eul, tl)
        a.reset(); a.angles[:] = np.array([0.3, 1.1, 2.0, 4.0])
        scn = mujoco.MjvScene(m, maxgeom=100); a.add_to_scene(d, scn)
        k = 0
        for r in range(N_ROTORS):
            Rb = d.xmat[a.body_id[r]].reshape(3, 3)
            pivot = d.xpos[a.body_id[r]] + Rb @ a.point[r]
            axis_w = Rb @ a.axis[r]
            for g in a.geom_ids[r]:
                sg = scn.geoms[k]; k += 1
                Rs = np.array(sg.mat).reshape(3, 3) @ d.geom_xmat[g].reshape(3, 3).T
                ax, ang = rot_axis_angle(Rs)
                ok_axis &= abs(abs(ax @ axis_w) - 1) < 1e-9
                signed = ang * np.sign(ax @ axis_w)
                ok_ang &= abs(np.mod(signed - a.angles[r] + np.pi, 2 * np.pi) - np.pi) < 2e-6  # scene geoms are float32
                # pivot invariance: x' - pivot = Rs (x - pivot)
                ok_piv &= np.allclose(np.array(sg.pos) - pivot, Rs @ (d.geom_xpos[g] - pivot), atol=2e-6)
    check("spun copies rotate about world-transformed measured axis (tilt +-90, arbitrary aircraft attitude)", ok_axis)
    check("rotation angle equals accumulated angle", ok_ang)
    check("spin pivot (axis point) is a fixed point of every spun geom", ok_piv)

    print("\nRESULT:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
