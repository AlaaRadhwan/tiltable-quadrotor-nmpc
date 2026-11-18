"""Scene-only propeller spin animation for the CAD visual MJCF.

The visual MJCF (``assets/cad_visual/tiltable_quadrotor_visual.xml``) contains the
propeller blade and motor shaft as non-colliding geoms
in hidden geom group 3, parented to each ``rotor_i`` body.  They therefore follow the
existing ``tilt_i`` joints and the free joint, but are never drawn by default.

``PropellerSpinAnimator`` draws spun copies of those same geoms (meshes in the detailed
variant, primitives in the public one) by appending geoms to an
``mujoco.MjvScene``.  Nothing in ``MjModel``/``MjData`` is modified: no joints, DoFs,
actuators, inertias or forces are added, and the physical simulation is unaffected.

Usage (offscreen)::

    anim = PropellerSpinAnimator(model)
    ...
    anim.advance(dt, data.ctrl[:4])              # simulation-time step, thrusts in N
    renderer.update_scene(data, camera=cam)
    anim.add_to_scene(data, renderer.scene)
    frame = renderer.render()

Usage (passive viewer)::

    viewer.user_scn.ngeom = 0
    anim.add_to_scene(data, viewer.user_scn)
    viewer.sync()

Conventions
-----------
* Spin axis: measured CAD spin-Revolute axis (``assets/cad_visual/propeller_spin_axes.json``,
  derived from ``assembly_observations.json``), expressed in the rotor body frame.  Because
  the data are in the rotor frame, the axis follows tilt and whole-aircraft rotation.
* Direction: the actuator gear z-torque is the *reaction* torque on the airframe,
  ``tau_z = -sign(w_z) * kq/kt * f``.  Hence ``sign(w_z) = -sign(gear_z)``, where ``w_z`` is
  the propeller angular velocity about the rotor frame's local +Z (thrust direction).
  Rotors 1,3 spin +Z (CCW seen from above at zero tilt); rotors 2,4 spin -Z.
* Speed: visual speed ``w = min(w_max, w_hover * (f / f_hover) ** exponent)`` with
  ``exponent = 0.5`` (thrust ~ w^2).  ``w_hover`` is a deliberately slow *visual* scale
  so that frame-rate aliasing is avoided; it is not the physical rotor speed.
* Time: angles integrate the supplied simulation ``dt`` (never wall-clock time).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_AXES_JSON = Path(__file__).resolve().parents[3] / "assets" / "cad_visual" / "propeller_spin_axes.json"
N_ROTORS = 4
SPIN_GEOM_GROUP = 3  # hidden-by-default group holding the static CAD propeller geoms


@dataclass
class SpinConfig:
    hover_thrust: float = 7.4531475  # N, thrust per rotor at hover (keyframe "hover")
    hover_speed: float = 40.0        # rad/s visual speed at hover thrust (~6.4 rev/s)
    max_speed: float = 75.0          # rad/s cap, keeps 2-blade aliasing away at >=30 fps
    speed_exponent: float = 0.5      # thrust ~ speed**(1/exponent)  (square-root law)
    blur_ghosts: int = 3             # trailing translucent blade copies (0 disables blur)
    blur_alpha_max: float = 0.30     # opacity of the first ghost at/above blur_full_speed
    blur_full_speed: float = 60.0    # rad/s at which ghosts reach blur_alpha_max
    blur_exposure: float = 0.02      # s of rotation spanned by the ghost trail
    scale: float = 1.0               # global multiplier on the visual speed (before the cap)


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


_COPY_FIELDS = ("dataid", "matid", "texcoord", "segid", "objtype", "objid", "category", "size",
                "rgba", "emission", "specular", "shininess", "reflectance", "transparent")


class PropellerSpinAnimator:
    def __init__(self, model: mujoco.MjModel, config: SpinConfig | None = None,
                 axes_json: str | Path = DEFAULT_AXES_JSON):
        self.model = model
        self.cfg = config or SpinConfig()
        axes = json.loads(Path(axes_json).read_text())["rotors"]
        M = mujoco.mjtObj

        self.body_id, self.point, self.axis, self.spin_sign, self.ctrl_idx = [], [], [], [], []
        self.geom_ids: list[list[int]] = []
        self._blade_geom = []  # blade mesh geom id per rotor (ghost-blur source)
        for i in range(1, N_ROTORS + 1):
            b = mujoco.mj_name2id(model, M.mjOBJ_BODY, f"rotor_{i}")
            site = mujoco.mj_name2id(model, M.mjOBJ_SITE, f"rotor_{i}_thrust")
            if b < 0 or site < 0:
                raise ValueError(f"model lacks rotor_{i} body / rotor_{i}_thrust site")
            self.body_id.append(b)
            a = np.array(axes[f"rotor_{i}"]["axis_local_unit"], float)
            a = a if a @ np.array([0, 0, 1.0]) >= 0 else -a   # orient along rotor +Z (thrust direction)
            self.axis.append(a)
            self.point.append(np.array(axes[f"rotor_{i}"]["point_local_m"], float))
            j = next(k for k in range(model.nu) if model.actuator_trntype[k] == mujoco.mjtTrn.mjTRN_SITE
                     and model.actuator_trnid[k, 0] == site)
            self.ctrl_idx.append(j)
            # reaction torque on the airframe is opposite to the prop angular velocity
            self.spin_sign.append(-float(np.sign(model.actuator_gear[j, 5])) or 1.0)
            gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b
                    and model.geom_group[g] == SPIN_GEOM_GROUP]
            if not gids:
                raise ValueError(f"rotor_{i}: no geoms in group {SPIN_GEOM_GROUP}")
            self.geom_ids.append(gids)
            self._blade_geom.append(next((g for g in gids if mujoco.mj_id2name(model, M.mjOBJ_GEOM, g) == f"rotor_{i}_prop"), None))
        self.spin_sign = np.array(self.spin_sign)
        self._templates = self._build_templates()
        self.angles = np.zeros(N_ROTORS)
        self.speeds = np.zeros(N_ROTORS)

    # ------------------------------------------------------------------ setup helpers
    def _build_templates(self):
        """Capture scene geom structs for the hidden prop geoms (materials, mesh ids, ...)."""
        m = self.model
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        opt = mujoco.MjvOption()
        opt.geomgroup[SPIN_GEOM_GROUP] = 1
        scn = mujoco.MjvScene(m, maxgeom=4000)
        mujoco.mjv_updateScene(m, d, opt, None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scn)
        want = {g for gl in self.geom_ids for g in gl}
        tpl = {}
        for k in range(scn.ngeom):
            sg = scn.geoms[k]
            if sg.objtype == mujoco.mjtObj.mjOBJ_GEOM and sg.objid in want:
                tpl[sg.objid] = {f: np.array(getattr(sg, f)).copy() if hasattr(getattr(sg, f), "__len__") else getattr(sg, f)
                                 for f in _COPY_FIELDS}
        missing = want - set(tpl)
        if missing:
            raise RuntimeError(f"could not capture scene templates for geoms {sorted(missing)}")
        return tpl

    # ------------------------------------------------------------------ time stepping
    def visual_speed(self, thrust) -> np.ndarray:
        """Visual angular speed (rad/s, >= 0) for each rotor from thrust (N)."""
        f = np.clip(np.asarray(thrust, float)[:N_ROTORS], 0.0, None)
        w = self.cfg.scale * self.cfg.hover_speed * (f / self.cfg.hover_thrust) ** self.cfg.speed_exponent
        return np.minimum(w, self.cfg.max_speed)

    def advance(self, dt: float, thrust) -> np.ndarray:
        """Integrate prop angles over a *simulation* interval ``dt`` at the given thrusts (N)."""
        self.speeds = self.visual_speed(thrust)
        self.angles = np.mod(self.angles + self.spin_sign * self.speeds * dt, 2 * np.pi)
        return self.angles

    def advance_from_data(self, dt: float, data: mujoco.MjData) -> np.ndarray:
        return self.advance(dt, [data.ctrl[j] for j in self.ctrl_idx])

    def reset(self, angles=None):
        self.angles = np.zeros(N_ROTORS) if angles is None else np.asarray(angles, float).copy()
        self.speeds = np.zeros(N_ROTORS)

    # ------------------------------------------------------------------ scene injection
    def add_to_scene(self, data: mujoco.MjData, scene: mujoco.MjvScene) -> int:
        """Append spun propeller meshes (plus fading ghost blades for motion blur) to ``scene``.

        ``data`` must hold current kinematics (``mj_forward``/``mj_step``).  Returns the
        number of geoms added.  Existing scene contents are left untouched.
        """
        added = 0
        for r in range(N_ROTORS):
            b = self.body_id[r]
            Rb = data.xmat[b].reshape(3, 3)
            pivot = data.xpos[b] + Rb @ self.point[r]
            R = _rodrigues(Rb @ self.axis[r], self.angles[r])
            for g in self.geom_ids[r]:
                if scene.ngeom >= scene.maxgeom:
                    warnings.warn("scene full: propeller geoms skipped")
                    return added
                sg = scene.geoms[scene.ngeom]
                self._fill_geom(sg, g, pivot + R @ (data.geom_xpos[g] - pivot), R @ data.geom_xmat[g].reshape(3, 3))
                scene.ngeom += 1
                added += 1
            if self.cfg.blur_ghosts > 0:
                added += self._add_ghost_blades(data, scene, r, pivot, Rb)
        return added

    def _add_ghost_blades(self, data, scene, r, pivot, Rb) -> int:
        """Fading copies of the blade trailing behind the rotation (subtle motion blur).

        Ghost spacing follows the visual speed over a nominal exposure time, so the trail
        is invisible at rest and lengthens with thrust.  Only the blade mesh is ghosted.
        """
        cfg = self.cfg
        g = self._blade_geom[r]
        if g is None or self.speeds[r] < 1e-6:
            return 0
        amax = cfg.blur_alpha_max * min(self.speeds[r] / cfg.blur_full_speed, 1.0)
        n = cfg.blur_ghosts
        added = 0
        for k in range(1, n + 1):
            if scene.ngeom >= scene.maxgeom:
                break
            lag = self.spin_sign[r] * self.speeds[r] * cfg.blur_exposure * k / n
            R = _rodrigues(Rb @ self.axis[r], self.angles[r] - lag)
            rgba = np.array(self._templates[g]["rgba"], dtype=np.float32)
            rgba[3] = amax * (1.0 - (k - 1) / n)
            sg = scene.geoms[scene.ngeom]
            self._fill_geom(sg, g, pivot + R @ (data.geom_xpos[g] - pivot), R @ data.geom_xmat[g].reshape(3, 3), rgba)
            scene.ngeom += 1
            added += 1
        return added

    def _fill_geom(self, sg, g, pos, mat, rgba=None):
        t = self._templates[g]
        mujoco.mjv_initGeom(sg, int(self.model.geom_type[g]), self.model.geom_size[g], pos, mat.ravel(),
                            (t["rgba"] if rgba is None else rgba).astype(np.float32))
        for f in _COPY_FIELDS:
            if f == "rgba":
                continue
            v = t[f]
            if hasattr(v, "__len__"):
                getattr(sg, f)[:] = v
            else:
                setattr(sg, f, v)
