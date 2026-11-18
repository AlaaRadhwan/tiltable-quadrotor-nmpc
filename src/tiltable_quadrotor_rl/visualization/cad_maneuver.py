"""CAD visual adapter for closed-loop benchmarks.

The benchmark physics keeps running on the original rigid plant MJCF (no tilt joints,
wrench actuators).  This adapter owns a *separate* ``MjModel``/``MjData`` loaded from the
CAD visual MJCF and mirrors the plant state onto it purely for display:

    qpos_visual = [position_W (3), quaternion_WB (4), alpha_1..4 (4)]

The visual model has the same free-joint layout plus four ``tilt_i`` hinges whose sign
convention matches the NMPC ``alpha_i`` (verified by ``scripts/verify_cad_visual_mapping.py``).
Nothing here feeds back into the simulation or the controller.

Propeller spin is driven by the commanded rotor thrust and integrated over *simulation*
time (``advance``), reusing ``PropellerSpinAnimator``.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from tiltable_quadrotor_rl.visualization.propeller_spin import (
    PropellerSpinAnimator,
    SpinConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VISUAL_XML = PROJECT_ROOT / "assets" / "cad_visual" / "tiltable_quadrotor_visual.xml"

TARGET_RGBA = np.array([0.1, 1.0, 0.1, 0.70])
EFFECTOR_RGBA = np.array([1.0, 0.55, 0.1, 1.0])


class FollowCamera:
    """Smooth, sim-time-driven tracking camera (critically damped lookat)."""

    def __init__(self, *, distance: float, azimuth: float, elevation: float,
                 azimuth_rate: float = 0.0, tau: float = 0.45, lookat_offset=(0, 0, 0.05)):
        self.distance, self.azimuth0, self.elevation = distance, azimuth, elevation
        self.azimuth_rate, self.tau = azimuth_rate, tau
        self.offset = np.asarray(lookat_offset, float)
        self.lookat: np.ndarray | None = None
        self.fixed = False
        self.vel = np.zeros(3)
        self.t = 0.0

    def update(self, target: np.ndarray, t: float) -> None:
        if self.fixed:
            self.t = t
            return
        target = np.asarray(target, float) + self.offset
        if self.lookat is None:
            self.lookat = target.copy()
        else:
            dt = max(t - self.t, 0.0)
            w = 2.0 / self.tau                      # critically damped 2nd-order follower
            for _ in range(max(1, int(np.ceil(dt / 0.005)))):
                h = dt / max(1, int(np.ceil(dt / 0.005)))
                acc = w * w * (target - self.lookat) - 2.0 * w * self.vel
                self.vel += acc * h
                self.lookat += self.vel * h
        self.t = t

    def apply(self, cam: mujoco.MjvCamera, *, orbit: bool = True) -> None:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.lookat
        if orbit:
            cam.distance, cam.elevation = self.distance, self.elevation
            cam.azimuth = self.azimuth0 + self.azimuth_rate * self.t


class CadManeuverVisual:
    def __init__(self, *, xml: Path = DEFAULT_VISUAL_XML, spin: SpinConfig | None = None,
                 main_camera: dict | None = None, tool_camera: dict | None = None):
        """``main_camera`` (fixed): distance/azimuth/elevation/lookat overrides.
        ``tool_camera`` (inset, follows the tool point): distance/azimuth/elevation/tau overrides."""
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        if self.model.nq != 11 or self.model.nv != 10:
            raise RuntimeError("CAD visual model must have a free joint plus four tilt hinges")
        self.anim = PropellerSpinAnimator(self.model, spin)
        self.ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "body_origin")
        # Fixed main camera framing the whole maneuver workspace (tool center 1.5 m, +-0.35 m).
        mc = dict(distance=2.0, azimuth=140.0, elevation=-15.0, lookat=(0.0, 0.0, 1.45))
        mc.update(main_camera or {})
        self.cam_main = FollowCamera(distance=mc["distance"], azimuth=mc["azimuth"], elevation=mc["elevation"],
                                     azimuth_rate=0.0)
        self.cam_main.lookat = np.array(mc["lookat"], float)
        self.cam_main.fixed = True
        tc = dict(distance=0.36, azimuth=35.0, elevation=-12.0, tau=0.12)
        tc.update(tool_camera or {})
        self.cam_tool = FollowCamera(distance=tc["distance"], azimuth=tc["azimuth"], elevation=tc["elevation"],
                                     azimuth_rate=0.0, tau=tc["tau"], lookat_offset=(0, 0, 0))
        self.target_W = np.zeros(3)
        self.effector_W = np.zeros(3)
        self._renderers = {}
        self._writer = None

    # ------------------------------------------------------------------ state mirroring
    def advance_props(self, dt: float, thrust) -> None:
        """Integrate propeller angles over a simulation interval at the commanded thrust."""
        self.anim.advance(dt, thrust)

    def set_state(self, state17: np.ndarray, *, target_W, effector_W) -> None:
        """Mirror [p, v, q, w, alpha] (NMPC state) onto the visual model (no stepping)."""
        d = self.data
        d.qpos[0:3] = state17[0:3]
        d.qpos[3:7] = state17[6:10]
        d.qpos[7:11] = state17[13:17]
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        self.target_W = np.asarray(target_W, float).copy()
        self.effector_W = np.asarray(effector_W, float).copy()

    def _markers(self, scene) -> None:
        for pos, rgba, r in ((self.target_W, TARGET_RGBA, 0.014), (self.effector_W, EFFECTOR_RGBA, 0.007)):
            if scene.ngeom >= scene.maxgeom:
                return
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, r),
                                pos, np.eye(3).ravel(), rgba.astype(np.float32))
            scene.ngeom += 1

    def decorate(self, scene) -> None:
        """Append spun propellers and the two small markers to a scene."""
        self.anim.add_to_scene(self.data, scene)
        self._markers(scene)

    # ------------------------------------------------------------------ offscreen MP4
    def open_video(self, path: Path, *, fps: int = 30, size=(1920, 1080), inset_scale: float = 0.3) -> None:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        self.fps, self.size = fps, tuple(size)
        w, h = self.size
        self.inset = (int(w * inset_scale), int(h * inset_scale))
        self._renderers["main"] = mujoco.Renderer(self.model, h, w)
        self._renderers["tool"] = mujoco.Renderer(self.model, self.inset[1], self.inset[0])
        self._writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                                          macro_block_size=1, pixelformat="yuv420p")
        self.frames = 0
        self.next_frame_t = 0.0

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def frame_due(self, t: float) -> bool:
        return self.recording and t + 1e-9 >= self.next_frame_t

    def write_frame(self, t: float, tool_point_W) -> None:
        from PIL import Image, ImageDraw, ImageFont

        self.cam_main.update(self.data.xpos[1], t)
        self.cam_tool.update(tool_point_W, t)
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        imgs = {}
        for key, fc, orbit in (("main", self.cam_main, True), ("tool", self.cam_tool, True)):
            fc.apply(cam, orbit=orbit)
            r = self._renderers[key]
            r.update_scene(self.data, camera=cam, scene_option=opt)
            self.decorate(r.scene)
            imgs[key] = r.render()
        img = Image.fromarray(imgs["main"])
        w, h = self.size
        iw, ih = self.inset
        pad = 24
        ox, oy = w - iw - pad, h - ih - pad
        img.paste(Image.fromarray(imgs["tool"]), (ox, oy))
        dr = ImageDraw.Draw(img)
        dr.rectangle([ox - 2, oy - 2, ox + iw + 1, oy + ih + 1], outline=(235, 235, 235), width=2)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 28)
        except OSError:
            font = ImageFont.load_default()
        dr.text((pad, pad - 6), f"t = {t:5.2f} s", fill=(255, 255, 255), font=font)
        dr.text((ox, oy - 30), "end-effector close-up", fill=(255, 255, 255), font=font.font_variant(size=20)
                if hasattr(font, "font_variant") else font)
        self._writer.append_data(np.asarray(img))
        self.frames += 1
        self.next_frame_t = self.frames / self.fps

    def close_video(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
