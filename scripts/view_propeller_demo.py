"""Phase 2 demo: animated CAD propellers on the visual MJCF.

Interactive viewer (loops forever):  tools/python scripts/view_propeller_demo.py
Offscreen MP4:                       MUJOCO_GL=glfw tools/python scripts/view_propeller_demo.py --render artifacts/cad_visual_renders/propeller_demo.mp4

The scripted timeline (all in *simulation* time, fixed frame step => deterministic):
  A  0- 5 s  hover under real physics (mj_step, equal hover thrust on all four rotors)
  B  5-11 s  per-rotor thrust sweeps: visual speed follows sqrt(thrust)
  C 11-19 s  tilt joints swept through +-90 deg at hover thrust (close-up first, then wide)
  D 19-29 s  whole aircraft rolls, then pitches+yaws, with tilted rotors
Propeller animation is scene-only (tiltable_quadrotor_rl.visualization.propeller_spin);
segments B-D set qpos kinematically, they never alter the model or dynamics.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from tiltable_quadrotor_rl.visualization.propeller_spin import PropellerSpinAnimator, SpinConfig

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "assets/cad_visual/tiltable_quadrotor_visual.xml"
HOVER = 7.4531475
TOTAL = 29.0
SEGMENTS = [(0, 5, "A  Hover (real physics, equal thrust)"), (5, 11, "B  Thrust changes -> spin speed ~ sqrt(thrust)"),
            (11, 19, "C  Tilt joints sweep +-90 deg, props keep spinning"), (19, 29, "D  Whole-aircraft rotation, tilted rotors")]


def quat_from_euler(roll, pitch, yaw):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "xyz")
    return q


class Demo:
    def __init__(self, cfg: SpinConfig | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self.anim = PropellerSpinAnimator(self.model, cfg)
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.anim.reset()
        self.t = 0.0

    def step(self, dt: float):
        """Advance the demo by one frame of simulation time ``dt``."""
        m, d, t = self.model, self.data, self.t
        if t < 5.0:                                   # A: real physics
            d.ctrl[:4] = HOVER
            for _ in range(max(1, round(dt / m.opt.timestep))):
                mujoco.mj_step(m, d)
        else:                                          # B-D: kinematic poses, scripted thrust
            d.qvel[:] = 0
            d.qpos[0:3] = [0, 0, 1]
            quat, tilt, f = np.array([1.0, 0, 0, 0]), np.zeros(4), np.full(4, HOVER)
            if t < 11.0:                               # B
                f = HOVER * (1 + 0.9 * np.sin(2 * math.pi * (t - 5) / 6 + np.arange(4) * math.pi / 2))
            elif t < 19.0:                             # C
                tilt = np.radians(90) * np.sin(2 * math.pi * (t - 11) / 8 + np.arange(4) * math.pi / 2)
            else:                                      # D
                u = t - 19.0
                tilt = np.radians(30) * np.array([1, -1, 1, -1.0])
                quat = quat_from_euler(2 * math.pi * u / 5, 0, 0) if u < 5 else \
                    quat_from_euler(0, 2 * math.pi * (u - 5) / 5, 2 * math.pi * (u - 5) / 5)
            d.qpos[3:7] = quat
            d.qpos[7:11] = tilt
            d.ctrl[:4] = f
            mujoco.mj_forward(m, d)
        self.anim.advance_from_data(dt, d)
        self.t += dt

    def camera(self, cam: mujoco.MjvCamera):
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0, 0, 1]
        cam.azimuth, cam.elevation, cam.distance = 135 + 10 * self.t, -25, 1.0
        if 11 <= self.t < 15:                          # close-up on rotor 1 while it tilts
            cam.lookat[:] = [0.275, 0, 1]
            cam.azimuth, cam.elevation, cam.distance = 150 + 6 * (self.t - 11), -18, 0.34
        elif 0 <= self.t < 2.5:                        # low close-up of rotor 1 spinning in hover
            cam.lookat[:] = [0.275, 0, 1]
            cam.azimuth, cam.elevation, cam.distance = 150 + 6 * self.t, -18, 0.34

    def caption(self):
        seg = next(n for a, b, n in SEGMENTS if a <= min(self.t, TOTAL - 1e-6) < b)
        w = self.anim.spin_sign * self.anim.speeds
        lines = [f"t={self.t:5.2f}s   {seg}",
                 "rotor spin (rad/s, + = CCW from above): " + "  ".join(f"R{i + 1}:{w[i]:+6.1f}" for i in range(4)),
                 "thrust (N):                             " + "  ".join(f"R{i + 1}:{self.data.ctrl[i]:6.2f}" for i in range(4))]
        return lines


def render(out: Path, fps: int, size, cfg):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    demo = Demo(cfg)
    r = mujoco.Renderer(demo.model, size[1], size[0])
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    out.parent.mkdir(parents=True, exist_ok=True)
    dt, frames = 1.0 / fps, []
    while demo.t < TOTAL - 1e-9:
        demo.step(dt)
        demo.camera(cam)
        r.update_scene(demo.data, camera=cam, scene_option=opt)
        demo.anim.add_to_scene(demo.data, r.scene)
        img = Image.fromarray(r.render())
        d = ImageDraw.Draw(img)
        for k, line in enumerate(demo.caption()):
            d.text((14, 12 + 16 * k), line, fill=(255, 255, 255))
        frames.append(np.asarray(img))
    imageio.mimsave(out, frames, fps=fps, macro_block_size=1)
    print(f"wrote {out} ({len(frames)} frames, {len(frames) / fps:.1f} s, {out.stat().st_size / 1e6:.1f} MB)")


def interactive(cfg):
    import mujoco.viewer

    demo = Demo(cfg)
    dt = 1.0 / 60.0
    with mujoco.viewer.launch_passive(demo.model, demo.data) as v:
        while v.is_running():
            t0 = time.time()
            if demo.t >= TOTAL:
                demo.reset()
            demo.step(dt)                                  # sim time advances by a fixed dt, not wall time
            demo.camera(v.cam)
            v.user_scn.ngeom = 0
            demo.anim.add_to_scene(demo.data, v.user_scn)
            v.sync()
            time.sleep(max(0.0, dt - (time.time() - t0)))   # wall-clock only throttles playback


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", metavar="MP4", help="write an offscreen MP4 instead of opening the viewer")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    ap.add_argument("--hover-speed", type=float, default=SpinConfig.hover_speed, help="visual rad/s at hover thrust")
    ap.add_argument("--speed-scale", type=float, default=1.0)
    ap.add_argument("--no-blur", action="store_true")
    a = ap.parse_args()
    cfg = SpinConfig(hover_speed=a.hover_speed, scale=a.speed_scale, blur_ghosts=0 if a.no_blur else SpinConfig.blur_ghosts)
    render(Path(a.render), a.fps, a.size, cfg) if a.render else interactive(cfg)
