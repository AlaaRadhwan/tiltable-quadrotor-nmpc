"""Static CAD-visual demonstration of the tiltable quadrotor (no dynamics, no prop spin).

Interactive:   tools/python scripts/view_cad_visual.py
Offscreen PNG: tools/python scripts/view_cad_visual.py --render artifacts/cad_visual_renders
Optional MP4:  add --video

Only qpos of the four existing tilt joints is set; mj_forward is used (no stepping).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "assets/cad_visual/tiltable_quadrotor_visual.xml"
LOOKAT = np.array([0.0, 0.0, 1.0])

VIEWS = {  # name: (azimuth, elevation, distance)
    "front": (180, -8, 1.05), "side": (90, -8, 1.05), "top": (90, -89, 1.1),
    "iso_front_left": (135, -25, 1.0), "closeup_rotor1": (150, -20, 0.32), "iso_rear_right": (-45, -25, 1.0), "low_under": (150, 25, 1.0),
}
POSES = {  # name: tilt angles (deg) for tilt_1..4
    "tilt0": (0, 0, 0, 0), "all_+30": (30, 30, 30, 30), "all_+60": (60, 60, 60, 60),
    "all_-45": (-45, -45, -45, -45), "all_90": (90, 90, 90, 90), "mixed": (45, -45, 90, -90),
    "rotor1_only_60": (60, 0, 0, 0), "cartwheel_like": (180, 90, 0, -90),
}


def make_model():
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"tilt_{i}") for i in range(1, 5)]
    return model, data, [int(model.jnt_qposadr[j]) for j in ids]


def set_tilt(model, data, adr, deg):
    for a, d in zip(adr, deg):
        data.qpos[a] = math.radians(d)
    mujoco.mj_forward(model, data)


def make_cam(view):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = LOOKAT
    if view is VIEWS["closeup_rotor1"]:
        cam.lookat[:] = LOOKAT + np.array([0.275, 0.0, 0.0])
    cam.azimuth, cam.elevation, cam.distance = view
    return cam


def render(out_dir: Path, video: bool):
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    model, data, adr = make_model()
    W, H = 960, 720
    r = mujoco.Renderer(model, H, W)
    opt = mujoco.MjvOption()
    sheet_tiles = []
    for pname, deg in POSES.items():
        set_tilt(model, data, adr, deg)
        row = []
        for vname, view in VIEWS.items():
            r.update_scene(data, camera=make_cam(view), scene_option=opt)
            img = Image.fromarray(r.render())
            if pname in ("tilt0", "all_+60", "mixed") or vname == "iso_front_left":
                img.save(out_dir / f"{pname}__{vname}.png")
            row.append(img)
        sheet_tiles.append((pname, row))
    # contact sheet: poses (rows) x selected views (cols)
    cols = ["front", "top", "iso_front_left", "closeup_rotor1"]
    ci = [list(VIEWS).index(c) for c in cols]
    tw, th = 640, 480
    sheet = Image.new("RGB", (tw * len(cols), th * len(sheet_tiles)), "white")
    for ri, (pname, row) in enumerate(sheet_tiles):
        for cj, k in enumerate(ci):
            t = row[k].resize((tw, th))
            ImageDraw.Draw(t).text((8, 6), f"{pname} / {cols[cj]}", fill=(255, 255, 255))
            sheet.paste(t, (cj * tw, ri * th))
    sheet.save(out_dir / "contact_sheet_tilt_poses.png")
    print("wrote", out_dir / "contact_sheet_tilt_poses.png")
    if video:
        import imageio.v2 as imageio

        frames, n = [], 360
        for k in range(n):
            t = k / n
            deg = [90 * math.sin(2 * math.pi * (t * 2 + i / 4)) for i in range(4)]
            set_tilt(model, data, adr, deg)
            r.update_scene(data, camera=make_cam((360 * t, -22, 1.5)), scene_option=opt)
            frames.append(r.render())
        imageio.mimsave(out_dir / "tilt_orbit.mp4", frames, fps=30)
        print("wrote", out_dir / "tilt_orbit.mp4")


def interactive():
    import mujoco.viewer

    model, data, adr = make_model()
    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = LOOKAT
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 135, -25, 1.5
        t0 = time.time()
        while v.is_running():
            t = time.time() - t0
            # slow orbit + each tilt joint sweeps +-90 deg with a quarter-period phase offset
            v.cam.azimuth = 135 + 12 * t
            set_tilt(model, data, adr, [90 * math.sin(2 * math.pi * (t / 8 + i / 4)) for i in range(4)])
            v.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", metavar="DIR", help="render PNGs offscreen instead of opening the viewer")
    ap.add_argument("--video", action="store_true", help="with --render: also write tilt_orbit.mp4")
    args = ap.parse_args()
    render(Path(args.render), args.video) if args.render else interactive()
