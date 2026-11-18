"""Smoke-test the interactive CAD viewer path (passive viewer + scene-injected propellers).

    tools/python scripts/verify_cad_viewer.py

Opens the real passive viewer on the CAD visual model, drives it with synthetic tilt/thrust for a
few seconds through the same CadManeuverVisual calls the benchmarks use, and checks that the
propeller geoms, ghosts and markers are present in ``viewer.user_scn`` and that the props spin.
Needs a display.
"""
from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from tiltable_quadrotor_rl.visualization.cad_maneuver import CadManeuverVisual

vis = CadManeuverVisual()
state = np.zeros(17)
state[2], state[6] = 1.5, 1.0
hover = np.full(4, 7.4531475)
angles0 = vis.anim.angles.copy()
counts = []
with mujoco.viewer.launch_passive(vis.model, vis.data) as v:
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.0, 135.0, -15.0
    v.cam.lookat[:] = [0.0, 0.0, 1.45]
    t0 = time.time()
    n = 0
    while v.is_running() and time.time() - t0 < 4.0:
        t = time.time() - t0
        state[13:17] = np.radians(40) * np.sin(2 * np.pi * 0.5 * t + np.arange(4))
        with v.lock():
            vis.set_state(state, target_W=[0, 0, 1.5], effector_W=[0, 0, 1.5])
            v.user_scn.ngeom = 0
            vis.advance_props(0.02, hover)
            vis.decorate(v.user_scn)
            counts.append(v.user_scn.ngeom)
        v.sync()
        n += 1
        time.sleep(0.02)
moved = np.abs(np.angle(np.exp(1j * (vis.anim.angles - angles0)))).max()
print(f"frames={n}, user_scn geoms/frame min={min(counts)} max={max(counts)} (maxgeom={v.user_scn.maxgeom})")
print(f"propeller angles advanced by up to {moved:.2f} rad; viewer ran without error")
assert min(counts) >= 4 * 2 + 2 and moved > 0.1
print("OK")
