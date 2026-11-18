# CAD visualization of the NMPC maneuvers

Display and video rendering of the closed-loop NMPC experiments with the Onshape CAD model and
spinning propellers. **Visualization only**: it never feeds back into the simulation or controller.

## Architecture

```
physics + NMPC (unchanged)                 display only
--------------------------                 ---------------------------------------------
rigid plant MJCF (no tilt joints)  --state-->  CadManeuverVisual  (visualization/cad_maneuver.py)
EffectorNmpcController                           - separate MjModel/MjData from the CAD MJCF
  command.thrust, command.alpha_cmd              - qpos = [p, q_WB, alpha_1..4] (mirrored, never stepped)
                                                 - PropellerSpinAnimator (visualization/propeller_spin.py)
                                                 - target + tool-point markers, fixed camera, MP4 writer
```

* `assets/cad_visual/tiltable_quadrotor_visual.xml` – CAD visual MJCF (meshes in `meshes/`, generated
  from Onshape by `tools/cad_visual/`). The original plant MJCF is untouched. Its ground plane is a
  500 m visual-only plane (same color) so no edge is visible.
* Propellers are scene-only geoms appended to the render scene; their speed follows
  `sqrt(commanded thrust)` (a deliberately slow *visual* scale) and the angle integrates the 2 ms
  **simulation** step, so animation is tied to simulation time, never wall time.
* Both runners (`experiments/nmpc/run_6dof_effector_benchmark.py`, `experiments/nmpc/run_paper_maneuver.py`)
  share `CadManeuverVisual`; there is one visualization system.
* Camera: fixed (lookat `[0, 0, 1.45]`, distance 2.0 m, azimuth 135/140 deg, elevation -15 deg); a small
  end-effector close-up inset follows the tool point in MP4s. Orange sphere = controller's mathematical
  tool point, green sphere = true target.

### CAD tip vs controller tool point
The controller tool point is body `[0, 0, 0.15]`. The CAD end-effector mesh ends at body z = 0.1454, i.e. its
tip is **4.6 mm** short of the mathematical point (no lateral offset; 0.3 % of the tool offset, ~3 px at
1080p). This is negligible and is left as is; the control model, tool offset and references are unchanged.
The orange marker is drawn at the true mathematical point. `scripts/verify_cad_visual_mapping.py` prints this.

## Commands

```bash
# 6-DoF end-effector benchmark
tools/python experiments/nmpc/run_6dof_effector_benchmark.py --cad-visual                    # interactive
MUJOCO_GL=glfw tools/python experiments/nmpc/run_6dof_effector_benchmark.py \
    --render-mp4 artifacts/cad_visual_renders/nmpc_6dof_maneuver.mp4                         # 1920x1080, 30 fps

# Paper maneuvers (same flags: --cad-visual, --render-mp4 PATH, --headless, --video-fps, --video-size)
R=experiments/nmpc/run_paper_maneuver.py; O=artifacts/cad_visual_renders/paper_maneuvers
MUJOCO_GL=glfw tools/python $R --maneuver pitch --bias-compensation --render-mp4 $O/pitch_360.mp4
MUJOCO_GL=glfw tools/python $R --maneuver roll  --bias-compensation --render-mp4 $O/roll_360.mp4
MUJOCO_GL=glfw tools/python $R --maneuver cartwheel --predictive-preposition --render-mp4 $O/cartwheel.mp4
MUJOCO_GL=glfw tools/python $R --maneuver cartwheel --render-mp4 $O/cartwheel_baseline.mp4
MUJOCO_GL=glfw tools/python $R --maneuver yaw --render-mp4 $O/yaw_360.mp4
tools/python $R --maneuver yaw --cad-visual                                                   # interactive
```

`--maneuver yaw` (new) is one level 360 deg revolution about body/tool Z at the pitch/roll peak rate
(36 deg/s). It differs from `cartwheel`, which starts at +90 deg roll (tool horizontal) and rotates about
the same body Z axis, so the vehicle spins in a vertical plane. Existing flags are unchanged.

## Combined continuous maneuver (one run, one MP4)

`--maneuver combined` runs a single closed-loop simulation (no resets): level hover, validated 360 deg roll,
validated 360 deg pitch, pitch +90 deg so the tool axis (body +Z) points along world +X, a 360 deg spin about
the tool axis, pitch back to level, hover. The tool point stays at `[0, 0, 1.5]`.
`R(t) = Rx(roll) Ry(pitch) Ry(tilt) Rz(cartwheel)`, one angle moving at a time, all continuous scalars (the
cartwheel ends at the negated quaternion, no sign flip). Roll/pitch use the validated schedule
(0 -> +180, hold, +180 -> -180 = the 360 deg revolution, hold, -180 -> 0) at 36 deg/s with optional
`--bias-compensation`; the cartwheel uses the validated 12 deg/s profile. A single 0 -> 360 deg sweep through
the inverted attitude was tried and does not track (attitude error > 100 deg), so it is not used.

```bash
MUJOCO_GL=glfw tools/python scripts/compare_cad_visual_equivalence.py --suite combined   # A/B headless + render, writes
                                                                                         # artifacts/cad_visual_renders/combined_360_maneuvers.mp4
MUJOCO_GL=glfw tools/python experiments/nmpc/run_paper_maneuver.py --maneuver combined --bias-compensation \
    --render-mp4 artifacts/cad_visual_renders/combined_360_maneuvers.mp4
tools/python experiments/nmpc/run_paper_maneuver.py --maneuver combined --bias-compensation --cad-visual   # live
tools/python scripts/analyze_combined_maneuver.py [CSV]   # per-phase errors, rotations, continuity, saturation
```
The combined run uses its own fixed main camera (`COMBINED_MAIN_CAMERA` in the runner) and a larger
end-effector inset; other maneuvers keep the default camera.

## Verification

```bash
MUJOCO_GL=glfw tools/python scripts/compare_cad_visual_equivalence.py --suite 6dof    # original x2 vs CAD render
MUJOCO_GL=glfw tools/python scripts/compare_cad_visual_equivalence.py --suite paper   # all paper maneuvers
tools/python scripts/verify_cad_visual_mapping.py     # tilt sign/position mapping + CAD tip vs tool point
tools/python scripts/verify_cad_viewer.py             # interactive viewer smoke test (needs a display)
tools/python tools/cad_visual/verify_visual_model.py; tools/python tools/cad_visual/verify_propeller_animation.py
```

The comparison runs each experiment headless twice (noise floor) and once rendering, and diffs every
simulated-dynamics column of the CSV logs. Controller/solver timing and deadline flags are wall-clock
quantities and are reported separately: rendering consumes CPU, so deadline misses can rise even though
the simulated trajectory is bit-identical. Generated MP4s/CSVs are git-ignored.

## Public and detailed variants

The default `assets/cad_visual/tiltable_quadrotor_visual.xml` is the **public** variant: the frame, arms, mounts and
end-effector are meshes of the author's own design, while the propeller, BLDC motor and servo are original generic
MuJoCo primitives (cylinders, boxes, an ellipsoid blade) generated by `tools/cad_visual/build_visual_mjcf.py`. It loads
with no third-party files. A **detailed** variant that uses the downloaded component models is optional and not
distributed; see `THIRD_PARTY.md` and `tools/cad_visual/install_detailed_cad.py`.

## Regenerating the CAD assets
Two separate workflows:

**A. CAD asset refresh (author only).** The meshes, `cad_placements.json` and `propeller_spin_axes.json` were exported
from the author's private Onshape assembly. The export scripts are not included in this repository; the exported
files are kept as data.

**B. Local visual MJCF regeneration (offline, deterministic, no credentials).**
```bash
tools/python tools/cad_visual/build_visual_mjcf.py                    # public variant; rewrites assets/cad_visual/tiltable_quadrotor_visual.xml
tools/python tools/cad_visual/check_visual_mjcf_reproducible.py       # read-only check, exit 1 if inconsistent
```
The check generates twice into a temp dir, requires identical output, compares byte-for-byte with the checked-in
XML (never overwriting it) and confirms the visual settings (50 visual geoms, 500 m ground, `<statistic>`, `<map>`,
1920x1080 offscreen, headlight, skybox, 9 hidden placeholders). It is also covered by
`tests/test_cad_visual.py::test_visual_mjcf_regeneration_is_reproducible`.

Authoritative sources (hand-maintained): `src/tiltable_quadrotor_rl/robot/mjcf/tiltable_quadrotor.xml` (physical
MJCF, with tilt joints; *not* `tiltable_quadrotor_rigid.xml`), `assets/cad_visual/cad_placements.json`,
`assets/cad_visual/meshes/`, and the visual constants at the top of `build_visual_mjcf.py` (palette, ground size/colour,
`<statistic>`, `<map>`, headlight, skybox). **Generated, do not hand-edit:**
`assets/cad_visual/tiltable_quadrotor_visual.xml` - change the sources or the generator and regenerate. The
generator uses whitespace/comment-tolerant tag-level edits, validates the source (bodies `base`, `rotor_1..4`,
joints `tilt_1..4`, placeholder geoms, mesh files, placement fields) and fails with a clear error otherwise.
