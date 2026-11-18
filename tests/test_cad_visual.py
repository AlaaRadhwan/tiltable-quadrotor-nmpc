from __future__ import annotations

import mujoco
import numpy as np

from tiltable_quadrotor_rl.control.nmpc_effector_paper_maneuvers import (
    PaperStyleFixedEffectorTrajectory,
)
from tiltable_quadrotor_rl.visualization.cad_maneuver import (
    CadManeuverVisual,
)
from tiltable_quadrotor_rl.visualization.servo_overlay import (
    ROTOR_OFFSETS_B,
    ROTOR_TILT_DIRS_B,
)


def test_cad_tilt_convention_matches_plant():
    visual = CadManeuverVisual()
    state = np.zeros(17)
    state[2], state[6] = 1.0, 1.0
    state[13:17] = np.radians([25.0, -40.0, 10.0, 60.0])
    visual.set_state(state, target_W=np.zeros(3), effector_W=np.zeros(3))

    for i in range(4):
        body = mujoco.mj_name2id(
            visual.model, mujoco.mjtObj.mjOBJ_BODY, f"rotor_{i + 1}"
        )
        axis = visual.data.xmat[body].reshape(3, 3)[:, 2]
        expected = (
            np.sin(state[13 + i]) * ROTOR_TILT_DIRS_B[i]
            + np.cos(state[13 + i]) * np.array([0.0, 0.0, 1.0])
        )
        assert np.allclose(axis, expected, atol=1e-6)
        assert np.allclose(
            visual.data.xpos[body] - state[0:3], ROTOR_OFFSETS_B[i], atol=1e-6
        )


def test_propeller_angles_follow_simulation_time():
    visual = CadManeuverVisual()
    visual.advance_props(0.002, np.full(4, 7.4531475))
    first = visual.anim.angles.copy()
    visual.advance_props(0.002, np.full(4, 7.4531475))
    step = (visual.anim.angles - first) % (2 * np.pi)
    assert np.allclose(np.abs(step), np.abs((first) % (2 * np.pi)), atol=1e-9)


def test_yaw_maneuver_is_single_360_degree_body_z_revolution():
    trajectory = PaperStyleFixedEffectorTrajectory(
        position_W=np.array([0.0, 0.0, 1.5]), maneuver="yaw"
    )
    start = trajectory.sample(0.0)
    end = trajectory.sample(trajectory.end_time_s)
    assert np.allclose(start.quaternion_WE, [1, 0, 0, 0])
    assert abs(abs(end.quaternion_WE[0]) - 1.0) < 1e-9

    mid = trajectory.sample(0.5 * (2.0 + (trajectory.end_time_s - 3.0)))
    assert np.allclose(mid.omega_E[:2], 0.0)
    assert np.allclose(mid.position_W, [0.0, 0.0, 1.5])


def test_combined_trajectory_signs_continuity_and_tool_axis():
    from tiltable_quadrotor_rl.control.end_effector import (
        quaternion_to_rotation_np,
    )
    from tiltable_quadrotor_rl.control.nmpc_effector_paper_maneuvers import (
        PaperStyleCombinedManeuverTrajectory,
    )

    trajectory = PaperStyleCombinedManeuverTrajectory(
        position_W=np.array([0.0, 0.0, 1.5])
    )
    phases = {name: (t0, t1) for name, t0, t1 in trajectory.phases}

    times = np.arange(0.0, trajectory.end_time_s, 0.01)
    quaternions = np.array(
        [trajectory.sample(t).quaternion_WE for t in times]
    )
    assert np.all(np.sum(quaternions[1:] * quaternions[:-1], axis=1) > 0.0)
    assert np.max(np.linalg.norm(np.diff(quaternions, axis=0), axis=1)) < 0.01
    assert np.allclose(quaternions[0], [1, 0, 0, 0])
    assert np.allclose(quaternions[-1], [-1, 0, 0, 0], atol=1e-9)  # 360 deg cartwheel

    # Ry(+90 deg) maps the tool axis (body +Z) onto world +X ...
    t0, t1 = phases["hold_after_pitch_to_horizontal"]
    axis = quaternion_to_rotation_np(trajectory.sample(0.5 * (t0 + t1)).quaternion_WE)[:, 2]
    assert np.allclose(axis, [1.0, 0.0, 0.0], atol=1e-9)

    # ... and stays there for the whole spin about the tool axis.
    t0, t1 = phases["cartwheel_360"]
    for t in np.linspace(t0, t1, 25):
        sample = trajectory.sample(t)
        axis = quaternion_to_rotation_np(sample.quaternion_WE)[:, 2]
        assert np.allclose(axis, [1.0, 0.0, 0.0], atol=1e-9)
        assert np.allclose(sample.omega_E[:2], 0.0)
    assert np.allclose(trajectory.sample(0.5 * (t0 + t1)).position_W, [0.0, 0.0, 1.5])


def test_visual_mjcf_regeneration_is_reproducible():
    """Offline generator reproduces the checked-in visual MJCF byte-for-byte and fails on a bad source."""
    import sys
    import tempfile
    from pathlib import Path

    tools = Path(__file__).resolve().parents[1] / "tools" / "cad_visual"
    sys.path.insert(0, str(tools))
    import build_visual_mjcf as B

    text, _ = B.build()
    assert text == B.DEFAULT_OUT.read_text()
    assert B.build()[0] == text  # deterministic
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.xml"
        bad.write_text(B.DEFAULT_SRC.read_text().replace('name="rotor_3"', 'name="rotor_x"', 1))
        try:
            B.build(src_xml=bad)
        except B.GenError:
            pass
        else:
            raise AssertionError("missing rotor body must raise GenError")


def test_public_visual_model_uses_no_restricted_meshes_and_loads():
    """The default visualization must load without any third-party mesh and reference none."""
    import sys
    import xml.etree.ElementTree as ET
    from pathlib import Path

    tools = Path(__file__).resolve().parents[1] / "tools" / "cad_visual"
    sys.path.insert(0, str(tools))
    import build_visual_mjcf as B

    root = ET.fromstring(B.DEFAULT_OUT.read_text())
    used = {Path(m.get("file")).stem for m in root.iter("mesh")}
    assert used and not (used & B.THIRD_PARTY_MESHES)
    for name in B.THIRD_PARTY_MESHES:
        assert not (B.DEFAULT_MESHES / f"{name}.stl").exists()
    model = mujoco.MjModel.from_xml_path(str(B.DEFAULT_OUT))
    for i in range(1, 5):  # spinnable propeller geoms exist in the hidden group for every rotor
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"rotor_{i}")
        assert any(model.geom_bodyid[g] == body and model.geom_group[g] == 3 for g in range(model.ngeom))
