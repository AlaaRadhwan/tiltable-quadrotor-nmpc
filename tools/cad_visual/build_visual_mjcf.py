"""Regenerate assets/cad_visual/tiltable_quadrotor_visual.xml (workflow B: local, offline, deterministic).

Two variants:
  public    (default) the redistributable visualization. The 14 meshes derived from the three
            third-party downloads (see THIRD_PARTY.md) are NOT used; original generic MuJoCo
            primitives written in this file stand in for the propeller, motor and servo.
  detailed  optional, for users who have obtained the third-party models themselves and placed
            the converted meshes in assets/cad_visual/meshes/third_party/ (see THIRD_PARTY.md).
            Never committed; writes assets/cad_visual/tiltable_quadrotor_visual_detailed.xml.

Inputs (all committed; no Onshape access or credentials needed):
  * src/tiltable_quadrotor_rl/robot/mjcf/tiltable_quadrotor.xml   physical MJCF (with tilt joints)
  * assets/cad_visual/cad_placements.json                         CAD part placements (74)
  * assets/cad_visual/meshes/*.stl                                CAD meshes
Run:    tools/python tools/cad_visual/build_visual_mjcf.py            (overwrites the checked-in XML)
        tools/python tools/cad_visual/build_visual_mjcf.py --variant detailed
        tools/python tools/cad_visual/build_visual_mjcf.py --out /tmp/v.xml
Check:  tools/python tools/cad_visual/check_visual_mjcf_reproducible.py

Only visual material is added to the original text: mesh/material assets, non-colliding
mesh geoms (contype=conaffinity=0, group 2), and the original primitive placeholder geoms
are moved to hidden group 4.  Bodies, inertials, joints, sites, actuators, sensors and
the keyframe are copied verbatim.

Coordinate alignment: CAD assembly -> MuJoCo base frame is a pure translation by
(0, 0, -TILT_AXIS_Z), which puts the CAD tilt axes (z = 24.612 mm) on the MJCF rotor-frame
origin.  Rotor-group geoms are then expressed in the rotor body frame
(inverse of translate(c_i) * Rz(theta_i)), so they follow the existing tilt joints.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "src/tiltable_quadrotor_rl/robot/mjcf/tiltable_quadrotor.xml"
DEFAULT_PLACEMENTS = ROOT / "assets/cad_visual/cad_placements.json"
DEFAULT_OUT = ROOT / "assets/cad_visual/tiltable_quadrotor_visual.xml"
DEFAULT_MESHES = ROOT / "assets/cad_visual/meshes"
DETAILED_OUT = ROOT / "assets/cad_visual/tiltable_quadrotor_visual_detailed.xml"
THIRD_PARTY_SUBDIR = "third_party"   # under meshes/: optional, user-supplied, never committed

# Meshes derived from the three downloaded third-party models. Excluded from the public variant.
THIRD_PARTY_MESHES = frozenset({
    "prop",                                                                    # APC 8x8E propeller
    "motor_partbody", "motor_corpo2", "motor_corpo3", "motor_corpo62",
    "motor_plate", "motor_shaft",                                              # T-Motor AT2814
    "servo_case_b", "servo_case_f", "servo_case_m", "servo_horn",
    "servo_horn_idle", "servo_idle_cap", "servo_connector",                    # ROBOTIS servo
})

# Visual-only presentation settings applied on top of the physical MJCF.
GROUND_SIZE = "500 500 0.1"           # enlarged visual ground plane
GROUND_RGBA = "0.38 0.41 0.45 1"      # visual-only ground tone
STATISTIC = ('    <!-- The 500 m visual ground plane inflates the auto-computed model extent, which would push the\n'
             '         near clip plane past close-up cameras. Fix the extent and make znear tight instead. -->\n'
             '    <statistic extent="10" center="0 0 1.4"/>\n')
MAP = '        <map znear="0.001" zfar="60"/>\n'
HEADLIGHT = '        <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>\n'
SKYBOX = '        <texture type="skybox" builtin="gradient" rgb1="0.45 0.62 0.80" rgb2="0.05 0.06 0.09" width="512" height="3072"/>'
CAD_DEFAULT = ('        <!-- CAD visual meshes: never collide -->\n'
               '        <default class="cad">\n            <geom group="2" contype="0" conaffinity="0" density="0"/>\n'
               '        </default>\n\n')
PLACEHOLDERS = [f"base_visual"] + [f"arm_{i}_visual" for i in range(1, 5)] + [f"rotor_{i}_visual" for i in range(1, 5)]

TILT_AXIS_Z = 0.024612  # m, measured CAD tilt-axis height (Revolute 1 mate connector)
ROTORS = {  # MJCF rotor body -> (position in base frame, yaw of body frame)
    1: (np.array([0.275, 0.0, 0.0]), 0.0),
    2: (np.array([0.0, 0.275, 0.0]), np.pi / 2),
    3: (np.array([-0.275, 0.0, 0.0]), np.pi),
    4: (np.array([0.0, -0.275, 0.0]), -np.pi / 2),
}
# curated palette (Onshape part colors are the generic default blue/grey)
RGBA = {
    "base": (0.12, 0.12, 0.13, 1), "arm": (0.62, 0.64, 0.67, 1), "end_effector": (0.23, 0.38, 0.71, 1),
    "servo_mount": (0.10, 0.10, 0.11, 1), "bldc_mount": (0.10, 0.10, 0.11, 1),
    "servo_case": (0.16, 0.17, 0.19, 1), "servo_horn": (0.75, 0.77, 0.80, 1),
    "servo_idle_cap": (0.97, 0.53, 0.0, 1), "servo_connector": (0.85, 0.87, 0.82, 1),
    "motor_dark": (0.15, 0.15, 0.16, 1), "motor_black": (0.03, 0.03, 0.03, 1),
    "motor_metal": (0.70, 0.72, 0.75, 1), "prop": (0.07, 0.07, 0.08, 1),
}


def material_of(mesh: str) -> str:
    if mesh.startswith("arm_"): return "arm"
    if mesh.startswith("servo_case"): return "servo_case"
    if mesh in ("servo_horn_idle",): return "servo_horn"
    if mesh in ("motor_partbody", "motor_corpo3"): return "motor_dark"
    if mesh == "motor_corpo2": return "motor_black"
    if mesh in ("motor_shaft", "motor_corpo62", "motor_plate"): return "motor_metal"
    return mesh


def rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def mat_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        q = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    else:  # 180-degree rotation fallback
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1) * 2
        v = np.zeros(4); v[0] = (R[k, j] - R[j, k]) / s; v[1 + i] = s / 4
        v[1 + j] = (R[j, i] + R[i, j]) / s; v[1 + k] = (R[k, i] + R[i, k]) / s
        q = v
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def fmt(x):
    x = 0.0 if abs(x) < 5e-10 else x
    return f"{x:.9g}"


SPINNING_MESHES = ("prop", "motor_shaft")  # drawn by the scene-only propeller animator


def geom_xml(name, mesh, T, indent):
    pos, quat = T[:3, 3], mat_to_quat(T[:3, :3])
    grp = ' group="3"' if mesh in SPINNING_MESHES else ""  # hidden by default; animator draws spun copies
    return (f'{indent}<geom name="{name}" class="cad" type="mesh" mesh="{mesh}" material="{material_of(mesh)}"{grp} '
            f'pos="{" ".join(fmt(v) for v in pos)}" quat="{" ".join(fmt(v) for v in quat)}"/>')


# --------------------------------------------------------------------------------------------
# Original generic stand-ins for the third-party components (public variant).
# Round-number dimensions chosen for this project; they are NOT copied or fitted to the
# restricted meshes. Positions follow the author's own assembly layout (rotor frame / arm axis).
# --------------------------------------------------------------------------------------------
SERVO_CENTER_X = 0.2362    # servo body centre along the arm (m), base frame, arm 1 (+X)
PROP_Z = -0.0384           # propeller plane in the rotor frame (the CAD spin mate, propeller_spin_axes.json)
GENERIC_CLASS = "cad"
GENERIC_MATERIALS = ("servo_case", "servo_horn", "servo_idle_cap", "servo_connector", "motor_dark", "motor_metal", "motor_black", "prop")


def _v(*xs):
    return " ".join(fmt(x) for x in xs)


def _cyl(name, p0, p1, r, material, group=None):
    g = f' group="{group}"' if group is not None else ""
    return f'<geom name="{name}" class="{GENERIC_CLASS}" type="cylinder" material="{material}"{g} fromto="{_v(*p0, *p1)}" size="{fmt(r)}"/>'


def _box(name, pos, half, material, quat=(1, 0, 0, 0)):
    return (f'<geom name="{name}" class="{GENERIC_CLASS}" type="box" material="{material}" '
            f'pos="{_v(*pos)}" quat="{_v(*quat)}" size="{_v(*half)}"/>')


def generic_chassis_geoms(indent):
    """Servo body, idle-side cap/horn and connector for each of the four arms (base frame)."""
    out = []
    for k in range(4):
        R = rotz(k * np.pi / 2)
        quat = mat_to_quat(R)
        pt = lambda x, y=0.0, z=0.0: R @ np.array([x, y, z])
        n = k + 1
        out.append(_box(f"chassis_servo_case_{n}", pt(SERVO_CENTER_X), (0.0100, 0.0085, 0.0130), "servo_case", quat))
        out.append(_cyl(f"chassis_servo_horn_idle_{n}", pt(0.2226), pt(0.2240), 0.0095, "servo_horn"))
        out.append(_cyl(f"chassis_servo_idle_cap_{n}", pt(0.2212), pt(0.2226), 0.0045, "servo_idle_cap"))
        out.append(_box(f"chassis_servo_connector_{n}", pt(SERVO_CENTER_X, 0.0, -0.0140), (0.0040, 0.0075, 0.0015), "servo_connector", quat))
    return [indent + g for g in out]


def generic_rotor_geoms(i, indent):
    """Servo horn, BLDC motor, shaft and two-blade propeller in the rotor frame (follows the tilt joint).

    The propeller blade and hub are in hidden group 3: PropellerSpinAnimator draws spun copies.
    """
    n = f"rotor_{i}"
    out = [
        _cyl(f"{n}_servo_horn", (-0.0270, 0, 0), (-0.0240, 0, 0), 0.0105, "servo_horn"),
        _cyl(f"{n}_motor_stator", (0, 0, -0.0130), (0, 0, 0.0200), 0.0165, "motor_dark"),
        _cyl(f"{n}_motor_bell", (0, 0, -0.0230), (0, 0, -0.0130), 0.0190, "motor_black"),
        _cyl(f"{n}_motor_shaft", (0, 0, -0.0230), (0, 0, PROP_Z - 0.0030), 0.0040, "motor_metal", group=3),
        f'<geom name="{n}_prop" class="{GENERIC_CLASS}" type="ellipsoid" material="prop" group="3" '
        f'pos="{_v(0, 0, PROP_Z)}" size="{_v(0.0850, 0.0105, 0.0015)}"/>',
    ]
    return [indent + g for g in out]


class GenError(RuntimeError):
    pass


def need(cond, msg):
    if not cond:
        raise GenError(msg)


def tag_re(name):
    """Regex for a whole start tag `<name .../>` (attributes may span lines)."""
    return re.compile(rf"<{name}\b[^>]*?/?>", re.S)


def find_tag(txt, name, attr=None, value=None):
    for m in tag_re(name).finditer(txt):
        if attr is None or re.search(rf'\b{attr}\s*=\s*"{re.escape(value)}"', m.group(0)):
            return m
    raise GenError(f"<{name}{'' if attr is None else f' {attr}={value!r}'}> not found in source MJCF")


def set_attr(tag, attr, value, after=None):
    """Set/replace an attribute in a start-tag string (optionally inserting after another attribute)."""
    m = re.search(rf'(\b{attr}\s*=\s*)"[^"]*"', tag)
    if m:
        return tag[:m.start()] + f'{attr}="{value}"' + tag[m.end():]
    if after:
        a = re.search(rf'\b{after}\s*=\s*"[^"]*"', tag)
        if a:
            return tag[:a.end()] + f' {attr}="{value}"' + tag[a.end():]
    end = tag.rindex("/>") if tag.endswith("/>") else tag.rindex(">")
    return tag[:end].rstrip() + f' {attr}="{value}"' + tag[end:] if not tag[:end].endswith(" ") else tag[:end] + f'{attr}="{value}"' + tag[end:]


def replace_span(txt, m, new):
    return txt[:m.start()] + new + txt[m.end():]


def line_start(txt, pos):
    return txt.rfind("\n", 0, pos) + 1


def skip_leading_comments(txt, pos):
    """Move `pos` (a line start) back over directly preceding single-line XML comments."""
    while True:
        prev = line_start(txt, pos - 1) if pos > 0 else 0
        if re.fullmatch(r"[ \t]*<!--.*-->[ \t]*\n", txt[prev:pos]):
            pos = prev
        else:
            return pos


def validate_source(src_txt):
    try:
        root = ET.fromstring(src_txt)
    except ET.ParseError as e:
        raise GenError(f"source MJCF is not well-formed XML: {e}")
    need(root.tag == "mujoco", "source root is not <mujoco>")
    names = {e.get("name") for e in root.iter("body")}
    for b in ["base"] + [f"rotor_{i}" for i in ROTORS]:
        need(b in names, f"source MJCF has no <body name='{b}'>")
    joints = {e.get("name") for e in root.iter("joint")}
    for i in ROTORS:
        need(f"tilt_{i}" in joints, f"source MJCF has no <joint name='tilt_{i}'>")
    geoms = {e.get("name") for e in root.iter("geom")}
    for g in PLACEHOLDERS + ["ground_visual"]:
        need(g in geoms, f"source MJCF has no <geom name='{g}'>")
    need(root.find("statistic") is None and not any(e.tag == "map" for e in root.iter()),
         "source MJCF already defines <statistic>/<map>; refusing to guess how to merge")
    need(root.find("asset") is None, "source MJCF already has an <asset> block")
    return root


def load_placements(path, meshes_dir, variant="public"):
    data = json.loads(Path(path).read_text())
    need("placements" in data, f"{path}: missing 'placements'")
    plc = data["placements"]
    need(plc, f"{path}: empty placement list")
    valid = {"chassis", *(f"rotor_{i}" for i in ROTORS)}
    for k, p in enumerate(plc):
        for key in ("group", "mesh", "role", "transform"):
            need(key in p, f"{path}: placement {k} lacks '{key}'")
        need(p["group"] in valid, f"{path}: placement {k} has unknown group {p['group']!r}")
        need(len(p["transform"]) == 16 and np.all(np.isfinite(p["transform"])), f"{path}: placement {k} bad transform")
        if p["mesh"] in THIRD_PARTY_MESHES:
            if variant == "detailed":
                need((Path(meshes_dir) / THIRD_PARTY_SUBDIR / f"{p['mesh']}.stl").is_file(),
                     f"detailed variant: {p['mesh']}.stl missing in {Path(meshes_dir) / THIRD_PARTY_SUBDIR} "
                     f"(run tools/cad_visual/install_detailed_cad.py)")
        else:
            need((Path(meshes_dir) / f"{p['mesh']}.stl").is_file(), f"mesh file {p['mesh']}.stl missing in {meshes_dir}")
        need(material_of(p["mesh"]) in RGBA, f"no curated material for mesh {p['mesh']!r}")
    return plc


def build(src_xml=DEFAULT_SRC, placements_json=DEFAULT_PLACEMENTS, meshes_dir=DEFAULT_MESHES,
          variant="public") -> tuple[str, str]:
    """Return (xml_text, summary)."""
    txt = Path(src_xml).read_text()
    validate_source(txt)
    need(variant in ("public", "detailed"), f"unknown variant {variant!r}")
    placements = load_placements(placements_json, meshes_dir, variant)
    if variant == "public":   # restricted third-party meshes are replaced by generic primitives below
        placements = [p for p in placements if p["mesh"] not in THIRD_PARTY_MESHES]

    shift = np.eye(4); shift[2, 3] = -TILT_AXIS_Z
    groups = {"chassis": [], **{f"rotor_{i}": [] for i in ROTORS}}
    counts = {}
    for p in placements:
        T = shift @ np.array(p["transform"], float).reshape(4, 4)
        if p["group"] != "chassis":
            i = int(p["group"].split("_")[1])
            pos, yaw = ROTORS[i]
            Tr = np.eye(4); Tr[:3, :3] = rotz(yaw); Tr[:3, 3] = pos
            T = np.linalg.inv(Tr) @ T
        n = counts.get((p["group"], p["mesh"]), 0)
        counts[(p["group"], p["mesh"])] = n + 1
        gname = f'{p["group"]}_{p["mesh"]}' + (f"_{n + 1}" if n or p["mesh"] in ("servo_connector",) else "")
        groups[p["group"]].append(geom_xml(gname, p["mesh"], T, "                    "))

    if variant == "public":
        groups["chassis"] += generic_chassis_geoms("                    ")
        for i in ROTORS:
            groups[f"rotor_{i}"] += generic_rotor_geoms(i, "                    ")

    meshes = sorted({p["mesh"] for p in placements})
    mats = sorted({material_of(m) for m in meshes} | (set(GENERIC_MATERIALS) if variant == "public" else set()))
    asset = ["    <asset>", SKYBOX]
    asset += [f'        <material name="{m}" rgba="{" ".join(str(v) for v in RGBA[m])}" specular="0.4" shininess="0.5"/>' for m in mats]
    asset += [f'        <mesh name="{m}" file="{THIRD_PARTY_SUBDIR + "/" if (variant == "detailed" and m in THIRD_PARTY_MESHES) else ""}{m}.stl"/>'
              for m in meshes]
    asset += ["    </asset>", ""]

    # --- header: model name, compiler meshdir, statistic, visual (offscreen size, map, headlight), cad class
    m = find_tag(txt, "mujoco"); txt = replace_span(txt, m, set_attr(m.group(0), "model", "tiltable_quadrotor_visual"))
    m = find_tag(txt, "compiler"); txt = replace_span(txt, m, set_attr(m.group(0), "meshdir", "meshes"))
    m = find_tag(txt, "option"); eol = txt.index("\n", m.end()) + 1
    txt = txt[:eol] + "\n" + STATISTIC + txt[eol:]
    m = find_tag(txt, "global"); eol = txt.index("\n", m.end()) + 1
    g = set_attr(set_attr(m.group(0), "offwidth", "1920"), "offheight", "1080")
    txt = txt[:m.start()] + g + txt[m.end():eol] + MAP + HEADLIGHT + txt[eol:]
    close = [mm.start() for mm in re.finditer(r"</default>", txt)]
    need(close, "source MJCF has no top-level </default>")
    pos = line_start(txt, close[-1])
    txt = txt[:pos] + CAD_DEFAULT + txt[pos:]
    # --- assets before the world body (and its leading comment)
    wb = find_tag(txt, "worldbody"); pos = skip_leading_comments(txt, line_start(txt, wb.start()))
    txt = txt[:pos] + "\n".join(asset) + txt[pos:]
    # --- ground plane styling
    m = find_tag(txt, "geom", "name", "ground_visual")
    tag = set_attr(set_attr(m.group(0), "size", GROUND_SIZE), "rgba", GROUND_RGBA)
    txt = replace_span(txt, m, tag)
    # --- primitive placeholders -> hidden group 4
    for name in PLACEHOLDERS:
        m = find_tag(txt, "geom", "name", name)
        need('class="visual"' in m.group(0), f"placeholder {name} is not class=visual")
        txt = replace_span(txt, m, set_attr(m.group(0), "group", "4", after="class"))
    # --- CAD mesh geoms ahead of the placeholder geoms
    def block(lines):
        return "\n".join(lines) + "\n\n"
    m = find_tag(txt, "geom", "name", "base_visual"); pos = skip_leading_comments(txt, line_start(txt, m.start()))
    chassis = ('                <!-- ===== CAD visual meshes (chassis, arms, servos, end effector) ===== -->\n'
               + block(groups["chassis"]))
    txt = txt[:pos] + chassis + txt[pos:]
    for i in ROTORS:
        m = find_tag(txt, "geom", "name", f"rotor_{i}_visual")
        ws = start = m.start()
        while ws > 0 and txt[ws - 1].isspace():
            ws -= 1
        blk = ('                    <!-- visual geoms: tilting motor assembly + propeller -->\n' + block(groups[f"rotor_{i}"]))
        txt = txt[:ws] + "\n" + blk.rstrip("\n") + "\n" + txt[ws:]
    try:
        ET.fromstring(txt)
    except ET.ParseError as e:
        raise GenError(f"generated XML is not well-formed: {e}")
    return txt, f"{variant}: {len(meshes)} meshes, {sum(len(v) for v in groups.values())} visual geoms"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--placements", type=Path, default=DEFAULT_PLACEMENTS)
    ap.add_argument("--meshes", type=Path, default=DEFAULT_MESHES)
    ap.add_argument("--variant", choices=("public", "detailed"), default="public")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = DEFAULT_OUT if a.variant == "public" else DETAILED_OUT
    try:
        txt, summary = build(a.src, a.placements, a.meshes, a.variant)
    except GenError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(txt)
    print(f"wrote {a.out}: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
