"""Check for (and optionally build) the OPTIONAL detailed CAD visualization.

The public repository ships only original generic stand-ins for three third-party components
(propeller, BLDC motor, servo). This helper never downloads anything and never bypasses a login.
It checks whether you have placed the converted meshes in

    assets/cad_visual/meshes/third_party/

and tells you where to obtain the source models (under their providers' own terms; see
THIRD_PARTY.md). With --build it generates assets/cad_visual/tiltable_quadrotor_visual_detailed.xml.

    tools/python tools/cad_visual/install_detailed_cad.py            # report what is missing
    tools/python tools/cad_visual/install_detailed_cad.py --build    # also write the detailed XML

Important: the meshes must be binary/ASCII STL in metres, with each part's origin and axes
matching the part-local frame expected by assets/cad_visual/cad_placements.json. Raw downloads
usually do NOT match, so you may have to re-export or re-align them. That alignment is not
provided and is not guaranteed to be possible. The default visualization needs none of this.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_visual_mjcf as B  # noqa: E402

SOURCES = {
    "APC 8x8E propeller (GrabCAD)": (
        "https://grabcad.com/library/apc-8-x-8e-8-propeller-recreation-from-optical-scan-1", ["prop"]),
    "T-Motor AT2814 BLDC motor (GrabCAD)": (
        "https://grabcad.com/library/t-motor-at2814-1",
        ["motor_partbody", "motor_corpo2", "motor_corpo3", "motor_corpo62", "motor_plate", "motor_shaft"]),
    "ROBOTIS servo (manufacturer download page)": (
        "https://en.robotis.com/service/downloadpage.php?ca_id=70",
        ["servo_case_b", "servo_case_f", "servo_case_m", "servo_horn", "servo_horn_idle",
         "servo_idle_cap", "servo_connector"]),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true", help="generate the detailed visual MJCF if all files are present")
    a = ap.parse_args(argv)
    target = B.DEFAULT_MESHES / B.THIRD_PARTY_SUBDIR
    assert set(sum((v[1] for v in SOURCES.values()), [])) == set(B.THIRD_PARTY_MESHES)
    missing = 0
    print(f"Looking in {target}\n")
    for label, (url, names) in SOURCES.items():
        absent = [n for n in names if not (target / f"{n}.stl").is_file()]
        missing += len(absent)
        print(f"{'OK     ' if not absent else 'MISSING'} {label}")
        if absent:
            print(f"        obtain from: {url}  (under the provider's own terms)")
            print(f"        expected files: {', '.join(n + '.stl' for n in absent)}")
    if missing:
        print(f"\n{missing} file(s) missing. The default (public) visualization works without them.")
        return 1
    print("\nAll optional files present.")
    if a.build:
        return B.main(["--variant", "detailed"])
    print("Run with --build to generate the detailed visual MJCF.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
