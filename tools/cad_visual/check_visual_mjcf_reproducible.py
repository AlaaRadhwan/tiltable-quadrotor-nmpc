"""Verify the checked-in visual MJCF is exactly what build_visual_mjcf.py generates (offline, read-only).

Generates twice into a temp dir (determinism), compares with the checked-in file byte-for-byte, and
confirms key visual config values. Never overwrites the checked-in file. Exit 0 = consistent, 1 = not.
    tools/python tools/cad_visual/check_visual_mjcf_reproducible.py
"""
from __future__ import annotations

import difflib
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_visual_mjcf as B  # noqa: E402


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as td:
        outs = []
        for k in range(2):
            out = Path(td) / f"gen{k}.xml"
            if B.main(["--out", str(out)]) != 0:
                return 1
            outs.append(out.read_text())
    if outs[0] != outs[1]:
        print("FAIL: two generator runs differ (non-deterministic)"); ok = False
    else:
        print("OK: two generator runs are identical")
    committed = B.DEFAULT_OUT.read_text()
    if outs[0] == committed:
        print(f"OK: generated output is byte-identical to {B.DEFAULT_OUT.relative_to(B.ROOT)}")
    else:
        ok = False
        print("FAIL: generated output differs from the checked-in file:")
        sys.stdout.writelines(difflib.unified_diff(committed.splitlines(True), outs[0].splitlines(True),
                                                   "checked-in", "generated", n=2))
    root = ET.fromstring(outs[0])
    checks = {
        "public variant: no restricted mesh referenced": not any(
            Path(m.get("file", "")).stem in B.THIRD_PARTY_MESHES for m in root.iter("mesh")),
        "public variant: original meshes + generic primitives": sum(1 for g in root.iter("geom") if g.get("class") == "cad") == 50,
        "ground size 500 500 0.1": any(g.get("name") == "ground_visual" and g.get("size") == B.GROUND_SIZE for g in root.iter("geom")),
        "statistic extent/center": root.find("statistic") is not None and root.find("statistic").get("extent") == "10",
        "map znear/zfar": root.find("visual/map") is not None and root.find("visual/map").get("zfar") == "60",
        "offscreen 1920x1080": root.find("visual/global").get("offwidth") == "1920" and root.find("visual/global").get("offheight") == "1080",
        "headlight": root.find("visual/headlight") is not None,
        "skybox": any(t.get("type") == "skybox" for t in root.iter("texture")),
        "placeholders in group 4": sum(1 for g in root.iter("geom") if g.get("group") == "4") == 9,
    }
    for k, v in checks.items():
        print(("OK:   " if v else "FAIL: ") + k); ok &= v
    print("REPRODUCIBLE" if ok else "NOT REPRODUCIBLE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
