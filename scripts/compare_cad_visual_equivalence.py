"""Physics-equivalence check: original (headless) runs vs. CAD-visualized MP4-rendering runs.

For every selected experiment the runner is executed three times with identical arguments:
  A  original, --headless
  B  original, --headless (repeat: establishes the run-to-run noise floor)
  C  --render-mp4 (CAD visual + spinning propellers, offscreen)   [this is the delivered video]
and the CSV logs are compared column by column.  Wall-clock columns (solver/controller times and
deadline flags) are reported separately from the simulated-dynamics columns.

    MUJOCO_GL=glfw tools/python scripts/compare_cad_visual_equivalence.py --suite 6dof
    MUJOCO_GL=glfw tools/python scripts/compare_cad_visual_equivalence.py --suite paper
    MUJOCO_GL=glfw tools/python scripts/compare_cad_visual_equivalence.py --suite paper --only yaw_360 [--no-video]

Suites (video outputs under artifacts/cad_visual_renders/):
  6dof   nmpc_6dof_maneuver.mp4
  paper  paper_maneuvers/{pitch_360,roll_360,cartwheel,cartwheel_baseline,yaw_360}.mp4
  combined  combined_360_maneuvers.mp4   (one continuous roll/pitch/cartwheel run)
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NMPC = ROOT / "experiments/nmpc"
OUT = ROOT / "artifacts/cad_visual_renders"
SUITES = {
    "6dof": {
        "nmpc_6dof_maneuver": (NMPC / "run_6dof_effector_benchmark.py", [], OUT / "nmpc_6dof_maneuver.mp4"),
    },
    "paper": {
        "pitch_360": (NMPC / "run_paper_maneuver.py", ["--maneuver", "pitch", "--bias-compensation"],
                      OUT / "paper_maneuvers/pitch_360.mp4"),
        "roll_360": (NMPC / "run_paper_maneuver.py", ["--maneuver", "roll", "--bias-compensation"],
                     OUT / "paper_maneuvers/roll_360.mp4"),
        "cartwheel": (NMPC / "run_paper_maneuver.py", ["--maneuver", "cartwheel", "--predictive-preposition"],
                      OUT / "paper_maneuvers/cartwheel.mp4"),
        "cartwheel_baseline": (NMPC / "run_paper_maneuver.py", ["--maneuver", "cartwheel"],
                               OUT / "paper_maneuvers/cartwheel_baseline.mp4"),
        "yaw_360": (NMPC / "run_paper_maneuver.py", ["--maneuver", "yaw"], OUT / "paper_maneuvers/yaw_360.mp4"),
    },
    "combined": {
        "combined_360_maneuvers": (NMPC / "run_paper_maneuver.py", ["--maneuver", "combined", "--bias-compensation"],
                                   OUT / "combined_360_maneuvers.mp4"),
    },
}


def is_wall_clock(col: str) -> bool:
    return col.endswith("_time_ms") or "deadline" in col


def run(script: Path, args: list[str]):
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, cwd=ROOT)
    wall = time.perf_counter() - t0
    if p.returncode:
        raise SystemExit(f"run failed ({' '.join(args)}):\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    out = p.stdout
    info = {
        "steps": int(re.search(r"Control steps\s*:\s*(\d+)", out).group(1)),
        "t_final": re.search(r"Final sim time \[s\]\s*:\s*(\S+)", out).group(1),
        "qpos": re.search(r"Final qpos\s*:\s*(\[.*?\])", out, re.S).group(1),
        "alpha": re.search(r"Final alpha \[deg\]\s*:\s*(\[.*?\])", out, re.S).group(1),
        "fail": "NMPC FAILURE" in out,
        "wall": wall,
    }
    csv_path = Path(re.search(r"CSV:\n(.+)\n", out).group(1).strip())
    return info, list(csv.DictReader(csv_path.open())), csv_path


def values(rows, name):
    return np.array([float(r[name]) for r in rows])


def nan_aware_max_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Max |a-b|; NaNs must sit at identical positions (e.g. unused diagnostic columns), else inf."""
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    ok = ~np.isnan(a)
    return float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else 0.0


def compare(name, script, args, mp4, render):
    A = run(script, ["--headless", *args])
    B = run(script, ["--headless", *args])
    C = run(script, ["--render-mp4", str(mp4), *args]) if render else None
    runs = [("A orig", A), ("B orig", B)] + ([("C cad+mp4", C)] if C else [])
    print(f"\n{'=' * 78}\n{name}: {' '.join(args)}\n{'=' * 78}")
    print("logs: " + " | ".join(f"{n}: {r[2].name}" for n, r in runs))
    print(f"{'':30s}" + "".join(f"{n:>16s}" for n, _ in runs))
    for k in ("steps", "t_final", "fail"):
        print(f"{k:30s}" + "".join(f"{r[0][k]!s:>16s}" for _, r in runs))
    print(f"{'rows':30s}" + "".join(f"{len(r[1]):>16d}" for _, r in runs))
    for k in ("qpos", "alpha"):
        print(f"{'final ' + k + ' == A':30s}" + "".join(f"{str(r[0][k] == A[0][k]):>16s}" for _, r in runs))
    print(f"{'wall-clock total [s]':30s}" + "".join(f"{r[0]['wall']:16.1f}" for _, r in runs))

    cols = list(A[1][0].keys())
    dyn = [c for c in cols if not is_wall_clock(c)]
    numeric = [c for c in dyn if re.fullmatch(r"[-+0-9.eE]+|nan|inf", A[1][0][c] or "0")]
    textual = [c for c in dyn if c not in numeric]
    worst = {}
    for tag, R in runs[1:]:
        assert len(R[1]) == len(A[1]), "row count differs"
        d = {c: nan_aware_max_diff(values(A[1], c), values(R[1], c)) for c in numeric}
        d.update({c: 0.0 if [r[c] for r in A[1]] == [r[c] for r in R[1]] else float("inf") for c in textual})
        worst[tag] = d
    print(f"\nsimulated-dynamics columns compared: {len(dyn)} ({len(numeric)} numeric, {len(textual)} text/status)")
    for tag, d in worst.items():
        c, v = max(d.items(), key=lambda kv: kv[1])
        print(f"  max |A - {tag}| over all columns = {v:.3e}  (worst column: {c})")
    key = [c for c in numeric if re.search(r"tool_(position_)?error|attitude_error|thrust_cmd|alpha_cmd|actual_(p|q|v|omega|alpha)", c)]
    print("  key columns (max |A-B| / max |A-C|):")
    for c in key:
        cc = worst.get("C cad+mp4", {}).get(c, float("nan"))
        print(f"    {c:30s} {worst['B orig'][c]:.2e} / {cc:.2e}")
    print(f"  solver_status identical A/B, A/C: " + " / ".join(
        str([r["solver_status"] for r in A[1]] == [r["solver_status"] for r in R[1]]) for _, R in runs[1:]))

    print("\nwall-clock metrics (not simulated dynamics):")
    for c in ("total_controller_time_ms", "solve_time_ms"):
        if c in cols:
            print(f"  {c:28s} mean " + " / ".join(f"{values(r[1], c).mean():.2f}" for _, r in runs) + " ms; max "
                  + " / ".join(f"{values(r[1], c).max():.2f}" for _, r in runs) + " ms")
    for c in [c for c in cols if "deadline" in c]:
        print(f"  {c:28s} count " + " / ".join(str(int(values(r[1], c).sum())) for _, r in runs))
    exact = all(v == 0.0 for d in worst.values() for v in d.values())
    print(f"\nRESULT {name}: dynamics {'BIT-IDENTICAL' if exact else 'DIFFER'} across runs")
    return exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES), default="6dof")
    ap.add_argument("--only", nargs="*", help="subset of experiment names in the suite")
    ap.add_argument("--no-video", action="store_true", help="skip the rendered run C (headless A/B only)")
    a = ap.parse_args()
    results = {}
    for name, (script, args, mp4) in SUITES[a.suite].items():
        if a.only and name not in a.only:
            continue
        results[name] = compare(name, script, args, mp4, not a.no_video)
    print("\nSUMMARY:", {k: ("bit-identical" if v else "DIFFERENT") for k, v in results.items()})


if __name__ == "__main__":
    main()
