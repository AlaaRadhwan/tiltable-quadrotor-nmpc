"""Per-phase performance and rotation/continuity checks for the combined maneuver CSV.

    tools/python scripts/analyze_combined_maneuver.py [CSV]      # default: newest paper_combined_* CSV
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from tiltable_quadrotor_rl.control.nmpc_effector_paper_maneuvers import PaperStyleCombinedManeuverTrajectory
from tiltable_quadrotor_rl.robot.parameters import MAX_THRUST, SERVO_COMMAND_LIMIT

ROOT = Path(__file__).resolve().parents[1]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else max(
    (p for p in (ROOT / "artifacts/nmpc").glob("paper_combined_*.csv") if "summary" not in p.name),
    key=lambda p: p.stat().st_mtime)
rows = list(csv.DictReader(path.open()))
col = lambda n: np.array([float(r[n]) for r in rows])
t = col("time_s")
tool_mm = 1000 * col("tool_error_m")
att = col("attitude_error_deg")
thrust = np.stack([col(f"thrust_cmd_{i}_N") for i in range(1, 5)], 1)
alpha_cmd = np.stack([col(f"alpha_cmd_{i}_deg") for i in range(1, 5)], 1)
alpha = np.stack([col(f"actual_alpha_{i}_deg") for i in range(1, 5)], 1)
q = np.stack([col(c) for c in ("qw", "qx", "qy", "qz")], 1)
pos = np.stack([col(c) for c in ("body_px_m", "body_py_m", "body_pz_m")], 1)
omega = np.stack([col(c) for c in ("omega_x_radps", "omega_y_radps", "omega_z_radps")], 1)
traj = PaperStyleCombinedManeuverTrajectory(position_W=np.array([0, 0, 1.5]))

print(f"file: {path.name}\nsamples: {len(t)}, duration {t[-1]:.2f} s")
print(f"\n{'phase':36s}{'t0':>7s}{'t1':>7s} | {'tool RMS':>9s}{'max':>8s} [mm] | {'att RMS':>8s}{'max':>8s} [deg] | "
      f"{'max thr':>8s}{'max|a|':>8s}")


def block(name, m):
    if not m.any():
        return
    print(f"{name:36s}{t[m][0]:7.1f}{t[m][-1]:7.1f} | {np.sqrt(np.mean(tool_mm[m] ** 2)):9.2f}{tool_mm[m].max():8.2f}      | "
          f"{np.sqrt(np.mean(att[m] ** 2)):8.3f}{att[m].max():8.3f}       | {thrust[m].max():8.2f}{np.abs(alpha[m]).max():8.1f}")


for name, t0, t1 in traj.phases:
    block(name, (t >= t0) & (t < t1 + (1e-9 if t1 >= traj.end_time_s else 0)))
block("ENTIRE RUN", np.ones_like(t, bool))


def rot(qq):
    w, x, y, z = qq
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


Rs = np.array([rot(qq / np.linalg.norm(qq)) for qq in q])
seg = {n: (t0, t1) for n, t0, t1 in traj.phases}
ry90T = np.array([[0, 0, -1.0], [0, 1, 0], [1, 0, 0]])   # Ry(pi/2)^T
print("\nactual rotation achieved (from logged attitude, unwrapped):")
for name, ang_fn in (
    ("roll_full_360", lambda R: np.arctan2(R[2, 1], R[2, 2])),
    ("pitch_full_360", lambda R: np.arctan2(R[0, 2], R[2, 2])),
    ("cartwheel_360", lambda R: np.arctan2((ry90T @ R)[1, 0], (ry90T @ R)[0, 0])),
):
    t0, t1 = seg[name]
    m = (t >= t0 - 3.0) & (t <= t1 + 3.0)             # include the surrounding holds
    a = np.degrees(np.unwrap([ang_fn(R) for R in Rs[m]]))
    tm = t[m]
    start = a[(tm >= t0 - 1.0) & (tm <= t0)].mean()     # mean over the last second of the hold before
    end = a[(tm >= t1) & (tm <= t1 + 1.0)].mean()       # ... and the first second of the hold after
    print(f"  {name:16s} net rotation {end - start:8.2f} deg between holds "
          f"(reference {np.degrees(-2 * np.pi) if name != 'cartwheel_360' else 360.0:.0f}); peak-to-peak span {a.max() - a.min():7.2f} deg")
t0, t1 = seg["cartwheel_360"]
m = (t >= t0) & (t <= t1)
axis = Rs[m][:, :, 2]
dev = np.degrees(np.arccos(np.clip(axis @ np.array([1.0, 0, 0]), -1, 1)))
print(f"  tool axis (body +Z) vs world +X during cartwheel: mean {dev.mean():.3f} deg, max {dev.max():.3f} deg")
m2 = (t >= seg["hold_after_pitch_to_horizontal"][0]) & (t < seg["hold_after_pitch_to_horizontal"][1])
d2 = np.degrees(np.arccos(np.clip(Rs[m2][:, :, 2] @ np.array([1.0, 0, 0]), -1, 1)))
print(f"  tool axis vs +X in the horizontal hold before the cartwheel: max {d2.max():.3f} deg")

print("\ncontinuity (single run, no reset):")
dt = np.diff(t)
print(f"  time step uniform: {dt.min():.6f}..{dt.max():.6f} s")
print(f"  max |Δ body position| per step {np.linalg.norm(np.diff(pos, axis=0), axis=1).max() * 1000:.3f} mm")
print(f"  max |Δ quaternion| per step {np.linalg.norm(np.diff(q, axis=0), axis=1).max():.5f} (no sign flips: "
      f"{np.all(np.sum(q[1:] * q[:-1], axis=1) > 0)})")
print(f"  max |Δ omega| per step {np.linalg.norm(np.diff(omega, axis=0), axis=1).max():.4f} rad/s")
print(f"  max |Δ thrust cmd| per step {np.abs(np.diff(thrust, axis=0)).max():.3f} N; "
      f"max |Δ alpha cmd| {np.abs(np.diff(alpha_cmd, axis=0)).max():.3f} deg; max |Δ actual alpha| {np.abs(np.diff(alpha, axis=0)).max():.3f} deg")

print("\nactuator saturation / solver:")
lim = np.degrees(SERVO_COMMAND_LIMIT)
print(f"  thrust: max {thrust.max():.2f} N (limit {MAX_THRUST:.0f} N); samples at upper limit: {(thrust >= MAX_THRUST - 1e-3).sum()}; "
      f"at ~0: {(thrust <= 1e-3).sum()}")
print(f"  servo command: max |alpha_cmd| {np.abs(alpha_cmd).max():.3f} deg (limit {lim:.3f} deg); "
      f"samples within 0.01 deg of limit: {(np.abs(alpha_cmd) >= lim - 0.01).sum()} "
      f"({100 * (np.abs(alpha_cmd) >= lim - 0.01).any(axis=1).mean():.1f}% of steps)")
status = [r["solver_status"] for r in rows]
print(f"  solver_status values: {sorted(set(status))}")
print(f"  solver deadline misses (acados): {int(col('solver_deadline_missed').sum())}; "
      f"total-controller deadline misses: {int(col('deadline_missed').sum())} (wall-clock; load dependent)")

fin = t >= traj.phases[-1][2] - 2.0
print(f"\nfinal stabilization (last 2 s): tool error max {tool_mm[fin].max():.3f} mm, attitude error max {att[fin].max():.4f} deg, "
      f"max |omega| {np.abs(omega[fin]).max():.5f} rad/s, final |alpha| max {np.abs(alpha[-1]).max():.5f} deg")
