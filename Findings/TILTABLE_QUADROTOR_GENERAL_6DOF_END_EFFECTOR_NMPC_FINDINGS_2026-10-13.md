# Tiltable Quadrotor — General 6-DoF End-Effector NMPC Benchmark and Bias-Compensation Findings

**Project:** `tiltable_quadrotor_rl`  
**Repository:** `~/Projects/tiltable_quadrotor_rl`  
**Working branch:** `feature/end-effector-nmpc`  
**Date:** 2026-10-13  
**Scope:** Progress after `TILTABLE_QUADROTOR_END_EFFECTOR_NMPC_CARTWHEEL_AND_PREDICTIVE_FINDINGS_2026-09-26.md`, covering the first general simultaneous 6-DoF end-effector benchmark, baseline tracking results, diagnosis of the remaining lateral error, reuse of the previously identified pitch/roll rate-bias compensation, a temporary timing regression caused by duplicate trajectory evaluation, the optimization that removed it, and the decision to freeze nominal trajectory work and move to model-mismatch / disturbance robustness testing.

---

# 1. Purpose of this update

The previous project milestone ended after validating:

- large-angle fixed-end-effector pitch,
- large-angle fixed-end-effector roll,
- continuous body-Z cartwheel motion,
- low-thrust pair handover,
- static feasibility of the cardinal cartwheel poses,
- predictive pre-positioning of unloaded rotors,
- complete controller timing rather than acados-only timing.

The remaining major nominal-control question was:

\[
\boxed{
\text{Can one unchanged end-effector NMPC track a genuinely simultaneous 6-DoF tool trajectory?}
}
\]

The work documented here answers that question.

The controller was deliberately **not retuned** for this phase.

The same:

- plant model,
- NMPC formulation,
- end-effector cost,
- state/input limits,
- actuator model,
- reference-generation architecture

were retained.

The new work is therefore a test of controller generality rather than another maneuver-specific tuning exercise.

---

# 2. Main conclusion

The general 6-DoF benchmark was successful.

The controller completed a trajectory containing simultaneous:

\[
x,\ y,\ z,\ \phi,\ \theta,\ \psi
\]

motion with:

```text
0 solver failures
0 acados-only 10 ms deadline misses
```

and substantial actuator margin.

The initial uncompensated full-amplitude result was:

```text
tool position RMS:
27.304 mm

tool position max:
42.770 mm

attitude RMS:
2.4095 deg

attitude max:
3.8755 deg
```

The remaining position error was strongly concentrated in X/Y rather than Z.

Analysis showed that the same rate-dependent lateral bias previously identified in isolated pitch/roll tests reappeared under mixed 6-DoF motion.

Reusing the **same previously identified gain**:

\[
k_b
=
0.000750
\frac{\text{m}}{\deg/\text{s}}
\]

without fitting it to the new benchmark reduced full-amplitude position RMS to:

```text
19.811 mm
```

while leaving attitude performance and actuator usage essentially unchanged.

The final optimized implementation restored the total controller timing to near the uncompensated baseline.

The nominal end-effector trajectory phase is therefore considered mature enough to freeze.

---

# 3. Why a new benchmark was needed

The earlier tests were intentionally structured around individual physical questions:

```text
pitch:
fixed tool point during large rotation

roll:
orthogonal fixed-point validation

cartwheel:
load transfer and low-thrust transitions
```

Those tests were useful because they isolated specific effects.

However, they did not demonstrate that the controller works when translational and rotational references are simultaneously changing in several directions.

A general benchmark needed to combine:

- nonzero X motion,
- nonzero Y motion,
- nonzero Z motion,
- changing roll,
- changing pitch,
- changing yaw,
- nonzero tool velocity,
- nonzero tool acceleration,
- nonzero angular velocity,
- nonzero angular acceleration,
- varying actuator allocation over time.

The benchmark was therefore designed to answer a broader architectural question rather than test another special maneuver.

---

# 4. Reusable experiment structure

A deliberate repository-design decision was made during this phase:

> Do not continue creating one-off scripts for every small experiment.

The general 6-DoF work was therefore organized around only two reusable files:

```text
src/tiltable_quadrotor_rl/control/
    nmpc_effector_6dof_trajectory.py

experiments/nmpc/
    run_6dof_effector_benchmark.py
```

Variants are controlled through flags rather than new scripts.

Current usage:

```bash
# Baseline
./tools/python experiments/nmpc/run_6dof_effector_benchmark.py
```

and:

```bash
# Same trajectory with rate-bias compensation
./tools/python experiments/nmpc/run_6dof_effector_benchmark.py \
  --bias-compensation
```

This is the preferred pattern for future robustness variants as well.

---

# 5. General 6-DoF trajectory design

The tool center is:

\[
p_{E,c}^W
=
\begin{bmatrix}
0\\
0\\
1.5
\end{bmatrix}
\text{ m}.
\]

The translational amplitudes are:

\[
A_p=
\begin{bmatrix}
0.35\\
0.30\\
0.20
\end{bmatrix}
\text{ m}.
\]

The translational periods are:

\[
T_p=
\begin{bmatrix}
8\\
10\\
6
\end{bmatrix}
\text{ s}.
\]

The orientation amplitudes are:

\[
A_\eta=
\begin{bmatrix}
40^\circ\\
30^\circ\\
60^\circ
\end{bmatrix}
\]

for:

```text
roll
pitch
yaw
```

respectively.

The orientation periods are:

\[
T_\eta=
\begin{bmatrix}
7\\
9\\
11
\end{bmatrix}
\text{ s}.
\]

Different phases are used for all channels so the six motions do not align into an artificially synchronized trajectory.

---

# 6. Activity window

The benchmark begins and ends at a stationary hover.

Schedule:

```text
0 - 2 s:
stationary center hover

2 - 5 s:
smooth ramp-in of all six trajectory channels

5 - 29 s:
full simultaneous XYZ + roll/pitch/yaw motion

29 - 32 s:
smooth ramp-out

32 - 35 s:
stationary final hover
```

The ramp uses quintic smootherstep:

\[
s(r)
=
10r^3
-
15r^4
+
6r^5.
\]

The trajectory implementation also computes:

\[
\dot s,
\quad
\ddot s,
\quad
\dddot s
\]

so position, velocity, acceleration, and the later compensation derivatives can be generated consistently.

---

# 7. Windowed harmonic trajectory

Each translational or Euler channel uses a windowed harmonic of the form:

\[
y(t)
=
A\,w(t)\sin(\omega t+\varphi).
\]

Its velocity is evaluated analytically:

\[
\dot y
=
A
\left[
\dot w\sin(\cdot)
+
w\omega\cos(\cdot)
\right].
\]

Its acceleration is:

\[
\ddot y
=
A
\left[
\ddot w\sin(\cdot)
+
2\dot w\omega\cos(\cdot)
-
w\omega^2\sin(\cdot)
\right].
\]

For the compensated implementation, jerk is also generated:

\[
\dddot y
=
A
\left[
\dddot w\sin(\cdot)
+
3\ddot w\omega\cos(\cdot)
-
3\dot w\omega^2\sin(\cdot)
-
w\omega^3\cos(\cdot)
\right].
\]

This avoids numerical differentiation inside the controller reference pipeline.

---

# 8. Orientation convention

The benchmark uses ZYX Euler composition:

\[
\boxed{
q_{WE}
=
q_z(\psi)
\otimes
q_y(\theta)
\otimes
q_x(\phi)
}
\]

with:

```text
phi:
roll

theta:
pitch

psi:
yaw
```

The quaternion uses scalar-first ordering:

\[
[q_w,q_x,q_y,q_z].
\]

---

# 9. Consistent body/tool angular velocity

The Euler rates are converted analytically into local end-effector angular velocity.

For ZYX Euler angles:

\[
\omega_E
=
\begin{bmatrix}
\dot\phi-\dot\psi\sin\theta\\
\dot\theta\cos\phi+\dot\psi\sin\phi\cos\theta\\
-\dot\theta\sin\phi+\dot\psi\cos\phi\cos\theta
\end{bmatrix}.
\]

The corresponding:

\[
\dot\omega_E
\]

is also evaluated analytically.

This matters because the end-effector inverse kinematics and feedforward wrench generation require mutually consistent:

- quaternion,
- angular velocity,
- angular acceleration.

The benchmark therefore does not treat Euler-angle derivatives as if they were directly equal to body angular velocity.

---

# 10. First benchmark philosophy

The first run intentionally used:

```text
predictive low-thrust pre-positioning:
OFF

empirical rate-bias compensation:
OFF
```

The goal was to isolate the performance of the frozen nominal controller.

The question was:

\[
\boxed{
\text{Does the controller track a general 6-DoF tool trajectory at all?}
}
\]

before adding previously developed reference-layer improvements.

---

# 11. Baseline general 6-DoF experiment

Artifact:

```text
general_6dof_effector_closed_loop_20261013_114056.csv
general_6dof_effector_closed_loop_20261013_114056_summary.txt
```

Trajectory:

```text
center:
[0, 0, 1.5] m

XYZ amplitudes:
[0.35, 0.30, 0.20] m

XYZ periods:
[8, 10, 6] s

roll/pitch/yaw amplitudes:
[40, 30, 60] deg

roll/pitch/yaw periods:
[7, 9, 11] s

active:
2 to 32 s

full amplitude:
5 to 29 s
```

---

# 12. Baseline execution result

The complete run finished normally.

```text
duration:
35.010 s

control updates:
3501

solver failures:
0

acados-only 10 ms deadline misses:
0

total-controller 10 ms deadline misses:
25
```

The total-controller miss rate was:

\[
\frac{25}{3501}
\approx
0.714\%.
\]

---

# 13. Baseline timing

Acados solve timing:

```text
mean:
0.8004 ms

p95:
0.8770 ms

max:
2.8160 ms
```

Complete `controller.solve()` timing:

```text
mean:
8.5667 ms

p95:
9.0866 ms

max:
14.1474 ms
```

This confirmed again that:

\[
\boxed{
\text{acados is not the dominant runtime cost}
}
\]

and that most computation is in the Python reference/controller wrapper.

However, unlike the cartwheel benchmark, this trajectory remained close to the strict 100 Hz budget with only a small number of wall-clock misses.

---

# 14. Baseline tracking over the complete run

```text
tool position RMS:
24.126 mm

tool position max:
42.770 mm

X RMS:
16.207 mm

Y RMS:
17.812 mm

Z RMS:
1.466 mm

tool velocity RMS:
0.025160 m/s

attitude RMS:
2.152981 deg

attitude max:
3.875483 deg

angular-velocity RMS:
0.039925 rad/s
```

The initial/final stationary sections reduce the complete-run RMS somewhat.

The full-amplitude interval is therefore the more important metric.

---

# 15. Baseline tracking during full-amplitude motion

For:

```text
5 to 29 s
```

the result was:

```text
tool position RMS:
27.304 mm

tool position max:
42.770 mm

X RMS:
18.981 mm

Y RMS:
19.572 mm

Z RMS:
1.469 mm

tool velocity RMS:
0.025156 m/s

attitude RMS:
2.409549 deg

attitude max:
3.875483 deg

angular-velocity RMS:
0.041632 rad/s
```

This showed a very clear anisotropy:

\[
e_z
\ll
e_x,e_y.
\]

Z tracking was already excellent.

The remaining translation problem was overwhelmingly lateral.

---

# 16. Baseline actuator usage

Measured maximums:

```text
max |actual alpha|:
63.655109 deg

max |alpha_cmd|:
63.697818 deg

max thrust:
11.210699 N
```

Model limits remain:

```text
max thrust:
23 N

NMPC servo-angle bound:
±3.15 rad ≈ ±180.5 deg
```

Therefore:

\[
\boxed{
\text{the baseline tracking error was not caused by actuator saturation}
}
\]

There was substantial remaining actuator authority.

---

# 17. Worst baseline position error

The maximum tool-position error occurred at approximately:

```text
t:
12.50 s
```

with:

```text
total:
42.770 mm

X:
-11.332 mm

Y:
-41.114 mm

Z:
+3.240 mm
```

The error was dominated by Y, not Z.

At the worst point the vehicle still had ample thrust and tilt margin.

---

# 18. The general benchmark rediscovered the earlier lateral bias

The baseline CSV was inspected against the desired roll/pitch rates.

During the full-amplitude interval:

\[
\operatorname{corr}
(e_x,\dot\theta)
\approx
-0.691
\]

and:

\[
\operatorname{corr}
(e_y,\dot\phi)
\approx
+0.729.
\]

Linear fits gave approximately:

\[
e_x
\approx
-0.000902
\dot\theta_{\deg/s}
+
0.00157
\]

and:

\[
e_y
\approx
+0.000567
\dot\phi_{\deg/s}
+
0.00051.
\]

The signs match the behavior found earlier in the isolated pitch/roll experiments.

This was an important result:

\[
\boxed{
\text{the dominant 6-DoF lateral error was not a new coupling failure}
}
\]

Instead, the previously identified rate-dependent lateral bias remained present under simultaneous motion.

---

# 19. Why this result matters

The earlier bias law had been identified under much simpler conditions:

```text
fixed tool point
single-axis pitch
single-axis roll
```

The new trajectory includes:

```text
simultaneous translation
simultaneous roll
simultaneous pitch
simultaneous yaw
changing velocity
changing acceleration
changing allocation
```

Yet the same X/pitch-rate and Y/roll-rate relationships remained visible.

This suggested that the empirical effect was not merely an artifact of the isolated benchmark.

It was therefore reasonable to test whether the previously identified compensation could generalize.

---

# 20. Generalized rate-bias compensation

The existing gain was reused unchanged:

\[
\boxed{
k_b
=
0.000750
\frac{\text{m}}{\deg/\text{s}}
}
\]

No coefficient was fitted to the 6-DoF data.

The controller reference receives:

\[
\Delta x
=
+k_b\dot\theta_{\deg/s}
\]

\[
\Delta y
=
-k_b\dot\phi_{\deg/s}
\]

\[
\Delta z
=
0.
\]

The true physical tool target is **not** moved.

Only the internal NMPC translational reference is shifted.

---

# 21. Derivative-consistent compensation

The compensation velocity is:

\[
\Delta v_x
=
+k_b\ddot\theta_{\deg/s^2}
\]

\[
\Delta v_y
=
-k_b\ddot\phi_{\deg/s^2}.
\]

The compensation acceleration is:

\[
\Delta a_x
=
+k_b\dddot\theta_{\deg/s^3}
\]

\[
\Delta a_y
=
-k_b\dddot\phi_{\deg/s^3}.
\]

Thus:

\[
\Delta p,
\quad
\Delta v,
\quad
\Delta a
\]

remain analytically consistent.

---

# 22. True target vs controller reference

To avoid misleading evaluation, the runner was extended to distinguish:

```text
true physical reference
controller reference
actual tool state
```

The primary tracking metrics always compare:

\[
p_{E,\text{actual}}
\]

against:

\[
p_{E,\text{true}}.
\]

The compensation cannot improve the reported metric merely by moving the target definition.

The CSV now logs:

```text
true_ref_px/py/pz
controller_ref_px/py/pz
compensation_x/y/z

true_ref_vx/vy/vz
controller_ref_vx/vy/vz

actual tool state
```

The viewer target marker also represents the **true physical target**, not the shifted internal reference.

---

# 23. First compensated 6-DoF result

Artifact:

```text
general_6dof_effector_closed_loop_bias_comp_20261013_115533.csv
general_6dof_effector_closed_loop_bias_comp_20261013_115533_summary.txt
```

The full-amplitude tracking result became:

```text
tool position RMS:
19.811 mm

tool position max:
37.211 mm

X RMS:
13.800 mm

Y RMS:
14.141 mm

Z RMS:
1.442 mm

tool velocity RMS:
0.020281 m/s

attitude RMS:
2.408233 deg

attitude max:
3.878781 deg

angular-velocity RMS:
0.041625 rad/s
```

---

# 24. Baseline vs compensated tracking

| Metric | Baseline | Compensated | Change |
|---|---:|---:|---:|
| Tool RMS | 27.304 mm | 19.811 mm | -27.4% |
| Tool max | 42.770 mm | 37.211 mm | -13.0% |
| X RMS | 18.981 mm | 13.800 mm | -27.3% |
| Y RMS | 19.572 mm | 14.141 mm | -27.7% |
| Z RMS | 1.469 mm | 1.442 mm | -1.8% |
| Tool velocity RMS | 0.025156 m/s | 0.020281 m/s | -19.4% |
| Attitude RMS | 2.4095° | 2.4082° | essentially unchanged |
| Attitude max | 3.8755° | 3.8788° | essentially unchanged |
| Angular-velocity RMS | 0.041632 rad/s | 0.041625 rad/s | essentially unchanged |

This is the most important result from this phase.

The independently identified bias correction generalized to mixed 6-DoF motion.

---

# 25. Compensation did not simply make the controller chase an easier target

The compensated run also logs the error relative to the shifted internal controller reference.

Full-amplitude controller-reference position RMS was:

```text
27.299 mm
```

while true-target RMS was:

```text
19.811 mm
```

This is expected.

The controller is intentionally biased away from the true target internally so that the physical response lands closer to the true target.

Therefore:

\[
\boxed{
\text{larger internal-reference error can coexist with smaller physical-target error}
}
\]

for a feedforward bias compensator.

---

# 26. Attitude performance remained unchanged

The compensation only shifts the translational reference.

The orientation target remains unchanged.

Measured full-amplitude attitude RMS:

```text
baseline:
2.409549 deg

compensated:
2.408233 deg
```

This difference is negligible.

Therefore the translation improvement was not purchased by sacrificing orientation tracking.

---

# 27. Actuator usage remained unchanged

Compensated actuator usage:

```text
max |actual alpha|:
63.695574 deg

max |alpha_cmd|:
63.738513 deg

max thrust:
11.214863 N
```

Baseline:

```text
max |actual alpha|:
63.655109 deg

max |alpha_cmd|:
63.697818 deg

max thrust:
11.210699 N
```

The difference is negligible.

The improvement is therefore primarily a reference-placement effect, not an increase in actuator effort.

---

# 28. The benchmark never entered the low-thrust/cartwheel regime

Across the general 6-DoF benchmark:

```text
minimum reference rotor thrust:
~5.735 N

minimum commanded rotor thrust:
~5.625 N
```

The predictive pre-positioning system from the cartwheel milestone uses approximately:

```text
unloaded threshold:
1.5 N

re-engagement threshold:
2.5 N
```

Therefore:

\[
\boxed{
\text{predictive low-thrust pre-positioning is not relevant to this benchmark}
}
\]

No rotor becomes close to unloaded.

This is useful because it cleanly separates:

```text
general trajectory tracking
```

from:

```text
low-thrust actuator branch management
```

as two different control problems.

---

# 29. Residual bias after compensation

The compensated CSV was analyzed using the same full-amplitude interval.

Residual correlations became:

\[
\operatorname{corr}
(e_x,\dot\theta)
\approx
-0.160
\]

and:

\[
\operatorname{corr}
(e_y,\dot\phi)
\approx
-0.324.
\]

Approximate residual linear slopes:

\[
e_x
\approx
-0.000153
\dot\theta_{\deg/s}
+
0.00157
\]

and:

\[
e_y
\approx
-0.000182
\dot\phi_{\deg/s}
+
0.00053.
\]

The original strong relationships were therefore substantially suppressed.

The Y slope also changes sign, indicating that the common:

\[
k_b=0.000750
\]

slightly over-compensates that relationship under mixed 6-DoF motion.

---

# 30. Why separate X/Y gains were not tuned

It would be possible to fit separate gains to this trajectory, for example:

```text
k_pitch_to_X
k_roll_to_Y
```

However, this was deliberately not done.

Reasons:

1. The current gain was identified independently from simpler experiments.
2. It already reduces full-motion position RMS by about 27%.
3. Re-fitting gains to this trajectory would make the benchmark less independent.
4. The remaining errors are already small relative to the motion amplitude.
5. The next scientific question is robustness to model mismatch, not trajectory-specific optimization.

The preferred conclusion is therefore:

> A simple compensation identified from isolated maneuvers generalizes meaningfully, but it should not be over-tuned to one nominal trajectory.

---

# 31. Worst compensated position error

After compensation, the maximum tool error occurs around:

```text
t:
26.02 s
```

with approximately:

```text
total:
37.211 mm

X:
+21.085 mm

Y:
+30.629 mm

Z:
+1.395 mm
```

Again, the error remains lateral.

Z tracking is still very strong.

---

# 32. First compensated timing regression

Although the tracking result was successful, the first compensated implementation had a serious computational regression.

First compensated run timing:

```text
acados mean:
0.8490 ms

acados p95:
0.9550 ms

acados max:
2.3040 ms
```

but complete controller timing became:

```text
mean:
10.0112 ms

p95:
10.9192 ms

max:
16.1808 ms
```

Deadline misses:

```text
acados-only:
0

total controller:
1422 / 3501
```

or:

\[
40.6\%.
\]

Because acados remained below 1 ms on average, the new cost clearly came from Python reference generation.

---

# 33. Root cause of the timing regression

The first compensated implementation effectively did the following for every horizon sample:

```text
sample(t)
    -> true_sample(t)
        -> translation_state(t)
        -> euler_state(t)

    -> bias_compensation_state(t)
        -> euler_state(t) AGAIN
```

Thus, the relatively expensive harmonic/Euler calculations were duplicated at every prediction node.

With:

```text
N = 20
```

the controller builds:

```text
21 trajectory samples
```

per control update.

The duplicated work therefore accumulated across the full horizon.

This was an implementation error, not an inherent computational cost of the compensation law.

---

# 34. Optimization

The trajectory code was reorganized so that:

```text
translation_state
euler_state
```

are computed only once inside:

```text
sample(t)
```

The compensation is then evaluated from the already-computed:

```text
EulerKinematics
```

using an internal helper.

Conceptually:

```text
before:
trajectory -> Euler
compensation -> Euler again

after:
trajectory -> Euler once
          -> orientation reference
          -> bias compensation
```

No control equation, gain, trajectory parameter, or NMPC parameter changed.

---

# 35. Final optimized compensated run

Artifact:

```text
general_6dof_effector_closed_loop_bias_comp_20261013_140130.csv
general_6dof_effector_closed_loop_bias_comp_20261013_140130_summary.txt
```

Tracking was **numerically identical** to the first compensated run.

Full-amplitude result remained:

```text
tool position RMS:
19.811 mm

tool position max:
37.211 mm

X RMS:
13.800 mm

Y RMS:
14.141 mm

Z RMS:
1.442 mm

tool velocity RMS:
0.020281 m/s

attitude RMS:
2.408233 deg

attitude max:
3.878781 deg

angular-velocity RMS:
0.041625 rad/s
```

This confirms that the optimization changed computation only.

---

# 36. Final optimized timing

Acados:

```text
mean:
0.7983 ms

p95:
0.8770 ms

max:
2.1920 ms
```

Complete controller:

```text
mean:
8.7489 ms

p95:
9.1017 ms

max:
14.8597 ms
```

Deadline misses:

```text
acados-only:
0

total controller:
29 / 3501
```

or:

\[
0.828\%.
\]

---

# 37. Timing comparison

| Configuration | Total mean | p95 | max | 10 ms misses |
|---|---:|---:|---:|---:|
| Baseline 6-DoF | 8.5667 ms | 9.0866 ms | 14.1474 ms | 25 / 3501 |
| First compensated implementation | 10.0112 ms | 10.9192 ms | 16.1808 ms | 1422 / 3501 |
| Optimized compensated implementation | 8.7489 ms | 9.1017 ms | 14.8597 ms | 29 / 3501 |

The optimization reduced mean total controller time by:

\[
10.0112-8.7489
=
1.2623\text{ ms}
\]

or about:

\[
12.6\%
\]

relative to the first compensated implementation.

Compared with the uncompensated baseline, the final compensation adds only:

\[
8.7489-8.5667
=
0.1822\text{ ms}
\]

or about:

\[
2.1\%.
\]

---

# 38. Updated real-time interpretation

The previous cartwheel findings established that complete controller time, rather than acados solve time alone, must be measured.

This phase reinforces that conclusion.

For the optimized general 6-DoF benchmark:

```text
mean total:
8.75 ms

p95 total:
9.10 ms
```

so the typical cycle remains inside the:

```text
10 ms
```

100 Hz period.

However:

```text
max:
14.86 ms
```

and a small number of deadline misses remain.

Therefore the accurate statement is:

> The current Python implementation is close to 100 Hz real-time behavior in this benchmark, but is not yet a hard real-time implementation.

This is sufficient for the current simulation/research stage.

Hard real-time engineering remains a later implementation task.

---

# 39. What the full 6-DoF benchmark validates

The general benchmark establishes that one unchanged NMPC can simultaneously handle:

- large X motion,
- large Y motion,
- vertical Z motion,
- roll motion,
- pitch motion,
- yaw motion,
- changing tool velocity,
- changing tool acceleration,
- changing angular velocity,
- changing angular acceleration,
- continuously changing control allocation.

The benchmark does not rely on a maneuver-specific controller mode.

This is a stronger architectural result than any one pitch, roll, or cartwheel test.

---

# 40. What it does not validate

The current benchmark remains a **matched-model nominal simulation**.

It does not yet test:

- incorrect mass,
- incorrect inertia,
- thrust coefficient mismatch,
- servo-time-constant mismatch,
- altered tool mass,
- altered tool offset,
- external force disturbances,
- external torque disturbances,
- sensor noise,
- estimator error,
- timing jitter representative of deployment,
- communication delay,
- aerodynamic/model mismatch,
- structural flexibility.

Therefore:

\[
\boxed{
\text{nominal general tracking is validated; robustness is not}
}
\]

---

# 41. Accepted findings

## 41.1 The end-effector NMPC generalizes beyond special maneuvers

One controller tracks simultaneous translation and orientation changes.

## 41.2 Z tracking is substantially better than lateral tracking

Baseline full-amplitude RMS:

```text
X:
18.981 mm

Y:
19.572 mm

Z:
1.469 mm
```

## 41.3 The dominant lateral error is related to the previously identified rate bias

The general benchmark reproduces the same pitch-rate/X and roll-rate/Y relationships.

## 41.4 The existing bias coefficient generalizes

Without re-fitting:

```text
k_b = 0.000750 m/(deg/s)
```

reduced full-amplitude tool RMS by about:

```text
27.4%
```

## 41.5 Translation improvement does not harm attitude tracking

Attitude RMS remains essentially:

```text
2.41 deg
```

before and after compensation.

## 41.6 Translation improvement does not require more actuator effort

Peak thrust and tilt remain essentially unchanged.

## 41.7 Predictive low-thrust pre-positioning is unnecessary for this trajectory

Rotor thrust never approaches the low-thrust thresholds.

## 41.8 The first compensated timing regression was an implementation artifact

Duplicate Euler/trajectory calculation, not compensation mathematics, caused the slowdown.

## 41.9 Reusing computed trajectory kinematics restores timing

The optimized compensated implementation returns to near-baseline runtime with identical tracking.

## 41.10 Further nominal tuning is not the best use of project effort

The current controller has demonstrated the intended nominal architecture.

---

# 42. Current selected general 6-DoF milestone

The preferred nominal benchmark configuration is:

```text
trajectory:
simultaneous XYZ + roll/pitch/yaw harmonics

center:
[0, 0, 1.5] m

XYZ amplitudes:
[0.35, 0.30, 0.20] m

RPY amplitudes:
[40, 30, 60] deg

bias compensation:
enabled

bias gain:
0.000750 m/(deg/s)

predictive low-thrust pre-positioning:
not needed for this benchmark
```

Measured full-amplitude performance:

```text
tool position RMS:
19.811 mm

tool position max:
37.211 mm

X RMS:
13.800 mm

Y RMS:
14.141 mm

Z RMS:
1.442 mm

tool velocity RMS:
0.020281 m/s

attitude RMS:
2.408233 deg

attitude max:
3.878781 deg

angular-velocity RMS:
0.041625 rad/s
```

Actuator usage:

```text
max thrust:
11.214863 N

max |actual alpha|:
63.695574 deg

max |alpha_cmd|:
63.738513 deg
```

Runtime:

```text
solver failures:
0

acados misses:
0

total-controller misses:
29 / 3501

acados mean:
0.7983 ms

total-controller mean:
8.7489 ms

total-controller p95:
9.1017 ms
```

---

# 43. Current code organization

The general benchmark is intentionally contained in:

```text
src/tiltable_quadrotor_rl/control/
    nmpc_effector_6dof_trajectory.py

experiments/nmpc/
    run_6dof_effector_benchmark.py
```

The trajectory module contains:

- activity window,
- analytical translation kinematics,
- analytical Euler kinematics,
- quaternion generation,
- local angular velocity,
- local angular acceleration,
- optional empirical bias compensation,
- true-target vs controller-reference separation.

The runner contains:

- baseline/compensated CLI selection,
- initialization,
- MuJoCo closed-loop execution,
- true target visualization,
- full telemetry,
- actuator logging,
- complete controller timing,
- active/full-amplitude summary metrics.

The preferred repository rule going forward is:

\[
\boxed{
\text{extend reusable runners with options instead of creating one-off scripts}
}
\]

unless an investigation is genuinely architecturally different.

---

# 44. Recommended next project phase

The nominal trajectory phase should now be frozen.

The next question is:

\[
\boxed{
\text{How much of this performance survives when the plant is not the controller model?}
}
\]

The first robustness experiment should be **mass mismatch** because it is physically interpretable and easy to isolate.

Keep the NMPC model at:

```text
3.039 kg
```

while modifying only the MuJoCo plant mass.

A reasonable first comparison is:

```text
nominal:
3.039 kg

-10%:
2.7351 kg

+10%:
3.3429 kg
```

Run the same exact optimized compensated 6-DoF benchmark in each case.

Do not retune the controller between runs.

---

# 45. Metrics for the upcoming mass-mismatch study

The same existing runner should be extended rather than creating a new test file.

For each plant mass, compare:

```text
tool position RMS
tool position max

X/Y/Z RMS

tool velocity RMS

attitude RMS/max

angular-velocity RMS

maximum thrust

maximum servo angle

solver failures

acados timing

total controller timing
```

Additional useful robustness quantities:

```text
mean signed X/Y/Z error
final steady offset
mean thrust shift
tracking degradation relative to nominal
```

This will establish whether the nominal controller:

- naturally tolerates mass error,
- develops a persistent vertical bias,
- increases thrust sufficiently through feedback,
- requires integral / disturbance estimation,
- or begins to violate actuator/controller limits.

---

# 46. Longer robustness roadmap

After mass mismatch, the logical sequence is:

```text
mass mismatch
↓
inertia mismatch
↓
servo time-constant mismatch
↓
thrust-effectiveness mismatch
↓
tool mass / tool-offset mismatch
↓
external force disturbance
↓
external torque disturbance
↓
combined uncertainty
```

Only after observing the actual failure modes should a robustness mechanism be selected.

Possible later options include:

- disturbance observer,
- offset-free NMPC,
- integral augmentation,
- online parameter adaptation,
- residual learned correction,
- robust/tube NMPC concepts,
- RL residual control.

The project should not add these mechanisms before the nominal controller's actual robustness weaknesses are measured.

---

# 47. Final milestone statement

The project has now moved from maneuver-specific end-effector tests to a reusable general free-space tracking benchmark.

The baseline general 6-DoF trajectory showed:

\[
\boxed{
\text{the existing NMPC can track simultaneous translation and rotation}
}
\]

without controller retuning.

The experiment also showed that the previously identified lateral rate bias persists under mixed motion.

Reusing the independently identified:

\[
k_b=0.000750
\frac{\text{m}}{\deg/\text{s}}
\]

reduced full-amplitude tool-position RMS from:

\[
27.304\text{ mm}
\]

to:

\[
19.811\text{ mm},
\]

a reduction of approximately:

\[
27.4\%.
\]

This occurred with essentially unchanged:

- attitude accuracy,
- peak thrust,
- servo excursion,
- solver reliability.

A temporary runtime regression was traced to duplicated Python trajectory evaluation rather than the control law.

After removing the duplicate work, tracking remained numerically identical while mean total controller time returned from:

\[
10.0112\text{ ms}
\]

to:

\[
8.7489\text{ ms}.
\]

The nominal end-effector NMPC phase is therefore considered complete enough for the current simulation stage.

The next milestone is no longer:

> Can the controller follow a more carefully tuned nominal trajectory?

It is:

\[
\boxed{
\text{How robust is the controller when the real plant differs from the NMPC model?}
}
\]

The project should now proceed to controlled model-mismatch and disturbance experiments.

---

# 48. Experiment artifact index

## 48.1 Uncompensated general 6-DoF baseline

```text
general_6dof_effector_closed_loop_20261013_114056.csv
general_6dof_effector_closed_loop_20261013_114056_summary.txt
```

## 48.2 First generalized-bias compensated run

```text
general_6dof_effector_closed_loop_bias_comp_20261013_115533.csv
general_6dof_effector_closed_loop_bias_comp_20261013_115533_summary.txt
```

This run established the tracking improvement but exposed duplicate trajectory-computation overhead.

## 48.3 Optimized generalized-bias compensated run

```text
general_6dof_effector_closed_loop_bias_comp_20261013_140130.csv
general_6dof_effector_closed_loop_bias_comp_20261013_140130_summary.txt
```

This is the preferred final nominal 6-DoF benchmark result.

---

# 49. Source files used for this update

This document continues from:

```text
TILTABLE_QUADROTOR_END_EFFECTOR_NMPC_CARTWHEEL_AND_PREDICTIVE_FINDINGS_2026-09-26.md
```

and records the work represented by:

```text
src/tiltable_quadrotor_rl/control/
    nmpc_effector_6dof_trajectory.py

experiments/nmpc/
    run_6dof_effector_benchmark.py
```

together with the three general 6-DoF experiment CSV/summary pairs listed above.

Measured values in this document come from those experiment outputs.

Correlation/regression values were derived from the logged full-amplitude interval:

```text
5 to 29 s
```

using the desired roll/pitch trajectories and recorded X/Y tracking errors.

No robustness result is claimed yet; robustness testing is the next project phase.
