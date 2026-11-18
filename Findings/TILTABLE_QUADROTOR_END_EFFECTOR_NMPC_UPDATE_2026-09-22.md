# Tiltable Quadrotor — End-Effector NMPC Update After the 2026-09-20 Baseline

**Project:** `tiltable_quadrotor_rl`  
**Repository:** `~/Projects/tiltable_quadrotor_rl`  
**Date:** 2026-09-22  
**Scope:** Progress since `TILTABLE_QUADROTOR_END_EFFECTOR_NMPC_FINDINGS_2026-09-20.md`

---

# 1. Purpose of this update

The previous comprehensive document ended with the selected end-effector manipulation baseline:

- paper-style 360° fixed-pivot pitch validated,
- direct physical servo-angle tracking retained,
- stronger tool-space translation weights retained,
- `R_delta_alpha = 50` selected,
- fixed-pivot pitch tool RMS approximately `2.35 cm`,
- pitch attitude RMS approximately `1.35°`,
- no solver failures,
- no 10 ms deadline misses.

Since then, the project has progressed through four major steps:

1. the equivalent **360° roll** maneuver was implemented and validated,
2. pitch/roll testing was refactored into one reusable maneuver module and one generic runner,
3. the remaining lateral fixed-point error was identified as a repeatable rate-dependent bias and compensated with feedforward reference shifting,
4. the first compensation timing design was rejected and replaced by a **smooth angular-rate-coupled compensation law**, reducing true fixed-point translation RMS to about **8 mm** in both pitch and roll.

The main current conclusion is:

\[
oxed{
	ext{The drone can execute full 360° pitch and roll rotations about a nearly fixed end-effector point with sub-centimeter RMS translation error.}
}
\]

The dominant remaining errors are now concentrated in angular acceleration/deceleration transitions rather than during the steady constant-rate rotation.

---

# 2. Controller baseline remains unchanged

The underlying NMPC configuration was not changed during this phase.

Current selected weights:

```text
Q_position:
[600, 600, 800]

Q_velocity:
[30, 30, 30]

Q_attitude:
[300, 300, 600]

Q_omega:
[5, 5, 5]

Q_alpha:
2

R_thrust:
2

R_delta_alpha:
50
```

Other key values remain:

```text
servo time constant:
0.048 s

physics:
500 Hz

NMPC:
100 Hz

prediction horizon:
2.0 s

stages:
20

tool offset:
[0, 0, 0.15] m
```

No disturbance observer, external wrench estimator, integral augmentation, or offset-free compensation was added.

The improvements described below come from the **reference-generation layer**.

---

# 3. Experiment infrastructure refactor

The earlier implementation had begun accumulating maneuver-specific files. That was replaced by a reusable structure.

Preferred trajectory module:

```text
src/tiltable_quadrotor_rl/control/
    nmpc_effector_paper_maneuvers.py
```

Preferred generic runner:

```text
experiments/nmpc/
    run_paper_maneuver.py
```

The same runner can now execute:

```bash
./tools/python experiments/nmpc/run_paper_maneuver.py   --maneuver pitch
```

or:

```bash
./tools/python experiments/nmpc/run_paper_maneuver.py   --maneuver roll
```

Bias compensation can be enabled with:

```bash
--bias-compensation
```

This is now the preferred pattern for future paper-style maneuvers as well.

---

# 4. Generic maneuver interface

The reusable trajectory class is conceptually:

```python
PaperStyleFixedEffectorTrajectory(
    position_W=...,
    maneuver="pitch" | "roll",
    bias_compensation=...,
)
```

The NMPC itself remains maneuver-agnostic. It only receives:

- desired end-effector position,
- desired end-effector velocity,
- desired end-effector acceleration,
- desired end-effector quaternion,
- desired end-effector angular velocity,
- desired end-effector angular acceleration.

Thus:

\[
oxed{
	ext{maneuver-specific logic remains in the trajectory/reference layer}
}
\]

rather than being embedded in the plant or NMPC model.

---

# 5. Fixed-pivot roll geometry

For the rigid tool:

\[
r_{BE}^{B}
=
egin{bmatrix}
0\
0\
d
\end{bmatrix},
\qquad
d=0.15	ext{ m}.
\]

With a fixed world-frame tool point:

\[
p_E^W=
egin{bmatrix}
0\
0\
1.5
\end{bmatrix},
\]

a roll angle \(\phi\) requires the body origin to move according to:

\[
p_B^W
=
p_E^W
-
R_x(\phi)
egin{bmatrix}
0\
0\
d
\end{bmatrix}
\]

or:

\[
p_B^W
=
egin{bmatrix}
0\
d\sin\phi\
1.5-d\cos\phi
\end{bmatrix}.
\]

This is the roll counterpart of the already validated pitch orbit.

---

# 6. Initial uncompensated roll result

The roll benchmark was first run with the same selected controller as pitch:

\[
R_{\Deltalpha}=50.
\]

Result:

```text
duration:
31.010 s

control updates:
3101

solver failures:
0

10 ms deadline misses:
0
```

Solver timing:

```text
mean:
1.0578 ms

p95:
1.3720 ms

max:
2.9670 ms
```

Tracking:

```text
tool position RMS:
2.3546 cm

tool position max:
4.0713 cm

attitude RMS:
1.3445°

attitude max:
4.8439°
```

Actuator usage:

```text
max |actual alpha|:
180.482°

max |alpha_cmd|:
180.482°

max thrust:
16.099 N
```

This almost exactly matched the selected uncompensated pitch baseline.

---

# 7. Pitch/roll symmetry

Selected uncompensated pitch:

```text
tool RMS:
2.350 cm

attitude RMS:
1.353°
```

Uncompensated roll:

```text
tool RMS:
2.355 cm

attitude RMS:
1.345°
```

The error direction also behaved symmetrically:

```text
pitch:
dominant residual in X

roll:
dominant residual in Y
```

This strongly suggested that the remaining offset was not a pitch-specific bug but a general dynamic tradeoff related to angular rate and actuator use.

---

# 8. Measured rate-dependent roll bias

During constant-rate roll, the end-effector settled at an approximately constant lateral offset.

Typical values:

```text
+36°/s roll:
+2.6 to +2.7 cm Y bias

-36°/s roll:
-2.7 to -2.8 cm Y bias
```

The sign reversed with the sign of roll rate.

This closely mirrored the earlier pitch behavior.

The empirical relationship was therefore approximated by:

\[
k_b
=
rac{0.027}{36}
pprox
7.5	imes10^{-4}
rac{	ext{m}}{\deg/	ext{s}}.
\]

---

# 9. Feedforward bias compensation concept

The compensation idea was:

> If a repeatable rotation-rate-dependent offset is observed, intentionally shift the NMPC tool-position reference by an equal-and-opposite amount.

For pitch:

\[
\Delta p_E
=
egin{bmatrix}
+k_b\dot	heta_{\deg/s}\
0\
0
\end{bmatrix}.
\]

For roll:

\[
\Delta p_E
=
egin{bmatrix}
0\
-k_b\dot\phi_{\deg/s}\
0
\end{bmatrix}.
\]

At:

\[
36^\circ/	ext{s},
\]

the compensation magnitude is:

\[
0.00075	imes36
=
0.027	ext{ m}.
\]

So the maximum shift is approximately:

\[
27	ext{ mm}.
\]

The **true physical target does not move**.

Only the internal controller reference is shifted.

---

# 10. True target vs compensated reference

The real physical target remains:

\[
p_{E,	ext{true}}
=
[0,0,1.5]^T.
\]

The NMPC receives:

\[
p_{E,	ext{NMPC}}
=
p_{E,	ext{true}}
+
\Delta p_E.
\]

Two errors are therefore logged.

## True fixed-point error

\[
e_{	ext{true}}
=
p_{E,	ext{actual}}
-
p_{E,	ext{true}}.
\]

This is the real manipulation metric.

## Controller-reference error

\[
e_{	ext{ctrl}}
=
p_{E,	ext{actual}}
-
p_{E,	ext{NMPC}}.
\]

A good feedforward compensator can intentionally make the second error larger while reducing the first.

That behavior is expected.

---

# 11. First compensation design

The first implementation used a separate translation schedule:

```text
1 s before rotation:
ramp compensation from 0 to full value

during rotation:
hold full compensation

1 s after rotation:
ramp compensation back to zero
```

The translation ramp used quintic smootherstep:

\[
s(r)
=
10r^3
-
15r^4
+
6r^5.
\]

The purpose was to pre-position the drone before the known rate-dependent offset appeared.

---

# 12. First compensated roll result

The first compensated roll run produced:

```text
true fixed-point tool RMS:
0.9906 cm

tool max:
3.9816 cm
```

Compared with:

```text
uncompensated roll RMS:
2.3546 cm
```

this was a large translation improvement.

However:

```text
attitude RMS:
4.826°

attitude max:
15.275°
```

which was substantially worse than the uncompensated case.

The run still had:

```text
0 solver failures
0 deadline misses
```

and repeated trials produced essentially the same behavior.

So the problem was deterministic.

---

# 13. Failure mode of the first compensation design

Visually and in telemetry:

1. the tool remained nearly fixed during most of the rotation,
2. near the final inverted/top attitude, the vehicle stopped settling cleanly,
3. attitude error grew,
4. a lateral translation happened after the rotation,
5. only then did the vehicle return toward the nominal point.

The issue was traced primarily to **timing architecture**, not to the compensation gain itself.

---

# 14. Why the first timing architecture was problematic

The NMPC prediction horizon is:

\[
2.0	ext{ s}.
\]

With the old schedule, the horizon could see:

- the upcoming angular-rate stop,
- the compensation removal,
- and later the next opposite-direction maneuver.

Thus, before reaching the nominal endpoint, the optimizer could already begin preparing for future reference changes.

This encouraged a tradeoff where tool position remained good but attitude settling became poor.

---

# 15. Angular-rate discontinuity was also a problem

The original paper-style linear trajectory used an abrupt desired angular-rate transition:

\[
0
ightarrow
36^\circ/	ext{s}
\]

and later:

\[
36^\circ/	ext{s}
ightarrow
0.
\]

This means the desired angular acceleration is discontinuous at segment boundaries.

That was acceptable for the earlier baseline, but it became problematic once translation compensation was tied to the maneuver.

---

# 16. Revised compensation architecture

The new design makes compensation a direct function of the **smooth desired angular rate**:

\[
oxed{
\Delta p_E
=
k_b\dot\psi
}
\]

where \(\psi\) is pitch or roll.

For pitch:

\[
\Delta p_E
=
egin{bmatrix}
+k_b\dot	heta_{\deg/s}\
0\
0
\end{bmatrix}.
\]

For roll:

\[
\Delta p_E
=
egin{bmatrix}
0\
-k_b\dot\phi_{\deg/s}\
0
\end{bmatrix}.
\]

Therefore:

\[
\dot\psiightarrow0
\quad\Rightarrow\quad
\Delta p_Eightarrow0.
\]

There is no independent post-rotation translation schedule anymore.

---

# 17. Smooth angular-rate profile

The successful peak rate was preserved:

\[
|\dot\psi|_{\max}=36^\circ/	ext{s}.
\]

Each maneuver now uses:

```text
smooth acceleration ramp
constant-rate cruise
smooth deceleration ramp
```

with:

```text
ramp duration:
1.0 s
```

The rate ramp uses the quintic smootherstep:

\[
s(r)=10r^3-15r^4+6r^5.
\]

During acceleration:

\[
\dot\psi
=
\dot\psi_{\max}s(r).
\]

During deceleration:

\[
\dot\psi
=
\dot\psi_{\max}[1-s(r)].
\]

---

# 18. New segment durations

For a 180° move:

```text
1 s acceleration
4 s cruise
1 s deceleration
```

Total:

\[
6	ext{ s}.
\]

For the full 360° move:

```text
1 s acceleration
9 s cruise
1 s deceleration
```

Total:

\[
11	ext{ s}.
\]

---

# 19. Current benchmark timing

The current smooth profile is:

```text
0–2 s:
hold 0°

2–8 s:
0 → +180°
smooth rate
max +36°/s

8–11 s:
hold +180°

11–22 s:
+180° → -180°
full 360°
smooth rate
max -36°/s

22–25 s:
hold -180°

25–31 s:
-180° → 0°
smooth rate
max +36°/s

31–34 s:
final hold
```

Total:

\[
34	ext{ s}.
\]

---

# 20. Compensation derivatives are now consistent

The compensation is generated from the smooth rotation kinematics.

Position:

\[
\Delta p_E
=
k_b\dot\psi_{\deg/s}\hat d.
\]

Velocity:

\[
\Delta v_E
=
k_b\ddot\psi_{\deg/s^2}\hat d.
\]

Acceleration:

\[
\Delta a_E
=
k_b\dddot\psi_{\deg/s^3}\hat d.
\]

For pitch:

\[
\hat d=
egin{bmatrix}
1\
0\
0
\end{bmatrix}.
\]

For roll:

\[
\hat d=
egin{bmatrix}
0\
-1\
0
\end{bmatrix}.
\]

Thus the reference is internally consistent in position, velocity, and acceleration.

---

# 21. Current pitch result

Configuration:

```text
maneuver:
pitch

smooth angular-rate profile:
True

max angular rate:
36 deg/s

angular-rate ramp:
1.0 s

rate-bias compensation:
True

bias gain:
0.000750 m/(deg/s)

maximum compensation:
0.027 m
```

Runtime:

```text
duration:
34.010 s

control updates:
3401

solver failures:
0

10 ms deadline misses:
0
```

Solver timing:

```text
mean:
1.1294 ms

p95:
1.5120 ms

max:
2.4370 ms
```

Tracking relative to the **true fixed point**:

```text
tool RMS:
0.8038 cm

tool max:
2.6542 cm

attitude RMS:
2.7943°

attitude max:
12.7827°
```

Actuator usage:

```text
max |actual alpha|:
180.482°

max |alpha_cmd|:
180.482°

max thrust:
15.997 N
```

---

# 22. Current roll result

Configuration is the same except for maneuver axis.

Runtime:

```text
duration:
34.010 s

control updates:
3401

solver failures:
0

10 ms deadline misses:
0
```

Solver timing:

```text
mean:
1.1162 ms

p95:
1.4290 ms

max:
2.3840 ms
```

Tracking relative to the **true fixed point**:

```text
tool RMS:
0.8108 cm

tool max:
2.6548 cm

attitude RMS:
2.8135°

attitude max:
12.8835°
```

Actuator usage:

```text
max |actual alpha|:
180.482°

max |alpha_cmd|:
180.482°

max thrust:
16.096 N
```

---

# 23. Pitch/roll symmetry remains excellent

| Metric | Pitch | Roll |
|---|---:|---:|
| Tool RMS | `8.04 mm` | `8.11 mm` |
| Tool max | `26.54 mm` | `26.55 mm` |
| Attitude RMS | `2.79°` | `2.81°` |
| Attitude max | `12.78°` | `12.88°` |
| Mean solve | `1.129 ms` | `1.116 ms` |
| p95 solve | `1.512 ms` | `1.429 ms` |
| Solver failures | `0` | `0` |
| Deadline misses | `0` | `0` |

The near-identical behavior across two orthogonal axes is a strong validation of the architecture.

---

# 24. Improvement relative to uncompensated R=50 baseline

Pitch:

```text
uncompensated:
23.50 mm RMS

smooth compensated:
8.04 mm RMS
```

Reduction:

\[
pprox66\%.
\]

Roll:

```text
uncompensated:
23.55 mm RMS

smooth compensated:
8.11 mm RMS
```

Reduction:

\[
pprox66\%.
\]

So the feedforward correction removes roughly two-thirds of the true fixed-point translation error over the complete maneuver.

---

# 25. Constant-rate rotation performance

Detailed inspection showed that the **constant-rate cruise portions** are now much better than the global RMS suggests.

Approximate steady-rotation performance:

```text
pitch tool RMS:
~5.7 mm

roll tool RMS:
~5.9 mm

pitch attitude RMS:
~1.19°

roll attitude RMS:
~1.17°
```

This is the most important practical result.

During the main 36 deg/s rotation:

\[
oxed{
	ext{the tool remains essentially fixed to within a few millimeters RMS}
}
\]

while attitude tracking remains close to the original uncompensated quality.

---

# 26. Remaining error is concentrated in transitions

The largest remaining errors occur during the 1 s acceleration/deceleration ramps.

Approximate transition tool RMS:

```text
pitch:
~15.1 mm

roll:
similar
```

The worst tool error:

\[
pprox26.5	ext{ mm}
\]

occurs near a transition.

The largest attitude errors:

\[
pprox12.8^\circ
\]

also occur near transitions.

The dominant failure mode has therefore changed from:

```text
persistent steady rate-dependent lateral offset
```

to:

```text
short transition-related error during angular acceleration/deceleration
```

This is a major improvement.

---

# 27. Interpretation of the current bias model

The simple empirical relation:

\[
\Delta p_E=k_b\dot\psi
\]

works very well in the tested regime.

However, it remains an empirical model.

It should not yet be assumed to remain valid for:

- much larger angular rates,
- different end-effector offsets,
- different mass/inertia,
- different servo dynamics,
- contact manipulation,
- real hardware.

For the present nominal vehicle and 36 deg/s pitch/roll maneuvers, it is effective.

---

# 28. Why the revised architecture is preferred

The current smooth rate-coupled design is preferred because:

1. compensation is zero whenever angular rate is zero,
2. compensation changes continuously with angular rate,
3. there is no separate post-maneuver translation event,
4. position/velocity/acceleration compensation are mutually consistent,
5. the 2 s NMPC horizon handles anticipation naturally,
6. pitch and roll use the same generic code,
7. the plant and OCP remain unchanged.

---

# 29. Current logging

The generic runner now records:

```text
time_s

angle_ref_deg
angle_rate_ref_degps
angle_accel_ref_degps2

compensation_x_m
compensation_y_m
compensation_z_m

controller_ref_px_m
controller_ref_py_m
controller_ref_pz_m

solve_time_ms
solver_status
deadline_missed

tool_error_m
controller_reference_error_m
attitude_error_deg

body position
body quaternion
body angular velocity

raw thrust references
raw alpha references

refined thrust references
refined alpha references

actual alpha
alpha commands
thrust commands
```

This is sufficient for future maneuver analysis.

---

# 30. Current preferred commands

Compensated pitch:

```bash
./tools/python experiments/nmpc/run_paper_maneuver.py   --maneuver pitch   --bias-compensation
```

Compensated roll:

```bash
./tools/python experiments/nmpc/run_paper_maneuver.py   --maneuver roll   --bias-compensation
```

These are now the preferred pitch/roll demonstration configurations.

---

# 31. Current best pitch milestone

```text
R_delta_alpha:
50

bias compensation:
enabled

bias gain:
0.000750 m/(deg/s)

peak rate:
36 deg/s

rate ramp:
1.0 s

tool RMS:
8.04 mm

tool max:
26.54 mm

attitude RMS:
2.79°

attitude max:
12.78°

steady-rotation tool RMS:
~5.7 mm

steady-rotation attitude RMS:
~1.19°

solver failures:
0

deadline misses:
0
```

---

# 32. Current best roll milestone

```text
R_delta_alpha:
50

bias compensation:
enabled

bias gain:
0.000750 m/(deg/s)

peak rate:
36 deg/s

rate ramp:
1.0 s

tool RMS:
8.11 mm

tool max:
26.55 mm

attitude RMS:
2.81°

attitude max:
12.88°

steady-rotation tool RMS:
~5.9 mm

steady-rotation attitude RMS:
~1.17°

solver failures:
0

deadline misses:
0
```

---

# 33. Computational margin

The 100 Hz deadline is:

\[
10	ext{ ms}.
\]

Current compensated pitch:

```text
mean:
1.129 ms

p95:
1.512 ms

max:
2.437 ms
```

Current compensated roll:

```text
mean:
1.116 ms

p95:
1.429 ms

max:
2.384 ms
```

The controller retains substantial real-time margin.

---

# 34. What has deliberately not changed

During this phase, the following were kept fixed:

```text
rigid-body plant
vehicle mass
vehicle inertia
rotor geometry
tool offset
servo time constant
control rate
physics rate
prediction horizon
R_delta_alpha
tool-space weights
attitude weights
allocator
singularity refinement
actuator limits
```

This means the observed improvement can be attributed mainly to the new reference-generation strategy.

---

# 35. Accepted findings

## 35.1 Pitch and roll are highly symmetric

Both uncompensated and compensated results are nearly identical.

## 35.2 The original residual lateral bias is strongly rate-dependent

Its sign reverses with angular-rate direction.

## 35.3 A simple rate-proportional feedforward correction is effective

At 36 deg/s, approximately 27 mm of opposite reference shift removes most of the steady fixed-point error.

## 35.4 Independent pre/post translation scheduling is superseded

It improved position but harmed attitude settling.

## 35.5 Smooth angular-rate ramps are preferable

They remove the abrupt angular-rate stop and naturally return compensation to zero.

## 35.6 The dominant residual error is now transition-related

Steady rotation is much better than the acceleration/deceleration phases.

## 35.7 Real-time feasibility remains strong

The new reference logic does not threaten the 10 ms NMPC deadline.

---

# 36. Superseded approach

The old compensation strategy:

```text
pre-ramp compensation
+
constant compensation
+
post-ramp compensation
```

should no longer be used as the preferred implementation.

The preferred form is:

\[
oxed{
\Delta p_E=k_b\dot\psi
}
\]

with smooth \(\dot\psi\).

---

# 37. Current selected configuration

```text
R_delta_alpha:
50

bias compensation:
enabled

bias gain:
0.000750 m/(deg/s)

peak angular rate:
36 deg/s

angular-rate ramp:
1.0 s

tool target:
[0, 0, 1.5] m

tool offset:
[0, 0, 0.15] m

controller:
100 Hz

physics:
500 Hz

prediction horizon:
2.0 s
```

---

# 38. Optional final cleanup

If the transition behavior needs further improvement before moving on, the cleanest single-variable test would be:

```text
ANGULAR_RATE_RAMP_DURATION_S:
1.0 s -> 1.5 s
```

while keeping:

```text
peak rate:
36 deg/s

bias gain:
0.000750 m/(deg/s)

R_delta_alpha:
50
```

A longer ramp should reduce:

- angular acceleration,
- angular jerk,
- transition feedforward demand.

This is optional rather than required.

---

# 39. Current milestone

The project has now demonstrated:

\[
oxed{
	ext{full 360° pitch about a nearly fixed end-effector point}
}
\]

and:

\[
oxed{
	ext{full 360° roll about a nearly fixed end-effector point}
}
\]

with approximately:

\[
oxed{
8	ext{ mm RMS true fixed-point translation error}
}
\]

over the full smooth maneuver.

During constant-rate rotation:

\[
oxed{
	ext{tool error is approximately }6	ext{ mm RMS}
}
\]

and:

\[
oxed{
	ext{attitude error is approximately }1.2^\circ	ext{ RMS}
}
\]

for both axes.

Both maneuvers retain:

```text
0 solver failures
0 deadline misses
```

at:

```text
100 Hz NMPC
500 Hz physics
```

---

# 40. Recommended next step

The project is now ready to continue to the next paper-style omnidirectional maneuver.

The natural next candidate is the **vertical cartwheel / singularity-traversal experiment**.

Before that, only one optional cleanup remains: test a slightly longer angular-rate ramp if transition smoothness is considered important enough to justify another tuning run.

Otherwise, the current pitch/roll state should be preserved and committed before adding the next maneuver.

---

# 41. Final conclusion

Since the previous comprehensive document, the project has moved from a single validated pitch benchmark with roughly `2.35 cm` RMS fixed-point error to a reusable pitch/roll framework with smooth feedforward compensation.

The important sequence of findings was:

```text
uncompensated pitch works
↓
uncompensated roll behaves almost identically
↓
bias is systematic and rate-dependent
↓
independent compensation schedule improves translation but harms attitude settling
↓
smooth angular-rate-coupled compensation fixes the timing problem
↓
pitch and roll both reach ~8 mm RMS true fixed-point error
```

The final preferred compensation law is:

\[
oxed{
\Delta p_E=k_b\dot\psi
}
\]

with:

\[
k_b
=
7.5	imes10^{-4}
rac{	ext{m}}{\deg/	ext{s}}
\]

and a smooth rate profile capped at:

\[
36^\circ/	ext{s}.
\]

The dominant residual is now transition-related rather than a steady fixed-point bias.

The pitch/roll milestone can therefore be considered successful, and the project is ready to proceed to the next omnidirectional maneuver.
