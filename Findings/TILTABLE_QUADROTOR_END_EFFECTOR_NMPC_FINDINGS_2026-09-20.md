# Tiltable Quadrotor — End-Effector NMPC, Fixed-Pivot Rotation, Servo-Regularization Tuning, and Current Project Findings

**Project:** `tiltable_quadrotor_rl`  
**Repository:** `~/Projects/tiltable_quadrotor_rl`  
**Date:** 2026-09-20  
**Scope:** Consolidated project findings through the current fixed-pivot end-effector NMPC milestone

---

# 1. Purpose of this document

This document consolidates the most important engineering results obtained after the nominal rigid-body NMPC architecture was validated.

The earlier project work established:

- a rigid six-DoF MuJoCo vehicle model,
- a separate first-order model for the four tilt actuators,
- analytical rotor wrench generation,
- a 17-state / 8-input nonlinear prediction model,
- tight CasADi-vs-MuJoCo model consistency,
- an acados SQP-RTI controller running at 100 Hz,
- and successful nominal body/CoG trajectory tracking.

The focus then moved from ordinary flight control to **end-effector-centric aerial manipulation**.

Since then the project has implemented and tested:

- a fixed rigid end-effector transform,
- tool-space kinematics,
- tool-to-body reference conversion,
- an end-effector-centric NMPC cost,
- free-space tool trajectory tracking,
- fixed-pivot 360° pitch motion,
- singularity-aware actuator references,
- several trajectory-shaping strategies,
- a periodic servo-angle-cost experiment,
- stronger tool-space weighting,
- a detailed diagnosis of the remaining X-direction fixed-pivot bias,
- and a controlled sweep of the servo-motion regularization weight \(R_{\Delta\alpha}\).

The strongest current engineering conclusion is:

\[
\boxed{
\text{The end-effector NMPC works reliably, and the dominant residual tracking error was strongly tied to excessive servo-motion regularization.}
}
\]

The project has now reached a point where the main remaining questions are no longer about whether the architecture works. They are about:

- how aggressively the tilt actuators should be used,
- how to preserve hardware realism,
- how to extend the benchmark to roll and mixed-axis manipulation,
- and how to prepare for the later RL phase.

---

# 2. Current nominal vehicle model

The nominal rigid vehicle parameters are:

| Parameter | Current value |
|---|---:|
| Mass | `3.039 kg` |
| Gravity | `9.81 m/s²` |
| Weight | `29.81259 N` |
| Inertia | `diag(0.063, 0.062, 0.095) kg·m²` |
| Rotor center distance from body origin | `0.275 m` |
| Rotor radius | `0.114 m` |
| Rotor diameter | `0.228 m` |
| Max thrust / rotor | `23 N` |
| Reaction-torque ratio `kq/kt` | `0.0165 m` |
| Servo time constant | `0.048 s` |
| NMPC alpha bound | `±3.15 rad` |
| Conceptual mechanical servo range | `±225°` |
| Physics timestep | `0.002 s` |
| Physics rate | `500 Hz` |
| NMPC period | `0.010 s` |
| NMPC rate | `100 Hz` |
| NMPC stages | `20` |
| Prediction horizon | `2.0 s` |

The rigid plant intentionally contains **no mechanical tilt joints**.

The four tilt angles are actuator states external to MuJoCo's rigid generalized coordinates.

This remains a critical architectural decision.

---

# 3. Current plant/control architecture

The current nominal closed-loop chain is:

```text
desired trajectory
    ↓
reference generation
    ↓
body / end-effector reference conversion
    ↓
feed-forward body wrench
    ↓
pseudoinverse + singularity-aware actuator reference
    ↓
end-effector NMPC
state:
[p, v, q, omega, alpha]

input:
[f1..f4, alpha_cmd1..4]
    ↓
first-order servo dynamics
    ↓
actual alpha
    ↓
analytical rotor wrench
    ↓
rigid MuJoCo body
    ↓
state feedback
```

The prediction state remains:

\[
x=
[p^W,v^W,q_{WB},\omega^B,\alpha]
\]

with:

\[
n_x=17.
\]

The physical control input remains:

\[
u=
[f_1,f_2,f_3,f_4,\alpha_{c1},\alpha_{c2},\alpha_{c3},\alpha_{c4}]
\]

with:

\[
n_u=8.
\]

---

# 4. End-effector geometry

The current end-effector is rigidly attached to the body.

Selected transform:

\[
r_{BE}^{B}
=
\begin{bmatrix}
0\\
0\\
0.15
\end{bmatrix}
\text{ m}
\]

and:

\[
q_{BE}
=
[1,0,0,0].
\]

Therefore:

- the tool origin is `150 mm` along \(+Z_B\),
- the tool axes are aligned with the body axes,
- there is no wrist,
- there are no additional dynamic tool states.

The `0.15 m` offset is a project design choice.

---

# 5. End-effector forward kinematics

The rigid tool state is derived from the body state.

Position:

\[
p_E^W
=
p_B^W
+
R_{WB}r.
\]

Velocity:

\[
v_E^W
=
v_B^W
+
R_{WB}
\left(
\omega_B\times r
\right).
\]

Orientation:

\[
q_{WE}
=
q_{WB}\otimes q_{BE}.
\]

Angular velocity:

\[
\omega_E
=
R_{BE}^T\omega_B.
\]

For the current aligned tool:

\[
R_{BE}=I
\]

and therefore:

\[
\omega_E=\omega_B.
\]

---

# 6. Desired tool motion to required body motion

The controller still predicts body dynamics.

A desired tool trajectory is converted into the required body trajectory.

Orientation:

\[
q_{WB}
=
q_{WE}\otimes q_{EB}.
\]

Angular velocity:

\[
\omega_B
=
R_{BE}\omega_E.
\]

Angular acceleration:

\[
\dot{\omega}_B
=
R_{BE}\dot{\omega}_E.
\]

Body position:

\[
p_B^W
=
p_E^W
-
R_{WB}r.
\]

Body velocity:

\[
v_B^W
=
v_E^W
-
R_{WB}
\left(
\omega_B\times r
\right).
\]

Body acceleration:

\[
a_B^W
=
a_E^W
-
R_{WB}
\left[
\dot{\omega}_B\times r
+
\omega_B\times
\left(
\omega_B\times r
\right)
\right].
\]

The acceleration term contains both:

- tangential acceleration,
- centripetal acceleration.

This is necessary for fixed-pivot rotation.

---

# 7. Fixed-pivot pitch geometry

For the current tool offset:

\[
r=
[0,0,0.15]^T
\]

and a fixed tool target:

\[
p_E^W=
[0,0,1.5]^T,
\]

a pure pitch rotation \(\theta\) requires the body CoG to move according to:

\[
p_B^W
=
\begin{bmatrix}
-0.15\sin\theta\\
0\\
1.5-0.15\cos\theta
\end{bmatrix}.
\]

The body therefore travels on a circle of radius:

\[
0.15\text{ m}.
\]

The diameter of that CoG orbit is:

\[
0.30\text{ m}.
\]

This is the geometric basis of the fixed-end-effector benchmark.

---

# 8. End-effector NMPC cost

The current stage residual is:

\[
e=
\begin{bmatrix}
p_E-p_{E,r}\\
v_E-v_{E,r}\\
V(q_{WE,r}^*\otimes q_{WE})\\
\omega_E-\omega_{E,r}\\
\alpha-\alpha_r\\
f-f_r\\
\alpha_c-\alpha
\end{bmatrix}.
\]

The preferred current translation weights are:

\[
Q_{p,E}
=
[600,600,800].
\]

\[
Q_{v,E}
=
[30,30,30].
\]

Other weights remain:

\[
Q_q
=
[300,300,600]
\]

\[
Q_\omega
=
[5,5,5]
\]

\[
Q_\alpha
=
2
\]

\[
R_f
=
2.
\]

The servo-motion regularization is:

\[
R_{\Delta\alpha}
\]

and is discussed extensively later in this document.

---

# 9. First free-space end-effector result

The initial end-effector-centric closed-loop test completed successfully.

Approximate result:

```text
duration:
30.01 s

control updates:
3001

solver failures:
0

deadline misses:
0

tool position RMS:
0.9598 cm

tool position max:
2.1485 cm

velocity RMS:
0.6939 cm/s

attitude RMS:
1.434°

attitude max:
3.161°

mean solve time:
0.798 ms
```

This established that the tool-space formulation itself was viable.

---

# 10. First generic large-angle showcase

A more aggressive visual showcase attempted:

- pitch,
- roll,
- yaw,
- diagonal-axis rotations,

while keeping the tool point fixed.

The solver remained numerically healthy, but the tracking was unacceptable.

Approximate result:

```text
tool position RMS:
4.736 cm

tool position max:
13.118 cm

attitude RMS:
141.606°

attitude max:
179.961°
```

This experiment was rejected as a useful benchmark because it combined too many large-angle effects before the single-axis behavior was understood.

The project therefore moved to a controlled paper-style pitch experiment.

---

# 11. Paper-style fixed-pivot pitch benchmark

The current benchmark trajectory is:

```text
0–2 s:
hold 0°

2–7 s:
0° → +180°
+36°/s

7–10 s:
hold +180°

10–20 s:
+180° → -180°
-36°/s
one complete physical 360° rotation

20–23 s:
hold -180°

23–28 s:
-180° → 0°
+36°/s

28–31 s:
hold 0°
```

The tool target remains:

\[
p_E^W=
[0,0,1.5]^T.
\]

This linear trajectory remains the preferred benchmark.

---

# 12. Singular / low-thrust actuator-reference behavior

The reference allocator uses:

\[
w=Az
\]

with pseudoinverse:

\[
z_r=A^\dagger w_r.
\]

A low-thrust singularity refinement is then applied.

Current nominal low-thrust threshold:

\[
1.5\text{ N}.
\]

The refinement:

1. detects a low-thrust candidate rotor,
2. adjusts one rotor reference,
3. fixes that rotor's virtual pair,
4. solves the remaining six virtual components from a reduced \(6\times6\) system,
5. verifies conditioning and wrench residual.

The result is only a **coarse actuator reference**.

The NMPC remains free to deviate from it.

Offline analysis showed that the reference can still contain large numerical changes near the threshold.

This is consistent with the idea that the reference layer is not expected to replace the full dynamic optimizer.

---

# 13. Original linear fixed-pivot result

With the original tool-space weights and direct physical servo-angle tracking:

\[
\alpha-\alpha_r,
\]

the linear trajectory produced approximately:

```text
tool position RMS:
5.171 cm

tool position max:
9.005 cm

attitude RMS:
3.661°

attitude max:
9.829°

solver failures:
0

deadline misses:
0
```

This proved that the full 360° fixed-pivot pitch maneuver was already feasible.

The main remaining visible issue was the approximately `5 cm` tool drift.

---

# 14. Trajectory-shaping experiments

Several trajectory variants were tested.

## 14.1 Full quintic angle profile

Using:

\[
s(r)=10r^3-15r^4+6r^5
\]

with the same total segment times caused a peak angular rate of approximately:

\[
67.5^\circ/\text{s}.
\]

Result:

```text
tool RMS:
6.214 cm

tool max:
14.400 cm

attitude RMS:
3.940°

attitude max:
14.214°
```

This was worse.

The smoother endpoints did not compensate for the much higher mid-segment rate.

---

## 14.2 Bounded-rate smooth profile

A ramp-cruise-ramp profile was then tested.

The plateau rate became approximately:

\[
40^\circ/\text{s}.
\]

Result:

```text
tool RMS:
5.230 cm

tool max:
8.793 cm

attitude RMS:
6.052°

attitude max:
20.839°
```

The tool-position behavior was acceptable, but attitude behavior near inversion became much worse.

This profile was also rejected as the main benchmark.

---

# 15. Periodic servo-angle-cost experiment

A periodic direction residual was tested:

\[
\sin\alpha-\sin\alpha_r
\]

and:

\[
\cos\alpha-\cos\alpha_r.
\]

The motivation was to remove the artificial numerical difference between representations near:

\[
+\pi
\]

and:

\[
-\pi.
\]

However, this removed useful information about the real mechanical servo branch.

Result:

```text
tool RMS:
5.373 cm

attitude RMS:
4.012°
```

which was worse than direct angle tracking.

The conclusion was:

\[
\boxed{
\text{servo angle must remain a non-periodic mechanical state in the NMPC cost}
}
\]

and the project returned to:

\[
\alpha-\alpha_r.
\]

---

# 16. Stronger end-effector translation weights

The first successful tuning change was to increase only the end-effector translation weights.

Original:

\[
Q_{p,E}
=
[300,300,400]
\]

\[
Q_{v,E}
=
[10,10,10].
\]

New:

\[
Q_{p,E}
=
[600,600,800]
\]

\[
Q_{v,E}
=
[30,30,30].
\]

Everything else was unchanged.

With the original linear trajectory this produced:

```text
tool RMS:
3.785 cm

tool max:
6.456 cm

attitude RMS:
3.617°

attitude max:
9.912°

mean solve:
1.072 ms
```

This reduced tool RMS by approximately:

\[
27\%.
\]

The tool-position error was now much more clearly localized to the pitch-plane X direction.

---

# 17. Axis-by-axis error after stronger tool weighting

The reconstructed tool-position RMS components were approximately:

\[
e_{x,\mathrm{RMS}}
=
3.761\text{ cm}
\]

\[
e_{y,\mathrm{RMS}}
=
0.297\text{ cm}
\]

\[
e_{z,\mathrm{RMS}}
=
0.310\text{ cm}.
\]

Therefore almost all remaining error was in:

\[
\boxed{X}
\]

during pitch.

The controller was not broadly inaccurate in 3-D.

The remaining issue was strongly directional.

---

# 18. X-error diagnosis

The next analysis compared:

- desired CoG X motion,
- actual CoG X motion,
- reconstructed tool X position,
- pitch rate,
- servo motion,
- NMPC weighted cost terms.

A key result was that the X error behaved more like a **rate-dependent steady offset** than an accumulating lag.

During the constant-rate portions of the benchmark, the tool X error was approximately:

```text
+36°/s entry:
-4.19 cm

-36°/s full rotation:
+4.54 cm

+36°/s exit:
-4.51 cm
```

The sign changed with rotation direction.

A fit gave approximately:

\[
e_{x,E}
\approx
0.00095
-
0.001235\,
\dot{\theta}_{\deg/s}
\]

with:

\[
R^2\approx0.986.
\]

This showed that the residual was strongly coupled to rotational rate.

---

# 19. Cost contribution analysis

Using the then-current OCP weights, approximate mean weighted contributions during the moving portions were:

```text
tool position       ≈ 1.21
tool velocity       ≈ 0.02
attitude            ≈ 0.33
angular rate        ≈ 0.06

alpha reference     ≈ 2.00
thrust reference    ≈ 1.81
alpha_cmd-alpha     ≈ 2.28
```

The important observation was that:

\[
\alpha_c-\alpha
\]

was one of the largest cost contributions.

This directly motivated investigation of:

\[
R_{\Delta\alpha}.
\]

---

# 20. What \(R_{\Delta\alpha}\) means

The servo model is:

\[
\dot{\alpha}
=
\frac{\alpha_c-\alpha}{\tau}.
\]

The NMPC includes:

\[
R_{\Delta\alpha}
\|\alpha_c-\alpha\|^2.
\]

Rearranging the servo equation:

\[
\alpha_c-\alpha
=
\tau\dot{\alpha}.
\]

Therefore:

\[
R_{\Delta\alpha}
\|\alpha_c-\alpha\|^2
=
R_{\Delta\alpha}\tau^2
\|\dot{\alpha}\|^2.
\]

So, for this first-order actuator model:

\[
\boxed{
R_{\Delta\alpha}
\text{ behaves approximately like a servo-speed regularization weight}
}
\]

A larger value says:

> avoid moving the servo aggressively.

A smaller value says:

> servo motion is cheaper; use it if it improves tracking.

This interpretation became central to understanding the fixed-pivot X bias.

---

# 21. Why a large \(R_{\Delta\alpha}\) created tracking bias

The fixed-pivot maneuver requires:

- continuous body rotation,
- continuous CoG translation around the tool,
- continuously changing rotor force directions.

To produce the required wrench quickly, the optimizer sometimes needs:

\[
|\alpha_c-\alpha|
\]

to become larger.

At high:

\[
R_{\Delta\alpha},
\]

that is expensive.

The NMPC therefore finds a compromise:

```text
move the servos less aggressively
+
accept some persistent end-effector error
```

The approximately `4–5 cm` X offset at constant pitch rate was one expression of that compromise.

Reducing the servo-motion penalty allowed the optimizer to spend more actuator effort to improve:

- end-effector position,
- attitude tracking.

---

# 22. Controlled \(R_{\Delta\alpha}\) sweep

The project performed a controlled sweep while keeping the following fixed:

```text
trajectory:
linear 36°/s paper-style pitch

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

servo tau:
0.048 s

horizon:
20 stages / 2.0 s

control:
100 Hz

physics:
500 Hz
```

Only:

\[
R_{\Delta\alpha}
\]

changed.

---

# 23. \(R_{\Delta\alpha}=250\)

This was the original paper-inspired baseline.

Result:

```text
tool RMS:
3.785 cm

tool max:
6.456 cm

attitude RMS:
3.617°

attitude max:
9.912°
```

Approximate constant-rate X bias:

```text
+36°/s entry:
-4.19 cm

-36°/s full:
+4.54 cm

+36°/s exit:
-4.51 cm
```

Approximate servo command-state separation:

```text
RMS:
2.74°

max:
9.89°
```

---

# 24. \(R_{\Delta\alpha}=150\)

Result:

```text
tool RMS:
3.295 cm

tool max:
5.739 cm

attitude RMS:
2.666°

attitude max:
7.931°
```

The X bias reduced to approximately:

```text
+36°/s entry:
-3.66 cm

-36°/s full:
+3.94 cm

+36°/s exit:
-3.90 cm
```

Approximate servo separation:

```text
RMS:
2.82°

max:
10.92°
```

This strongly supported the servo-regularization hypothesis.

---

# 25. \(R_{\Delta\alpha}=100\)

Result:

```text
tool RMS:
2.922 cm

tool max:
5.119 cm

attitude RMS:
2.068°

attitude max:
6.596°
```

Approximate X bias:

```text
+36°/s entry:
-3.25 cm

-36°/s full:
+3.49 cm

+36°/s exit:
-3.45 cm
```

Approximate servo separation:

```text
RMS:
2.92°

max:
12.33°
```

The improvement remained consistent.

---

# 26. \(R_{\Delta\alpha}=50\)

This is the **current selected baseline**.

Result:

```text
tool position RMS:
2.3504 cm

tool position max:
4.0752 cm

attitude RMS:
1.3526°

attitude max:
4.8330°

solver failures:
0

10 ms deadline misses:
0
```

Solver timing:

```text
mean:
1.3553 ms

p95:
1.9280 ms

max:
2.9160 ms
```

Approximate X bias:

```text
+36°/s entry:
-2.64 cm

-36°/s full:
+2.80 cm

+36°/s exit:
-2.76 cm
```

Approximate overall X RMS:

\[
2.33\text{ cm}.
\]

Approximate servo command-state separation:

```text
RMS:
3.19°

max:
16.39°
```

This produced a very strong improvement while still retaining a meaningful actuator-motion penalty.

---

# 27. \(R_{\Delta\alpha}=25\)

A lower value was tested to understand the trend.

Result:

```text
tool position RMS:
1.8795 cm

tool position max:
3.1608 cm

attitude RMS:
0.9430°

attitude max:
3.6634°

solver failures:
0

deadline misses:
0
```

Approximate X bias:

```text
+36°/s entry:
-2.14 cm

-36°/s full:
+2.24 cm

+36°/s exit:
-2.21 cm
```

Approximate servo separation:

```text
RMS:
3.57°

max:
22.33°
```

The tracking was better than at \(R=50\).

However, actuator aggressiveness increased substantially.

The first-order model implied very high instantaneous servo rates in some regions.

Therefore the project deliberately chose **not** to continue reducing \(R\) based only on nominal tracking performance.

---

# 28. Consolidated \(R_{\Delta\alpha}\) sweep

| \(R_{\Delta\alpha}\) | Tool RMS | Tool max | Attitude RMS | Attitude max |
|---:|---:|---:|---:|---:|
| `250` | `3.785 cm` | `6.456 cm` | `3.617°` | `9.912°` |
| `150` | `3.295 cm` | `5.739 cm` | `2.666°` | `7.931°` |
| `100` | `2.922 cm` | `5.119 cm` | `2.068°` | `6.596°` |
| **`50`** | **`2.350 cm`** | **`4.075 cm`** | **`1.353°`** | **`4.833°`** |
| `25` | `1.880 cm` | `3.161 cm` | `0.943°` | `3.663°` |

The sweep is monotonic in nominal tracking quality.

However, lower \(R\) also increases actuator aggressiveness.

---

# 29. Why \(R_{\Delta\alpha}=50\) is selected instead of 25

The project is now explicitly choosing:

\[
\boxed{
R_{\Delta\alpha}=50
}
\]

as the preferred baseline.

This is **not** because \(R=50\) gives the numerically lowest tracking error.

It does not.

\(R=25\) tracks better in the current nominal simulation.

The reason is engineering conservatism regarding the actuator model.

At \(R=25\), the optimizer increasingly exploits:

\[
\alpha_c-\alpha
\]

to demand rapid servo motion.

The current servo model contains:

\[
\dot{\alpha}
=
\frac{\alpha_c-\alpha}{0.048}
\]

but does not yet contain a hard or experimentally identified:

\[
|\dot{\alpha}|
\le
\dot{\alpha}_{\max}.
\]

Therefore the nominal simulation can reward actuator behavior that might not be realistic on hardware.

\(R=50\) is selected as a better compromise between:

- tracking,
- actuator regularization,
- hardware plausibility,
- and numerical robustness.

This selection should be revisited once real servo speed data or a rate-limited model is available.

---

# 30. Improvement from the original servo regularization to the selected baseline

Comparing:

\[
R_{\Delta\alpha}=250
\]

with:

\[
R_{\Delta\alpha}=50,
\]

tool RMS improved from:

\[
3.785\text{ cm}
\]

to:

\[
2.350\text{ cm}.
\]

That is approximately:

\[
38\%
\]

lower.

Attitude RMS improved from:

\[
3.617^\circ
\]

to:

\[
1.353^\circ.
\]

That is approximately:

\[
63\%
\]

lower.

Therefore the original \(R=250\) was clearly too restrictive for this large-angle end-effector maneuver in the current architecture.

---

# 31. Important actuator-model lesson

The \(R\) sweep reveals an important modeling issue.

At the moment:

\[
R_{\Delta\alpha}
\]

does two jobs:

1. it regularizes control smoothness,
2. it partially compensates for the fact that the servo model lacks an explicit rate limit.

Those should ideally be separate concepts.

A more physically complete future servo model should include something like:

\[
|\dot{\alpha}|
\le
\dot{\alpha}_{\max}
\]

or an experimentally identified actuator model with:

- rate limit,
- torque/current limitations,
- deadband,
- backlash,
- nonlinear bandwidth,
- command saturation.

After that, \(R_{\Delta\alpha}\) can be tuned primarily as a control-quality preference rather than acting as a hidden safety mechanism.

---

# 32. Current interpretation of the remaining X bias

Even with \(R=50\), some rate-dependent X offset remains.

The fact that:

- the error changes sign with pitch rate,
- the offset stays approximately constant during constant-rate motion,
- and it decreases when servo motion becomes cheaper,

strongly indicates that it is a dynamic optimization tradeoff rather than a static geometry bug.

The problem is no longer:

```text
"the controller is wrong"
```

but rather:

```text
"the optimizer still trades some tool translation against actuator/reference costs"
```

The residual is now much smaller and should be investigated only if further reduction is needed for the manipulation milestone.

---

# 33. Current preferred pitch benchmark

The preserved benchmark remains:

```text
0–2 s:
hold 0°

2–7 s:
0 → +180°
+36°/s

7–10 s:
hold +180°

10–20 s:
+180 → -180°
-36°/s

20–23 s:
hold -180°

23–28 s:
-180 → 0°
+36°/s

28–31 s:
hold 0°
```

This exact trajectory should continue to be used for clean comparisons.

---

# 34. Current preferred NMPC weights

The selected current end-effector configuration is:

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

This is now the primary fixed-pivot baseline.

---

# 35. Current selected fixed-pivot performance

With:

\[
R_{\Delta\alpha}=50,
\]

the selected benchmark performance is:

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

Tracking:

```text
tool RMS:
2.350 cm

tool max:
4.075 cm

attitude RMS:
1.353°

attitude max:
4.833°
```

Solver:

```text
mean:
1.355 ms

p95:
1.928 ms

max:
2.916 ms
```

Actuation:

```text
max thrust:
15.999 N

max |actual alpha|:
180.482°

max |alpha_cmd|:
180.482°
```

This is the current performance reference for future pitch-related changes.

---

# 36. Computational status

The harder end-effector benchmark remains comfortably real-time.

The 100 Hz deadline is:

```text
10.0 ms
```

while the selected \(R=50\) run has:

```text
mean:
1.355 ms

p95:
1.928 ms

max:
2.916 ms
```

No solve missed the deadline.

The end-effector controller therefore still has significant computational margin.

---

# 37. Visual servo-angle representation

The rigid plant intentionally contains no physical MuJoCo tilt joints.

This is still the correct dynamics architecture.

To make the servo angles visible, a separate visualization layer was prepared.

The approach is:

```text
actual alpha states
    ↓
viewer-only rotor transform
    ↓
tilting visual rotor / motor geometry
```

The visual representation does not feed back into physics.

This avoids reintroducing hidden mechanical DoFs.

A temporary overlay can draw:

- blue tilted discs,
- direction markers,

but the longer-term plan is to use the actual CAD motor/propeller assemblies.

---

# 38. CAD / simulation visual-model direction

The vehicle is now being redesigned visually in Onshape.

The goal is to make the simulation look like a realistic tiltable quadrotor while preserving the validated control geometry.

Important current dimensions:

```text
center to rotor axis:
275 mm

propeller radius:
114 mm

propeller diameter:
228 mm

opposite rotor-center spacing:
550 mm

tool offset:
150 mm
```

The current MuJoCo central body visual is only a simple approximation:

```text
160 × 160 × 50 mm
```

and should not be treated as a paper-derived chassis dimension.

The arm visual radius is also only a simplified rendering parameter.

---

# 39. Recommended CAD hierarchy

The CAD should not be one fused part.

A useful structure is:

```text
Drone assembly
│
├── chassis
├── arm 1
│   └── tilt module 1
│       ├── motor
│       └── propeller
├── arm 2
│   └── tilt module 2
│       ├── motor
│       └── propeller
├── arm 3
│   └── tilt module 3
│       ├── motor
│       └── propeller
├── arm 4
│   └── tilt module 4
│       ├── motor
│       └── propeller
└── end-effector
```

The moving tilt modules should remain separable from the chassis.

This will allow them to follow:

\[
\alpha_1,\alpha_2,\alpha_3,\alpha_4
\]

visually in MuJoCo.

---

# 40. Visual CAD motion without changing physics

The preferred simulation architecture is:

```text
NMPC
    ↓
alpha_cmd
    ↓
first-order servo
    ↓
actual alpha
    ├──────────────→ wrench model / real physics
    │
    └──────────────→ visual CAD transform
```

The imported CAD motor/propeller modules should be treated as visual or kinematic objects.

They should **not** become physical MuJoCo hinge DoFs in the nominal plant.

This keeps:

\[
\boxed{
\alpha_i
}
\]

as the single source of truth.

---

# 41. Propeller visual rotation

Two different rotations can eventually be displayed:

## Tilt motion

\[
\alpha_i
\]

This is the real modeled servo state.

## Propeller spin

\[
\phi_i
\]

This can be maintained purely for visualization.

A simple visual model could use:

\[
\phi_i(t+\Delta t)
=
\phi_i(t)
+
\Omega_i\Delta t.
\]

Eventually, if a thrust coefficient is identified:

\[
f_i=k_t\Omega_i^2,
\]

the visual propeller RPM can be tied to actual modeled rotor speed.

Until then, propeller spin can remain an approximate visual effect.

The tilt angle is much more important for debugging than exact propeller RPM.

---

# 42. Propeller size currently required by the model

The current simulation uses:

\[
R_p=0.114\text{ m}.
\]

Therefore:

\[
D_p=0.228\text{ m}.
\]

This is approximately:

\[
9\text{ inches}.
\]

A visually appropriate CAD propeller is therefore approximately:

```text
9-inch diameter
```

and a `9×4.5` / `9045` style propeller is a good visual match.

Exact aerodynamic blade geometry is not currently used by the physics model.

The visual CAD primarily needs:

- correct diameter,
- correct hub axis,
- reasonable mass-independent geometry,
- clean rotation about its shaft.

---

# 43. Current validation status

## Validated

```text
rigid six-DoF plant
first-order servo mathematics
analytical rotor wrench
17-state CasADi model
MuJoCo/CasADi nominal consistency
body-centered NMPC
end-effector transform
tool-space reference conversion
end-effector NMPC cost
free-space tool tracking
fixed-pivot 360° pitch
paper-style linear benchmark
stronger tool translation weights
direct alpha tracking
R_delta_alpha sweep
100 Hz real-time feasibility
```

## Demonstrated but still subject to further refinement

```text
large-angle actuator branch handling
fixed-pivot X tracking
visual servo representation
CAD integration plan
```

## Not yet validated

```text
real servo rate capability
servo torque/current limit
motor thrust dynamics
physical propeller RPM model
hardware actuator identification
fixed-pivot roll
combined-axis tool manipulation
contact manipulation
external disturbance rejection
model mismatch robustness
state-estimation imperfections
real hardware deployment
RL policy
```

---

# 44. Rejected / deprioritized approaches

The following are not part of the current baseline:

```text
mechanical passive tilt hinges in MuJoCo
generic 43 s multi-axis spin showcase
full-quintic 67.5°/s pitch benchmark
bounded-rate 40°/s pitch benchmark
periodic sin/cos servo-angle tracking cost
R_delta_alpha = 250 for end-effector manipulation
immediate external-wrench observer
immediate offset-free augmentation
large robustness campaign before manipulation milestone
```

---

# 45. Current selected baseline configuration

For reproducibility, the current baseline is:

```text
Vehicle mass:
3.039 kg

Inertia:
diag(0.063, 0.062, 0.095) kg·m²

Rotor-center radius:
0.275 m

Propeller radius:
0.114 m

Maximum thrust:
23 N / rotor

Servo tau:
0.048 s

Tool offset:
[0, 0, 0.15] m

Tool relative orientation:
identity

Physics:
500 Hz

Controller:
100 Hz

Prediction:
20 stages / 2.0 s

Pitch benchmark:
linear 36°/s paper-style sequence

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

Alpha tracking:
direct alpha - alpha_ref

Thrust bounds:
0–23 N

Alpha state bounds:
±3.15 rad

Alpha command bounds:
±3.15 rad
```

This is the configuration that should be preserved before the next development step.

---

# 46. Current selected quantitative milestone

The current preferred benchmark result is:

```text
R_delta_alpha:
50

tool position RMS:
2.350 cm

tool position max:
4.075 cm

attitude RMS:
1.353°

attitude max:
4.833°

mean solve:
1.355 ms

p95 solve:
1.928 ms

max solve:
2.916 ms

solver failures:
0

10 ms deadline misses:
0
```

The lower-\(R\) `25` result is retained as an important experimental reference:

```text
tool RMS:
1.880 cm

attitude RMS:
0.943°
```

but is not selected as the baseline because actuator aggressiveness is more questionable without a real servo-rate limit.

---

# 47. Main engineering lessons from the \(R\) sweep

## 47.1 Cost weights are not abstract tuning numbers

They determine which physical errors the optimizer is willing to tolerate.

A large actuator penalty can directly create position or attitude bias.

## 47.2 Servo regularization can look like tracking lag

The X error initially appeared to be a geometric or phase-lag problem.

Detailed analysis showed it was strongly linked to servo-motion cost.

## 47.3 The sign of the bias matters

The X offset reversed with pitch direction.

That was a strong clue that the error was rate-related.

## 47.4 Lower \(R\) improved both translation and attitude

This shows that the old servo penalty was constraining overall control authority, not only position tracking.

## 47.5 Lower nominal error is not automatically the best engineering choice

The `R=25` controller performs better numerically.

The project still chooses `R=50` because the actuator model is incomplete.

This is a deliberate modeling decision rather than a failure to optimize the simulation score.

---

# 48. What should be done before lowering \(R\) further

The next actuator-model improvement should be to identify or impose realistic servo dynamics beyond a simple first-order response.

At minimum, investigate:

\[
|\dot{\alpha}|
\le
\dot{\alpha}_{\max}.
\]

The existing first-order equation implies:

\[
|\alpha_c-\alpha|
\le
\tau\dot{\alpha}_{\max}
\]

if a hard rate limit is known.

More complete future models may also include:

```text
maximum angular speed
torque/current limit
command deadband
backlash
saturation
bandwidth variation
load-dependent response
```

Once these are known, the choice of \(R_{\Delta\alpha}\) can be revisited with much more confidence.

---

# 49. Recommended next project steps

The recommended development order is now:

```text
1. preserve and commit the R_delta_alpha = 50 baseline

2. finish the CAD visual model while preserving:
   - 275 mm rotor-center radius
   - 228 mm prop diameter
   - separable tilt modules

3. integrate visual CAD tilt motion using actual alpha states

4. run the equivalent fixed-pivot roll benchmark

5. compare pitch vs roll behavior

6. create one mixed-axis end-effector trajectory

7. perform a basic contact-ready manipulation test

8. decide whether the remaining nominal manipulation quality is sufficient
   for the transition to RL

9. later:
   identify real servo rate / thrust dynamics

10. later:
    robustness, disturbance observer, offset-free compensation, sensor noise,
    model mismatch, and hardware deployment
```

---

# 50. Current project milestone statement

The project has moved through the following stages:

```text
static allocation
    ↓
servo modeling
    ↓
rigid-plant validation
    ↓
body-centered NMPC
    ↓
end-effector-centric NMPC
    ↓
large-angle fixed-pivot motion
    ↓
servo-regularization diagnosis
    ↓
current selected manipulation baseline
```

The current selected large-angle manipulation result is:

\[
\boxed{
2.35\text{ cm tool-position RMS}
}
\]

and:

\[
\boxed{
1.35^\circ\text{ attitude RMS}
}
\]

during a complete paper-style 360° pitch maneuver, with:

```text
0 solver failures
0 control deadline misses
```

at:

```text
100 Hz NMPC
500 Hz physics
```

This is a substantial improvement over the original fixed-pivot baseline and demonstrates that the architecture can perform meaningful end-effector-centric large-angle motion without adding disturbance estimators or additional model-error compensation.

---

# 51. Final conclusion

The most important recent finding is not merely that one tuning value is better than another.

The important finding is the **mechanism** behind the improvement.

The original controller heavily penalized:

\[
\alpha_c-\alpha.
\]

Because:

\[
\alpha_c-\alpha
=
\tau\dot{\alpha},
\]

this was effectively a strong penalty against servo motion.

For ordinary mild flight that may be reasonable.

For a fixed-pivot 360° manipulation maneuver, it caused the optimizer to sacrifice tool position and attitude in order to avoid using the tilt mechanism aggressively.

Reducing:

\[
R_{\Delta\alpha}
\]

systematically reduced:

- the rate-dependent X bias,
- tool-position RMS,
- attitude RMS.

The sweep showed:

\[
250
\rightarrow
150
\rightarrow
100
\rightarrow
50
\rightarrow
25
\]

with continuously improving nominal tracking.

However, the project deliberately selects:

\[
\boxed{
R_{\Delta\alpha}=50
}
\]

rather than `25` because the current servo model has no verified hard rate limit.

That choice reflects the current engineering objective:

\[
\boxed{
\text{good manipulation performance without relying excessively on unverified actuator capability}
}
\]

The next meaningful milestone is therefore not another arbitrary \(R\) reduction.

It is to preserve this controller, improve the physical/visual vehicle representation, test roll and mixed-axis tool motion, and then decide when the manipulation stack is mature enough to move into RL.
