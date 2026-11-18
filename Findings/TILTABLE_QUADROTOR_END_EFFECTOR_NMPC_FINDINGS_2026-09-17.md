# Tiltable Quadrotor — End-Effector-Centric NMPC, Fixed-Pivot Rotation, and Controller-Tuning Findings

**Project:** `tiltable_quadrotor_rl`  
**Repository:** `~/Projects/tiltable_quadrotor_rl`  
**Date:** 2026-09-17  
**Scope:** Work completed after `TILTABLE_QUADROTOR_NMPC_IMPLEMENTATION_AND_CLOSED_LOOP_FINDINGS_2026-09-16.md`

---

# 1. Purpose of this document

The previous project document established that the nominal rigid-plant NMPC architecture was working correctly.

By the end of 2026-09-16, the project already had:

- a rigid six-DoF MuJoCo plant,
- explicit first-order tilt-servo states,
- a validated 17-state / 8-input CasADi model,
- instantaneous and finite-time MuJoCo/CasADi agreement,
- a compiled acados NMPC,
- a general body/CoG-centered trajectory-reference system,
- and a successful 30-second multi-axis closed-loop flight with millimeter-level nominal position tracking.

The previous document therefore ended with a recommendation to move toward deliberate model mismatch and disturbance robustness testing.

The development priority changed after that document.

The current priority is now:

```text
1. finish the end-effector / aerial-manipulation formulation,
2. demonstrate that it works for meaningful large-angle maneuvers,
3. then move toward RL,
4. add disturbance observers, model-error compensation, and broader robustness
   machinery later if the project needs them.
```

This document records the work completed after that decision.

The major new milestone is:

\[
\boxed{
\text{The same 17-state NMPC can now track a rigid end-effector directly in tool space.}
}
\]

More specifically, the project now supports:

- a fixed rigid tool offset from the vehicle CoG,
- tool-space pose and velocity calculation,
- conversion of desired end-effector motion into the required body/CoG motion,
- tool-space NMPC cost terms,
- a dedicated end-effector controller and trajectory interface,
- a collidable end-effector tip in MuJoCo,
- successful free-space end-effector trajectory tracking,
- fixed-world-point pitch rotations through the inverted configuration,
- a paper-style 360° fixed-pivot pitch experiment,
- detailed actuator/singularity telemetry,
- several trajectory-shaping experiments,
- one rejected periodic servo-cost experiment,
- and a substantially improved fixed-pivot tracking configuration.

The strongest current fixed-pivot result is:

```text
tool position RMS:
3.785 cm

tool position max:
6.456 cm

attitude RMS:
3.617 deg

attitude max:
9.912 deg

solver failures:
0

10 ms deadline misses:
0
```

for a full paper-style pitch sequence that includes a complete 360° rotation while attempting to hold the end-effector origin fixed in world.

---

# 2. Starting point from the previous document

The previous body-centered controller was already highly successful in the nominal simulation.

Its first general closed-loop run produced approximately:

```text
position RMS:
2.596 mm

position max:
5.155 mm

attitude RMS:
0.362 deg

attitude max:
0.658 deg

mean solve time:
0.830 ms

solver failures:
0

10 ms deadline misses:
0
```

That test was intentionally mild in actuator usage:

```text
max thrust:
7.58 N

max commanded tilt:
3.30 deg
```

The next challenge was therefore not simply "make the vehicle fly."

The new challenge was:

> Can the controller regulate a point offset from the CoG while the body undergoes large rotations and the tilting rotors approach the difficult large-angle / low-thrust regions?

That is a significantly harder problem.

---

# 3. Development-priority decision

A deliberate project decision was made not to introduce the following yet:

- external-wrench observer,
- offset-free disturbance augmentation,
- integral disturbance compensation,
- large model-mismatch campaigns,
- large parameter sweeps,
- additional robustness machinery.

The reference paper includes additional mechanisms of this type, and they may become useful later.

However, they would make it harder to answer the immediate engineering question:

> Does the end-effector-centric formulation itself work on the current validated nominal plant?

The work in this document therefore keeps the plant and actuator model unchanged and focuses on the end-effector geometry, reference generation, cost formulation, and trajectory behavior.

---

# 4. Fixed end-effector model

A rigid tool transform was added between the vehicle body frame \(B\) and the end-effector frame \(E\).

The selected transform is:

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
\begin{bmatrix}
1\\
0\\
0\\
0
\end{bmatrix}.
\]

Therefore:

- the tool origin is `0.15 m` along \(+Z_B\) from the vehicle CoG,
- the tool axes are aligned with the body axes,
- there is no articulated wrist,
- there is no additional mechanical joint,
- the tool is part of the same rigid vehicle body.

The value `0.15 m` is a project design choice.

It should not be presented as a measured value from the reference paper.

---

# 5. MuJoCo tool representation

The rigid MJCF was extended with a fixed end-effector site and simple tool geometry.

The current conceptual structure is:

```text
rigid vehicle body
    |
    +-- end_effector site at [0, 0, 0.15]
    |
    +-- visual shaft
    |
    +-- spherical tool tip
```

No new body or joint was introduced.

This is important because the earlier project already rejected hidden mechanical tilt DoFs that made the nominal plant inconsistent with the NMPC model.

The end-effector must not reintroduce that class of problem.

The tool tip is collidable.

This prepares the plant for later contact/manipulation work without changing the current free-flight rigid-body equations.

The current tool origin coincides with the center of the spherical tip.

---

# 6. End-effector forward kinematics

A new kinematics layer computes the tool state from the body state.

For the fixed tool offset:

\[
r=r_{BE}^{B},
\]

the end-effector world position is:

\[
p_E^W
=
p_B^W
+
R_{WB}r.
\]

The end-effector world velocity is:

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

The end-effector orientation is:

\[
q_{WE}
=
q_{WB}\otimes q_{BE}.
\]

The angular velocity expressed in the tool frame is:

\[
\omega_E
=
R_{BE}^T\omega_B.
\]

For the current aligned tool:

\[
R_{BE}=I,
\]

so:

\[
\omega_E=\omega_B.
\]

The implementation provides both:

- NumPy versions for runtime/reference calculations,
- CasADi symbolic versions for use directly inside the NMPC cost.

---

# 7. Converting desired tool motion into required body motion

The NMPC state is still the body/CoG state.

The project did **not** add tool position or tool orientation as independent dynamic states.

Instead, a desired end-effector trajectory is converted into the corresponding body/CoG trajectory.

Given desired:

```text
p_E
v_E
a_E
q_WE
omega_E
omega_dot_E
```

the required body orientation is:

\[
q_{WB}
=
q_{WE}\otimes q_{EB}.
\]

The body angular velocity is:

\[
\omega_B
=
R_{BE}\omega_E.
\]

The body angular acceleration is:

\[
\dot{\omega}_B
=
R_{BE}\dot{\omega}_E.
\]

The body position is:

\[
p_B^W
=
p_E^W
-
R_{WB}r.
\]

The body velocity is:

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

The body acceleration is:

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

The acceleration correction contains both:

- tangential acceleration,
- centripetal acceleration.

This term is essential for fixed-pivot rotations.

If the desired tool point is fixed while the vehicle rotates, the CoG must continuously accelerate around the tool point.

---

# 8. Geometric meaning of a fixed end-effector during pitch

For the current offset:

\[
r=
\begin{bmatrix}
0\\
0\\
0.15
\end{bmatrix},
\]

suppose the end-effector is fixed at:

\[
p_E^W=
\begin{bmatrix}
0\\
0\\
1.5
\end{bmatrix}
\]

and the body executes a pure pitch rotation \(\theta\).

Then the ideal body/CoG position follows:

\[
p_B^W
=
\begin{bmatrix}
-0.15\sin\theta\\
0\\
1.5-0.15\cos\theta
\end{bmatrix}.
\]

Therefore the CoG moves on a circle of radius:

\[
0.15\text{ m},
\]

or diameter:

\[
0.30\text{ m}.
\]

This is the correct physical behavior.

The vehicle should visibly orbit around the fixed tool point.

The end-effector staying fixed does **not** mean the vehicle body remains fixed.

---

# 9. Important limitation of the rigid tool

Because the tool is rigidly attached to the vehicle:

\[
q_{WE}
=
q_{WB}\otimes q_{BE}.
\]

For the aligned current tool:

\[
q_{WE}=q_{WB}.
\]

Therefore the current system can do:

```text
fixed end-effector position
+
rotating vehicle/tool orientation
```

but it cannot independently hold both:

```text
fixed end-effector position
and
fixed end-effector orientation
```

while rotating the vehicle body.

That would require an articulated wrist or another independent tool orientation DoF.

This is a geometric limitation, not an NMPC defect.

---

# 10. New end-effector software architecture

The end-effector implementation introduced the following main modules:

```text
src/tiltable_quadrotor_rl/control/
    end_effector.py
    nmpc_effector_reference.py
    nmpc_effector_ocp.py
    nmpc_effector_controller.py
    nmpc_effector_trajectory.py
```

The rigid plant parameters and MJCF were also extended with the fixed tool transform.

The main responsibilities are:

## `end_effector.py`

Contains:

- rigid tool transform,
- quaternion utilities,
- NumPy body-to-tool kinematics,
- CasADi body-to-tool kinematics,
- tool-motion-to-body-motion conversion.

## `nmpc_effector_reference.py`

Contains the end-effector reference-point representation and converts a desired tool trajectory into:

- body reference motion,
- body feed-forward wrench,
- actuator reference,
- stagewise NMPC parameters.

## `nmpc_effector_ocp.py`

Keeps the same 17-state / 8-input dynamics but evaluates the primary tracking error at the end-effector.

## `nmpc_effector_controller.py`

Reuses the existing NMPC runtime architecture while constructing end-effector-centered horizons.

## `nmpc_effector_trajectory.py`

Provides concrete tool-space trajectories for closed-loop experiments.

The state dimension remains:

\[
n_x=17.
\]

The input dimension remains:

\[
n_u=8.
\]

No extra dynamic tool state was needed.

---

# 11. End-effector-centric NMPC cost

The initial end-effector OCP preserved the actuator portion of the body-centered controller but replaced the body pose/velocity tracking terms with end-effector terms.

The stage residual was:

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

The important architectural point is:

> The dynamics are body dynamics, but the controlled output is the end-effector.

The initial weights were inherited directly from the body-centered OCP:

```text
tool position:
[300, 300, 400]

tool velocity:
[10, 10, 10]

tool attitude:
[300, 300, 600]

tool angular velocity:
[5, 5, 5]

alpha reference:
2

thrust reference:
2

alpha_cmd - alpha:
250
```

These weights later became an important tuning topic.

---

# 12. End-effector solver generation

A dedicated acados solver was generated for the end-effector OCP.

It retained:

```text
N = 20
prediction horizon = 2.0 s
shooting interval = 0.1 s
controller update = 0.01 s
SQP_RTI
PARTIAL_CONDENSING_HPIPM
ERK
Gauss-Newton Hessian
```

The generated solver built successfully.

The current controller therefore still runs:

```text
MuJoCo:
500 Hz

NMPC:
100 Hz
```

with the end-effector transform evaluated inside the nonlinear least-squares cost.

---

# 13. Development-environment note: acados shared libraries

During one solver rebuild, C code generation and compilation succeeded but Python failed when loading the generated shared library:

```text
OSError:
libhpipm.so: cannot open shared object file
```

The problem was not the NMPC code.

The issue was that a new terminal did not contain the acados library path in the dynamic-loader environment.

The practical permanent options identified were:

```bash
export ACADOS_SOURCE_DIR="$HOME/Projects/acados"
```

in the shell startup configuration, and either:

```bash
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

or preferably registering:

```text
$ACADOS_SOURCE_DIR/lib
```

with the Linux dynamic linker using an `ld.so.conf.d` entry and `ldconfig`.

This is a tooling/environment issue rather than a controller issue, but it is worth recording because the generated solver can compile successfully and still fail only at dynamic loading.

---

# 14. First end-effector free-space closed-loop validation

The first full end-effector trajectory test ran for:

```text
30.010 s
3001 NMPC updates
```

and completed normally.

Results:

```text
solver failures:
0

10 ms deadline misses:
0

mean solve time:
0.7981 ms

p95 solve time:
0.8890 ms

max solve time:
2.8950 ms
```

End-effector tracking:

```text
position RMS:
0.009598 m

position max:
0.021485 m

velocity RMS:
0.006939 m/s

attitude RMS:
1.434221 deg

attitude max:
3.161289 deg
```

Actuator usage:

```text
max thrust:
8.207966 N

max |alpha_cmd|:
25.338006 deg
```

This was the first major end-effector milestone.

It demonstrated that:

\[
\boxed{
\text{tool-space reference}
\rightarrow
\text{body-space feed-forward}
\rightarrow
\text{same NMPC dynamics}
}
\]

works in closed loop.

The error was larger than the earlier body-centered test, which is expected because regulating an offset tool point couples translation and rotation.

---

# 15. Fixed-end-effector showcase concept

After the free-space test, a more visually demanding experiment was designed.

The goal was:

```text
hold the end-effector origin fixed in world
while the body rotates around it
```

with large rotations including full 360° motion.

The initial showcase attempted several rotation segments:

- pitch,
- roll,
- yaw,
- diagonal-axis rotation.

The tool position remained fixed as the desired world point.

This experiment was intended to look more like a robotic-arm pivot than a normal aerial trajectory.

---

# 16. First large-angle showcase failure

The first 43-second showcase completed without solver failures or deadline misses:

```text
duration:
43.000 s

control updates:
4300

solver failures:
0

deadline misses:
0

mean solve time:
1.1436 ms
```

However, the tracking result was clearly unacceptable:

```text
tool position RMS:
0.047358 m

tool position max:
0.131181 m

attitude RMS:
141.605742 deg

attitude max:
179.961436 deg
```

Actuator usage reached:

```text
max thrust:
16.261760 N

max |alpha|:
180.481705 deg

max |alpha_cmd|:
180.481705 deg
```

The solver remained numerically healthy, but the large-angle reference/control behavior did not.

This result showed an important distinction:

> A solver that keeps returning status 0 is not evidence that the physical closed-loop behavior is good.

The failure appeared once the maneuver moved into the difficult near-inverted / tilt-angle-branch region.

This motivated a more controlled paper-style experiment rather than continuing with a long multi-axis showcase.

---

# 17. Comparison with the reference paper

The reference paper investigated essentially the same class of problem:

> Effector-Centric NMPC of Tiltable-Multirotors for Offset-Free Omnidirectional Aerial Manipulation.

The paper contains a physical experiment involving:

```text
360° pitch and roll rotations about the end-effector
```

rather than merely translating the CoG.

The useful lessons extracted for this project were:

1. the large-angle maneuver is a valid target for this architecture,
2. the actuator-reference singularity near low thrust must be handled deliberately,
3. the paper uses a structured entry / full-cycle / exit motion rather than an arbitrary continuous showcase,
4. the paper's reference allocator includes explicit singularity refinement,
5. discontinuity can remain in the coarse actuator reference, with NMPC expected to absorb it,
6. additional disturbance/model-error compensation exists in the paper but is not required for the current nominal architecture study.

The project therefore moved from a generic "spin around the tool" showcase to a controlled paper-style pitch experiment.

---

# 18. Current singularity-aware actuator reference

The existing reference-allocation implementation already includes the relevant paper-style low-thrust refinement.

The nominal threshold is:

\[
f_{\text{threshold}}=1.5\text{ N}.
\]

The process is conceptually:

```text
desired body wrench
    ↓
A† wrench
    ↓
virtual rotor components
    ↓
identify low-thrust candidate rotor
    ↓
adjust one candidate rotor angle
    ↓
fix its two virtual components
    ↓
solve remaining six virtual components from a 6x6 system
    ↓
recover thrust and tilt references
```

The reduced solve is accepted only when the matrix conditioning and wrench residual are acceptable.

The result is a **reference**, not a hard control-allocation command.

The NMPC can deviate from it.

---

# 19. Offline paper-style reference diagnostic

Before rerunning the full closed loop, the paper-style pitch reference was analyzed offline.

For the linear pitch sequence, rotors 1 and 3 entered the raw low-thrust region during approximately:

```text
4.19–4.83 s
12.17–12.81 s
17.19–17.83 s
25.17–25.81 s
```

Rotors 2 and 4 did not enter the same low-thrust condition.

A representative transition occurred around:

```text
t = 4.84 s
pitch ≈ 102.24 deg
raw thrust rotor 1 ≈ 1.535 N
raw alpha rotor 1 ≈ -180 deg
```

At the preceding low-thrust stage, the refinement could place the same rotor on a very different tilt reference.

When the raw thrust crossed back above the threshold, the coarse reference could therefore contain an almost:

\[
180^\circ
\]

numerical step.

This is important:

> The singularity refinement does not guarantee a perfectly continuous scalar servo-angle reference.

It instead uses the low-authority region to improve reference conditioning and relies on NMPC dynamics/costs to manage the physical actuator.

The offline diagnostic therefore did **not** prove that the reference implementation was wrong.

It proved that the NMPC must be robust to a deliberately imperfect coarse reference.

---

# 20. Paper-style pitch trajectory

The controlled pitch experiment uses a fixed end-effector world position:

\[
p_E^W=
[0,\ 0,\ 1.5]^T\text{ m}.
\]

The original linear trajectory is:

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

Inside each moving segment:

```text
angular velocity = constant
angular acceleration = 0
```

The velocity changes discontinuously at the boundaries.

Although that is not theoretically smooth, it ultimately produced the best overall attitude behavior among the tested profiles.

---

# 21. First successful paper-style linear result

Using the original end-effector weights and direct physical servo-angle cost, the linear trajectory completed successfully.

Results:

```text
duration:
31.010 s

control updates:
3101

solver failures:
0

deadline misses:
0

mean solve time:
1.1228 ms

p95:
1.4750 ms

max:
3.8200 ms
```

Tracking:

```text
tool position RMS:
0.051709 m

tool position max:
0.090050 m

attitude RMS:
3.661387 deg

attitude max:
9.829425 deg
```

Actuator usage:

```text
max |actual alpha|:
180.481705 deg

max |alpha_cmd|:
180.481705 deg

max thrust:
16.094798 N
```

This became the original fixed-pivot baseline.

It showed that a complete 360° fixed-pivot pitch maneuver was feasible in the nominal simulation.

The main remaining weakness was the approximately:

\[
5.17\text{ cm}
\]

RMS tool-position error.

---

# 22. Trajectory-shaping experiment 1: full quintic angle profile

The first attempt to improve the linear trajectory replaced each moving segment with a quintic smoothstep in angle:

\[
s(r)
=
10r^3-15r^4+6r^5.
\]

This gives zero angular velocity and acceleration at the endpoints.

However, using the same total segment durations changes the peak angular speed.

For a quintic position smoothstep, the peak normalized slope is:

\[
1.875.
\]

Therefore the 180° / 5 s and 360° / 10 s segments reached approximately:

\[
67.5^\circ/\text{s}
\]

at their midpoint.

The closed-loop result was worse:

```text
tool RMS:
0.062144 m

tool max:
0.143996 m

attitude RMS:
3.939644 deg

attitude max:
14.213664 deg

mean solve:
1.1012 ms

solver failures:
0
```

The trajectory was smoother at the boundaries but dynamically more aggressive in the middle.

The higher peak rate created more tool drift.

Conclusion:

\[
\boxed{
\text{smooth endpoint behavior alone is not sufficient;}
\text{ peak rate matters strongly.}
}
\]

---

# 23. Trajectory-shaping experiment 2: bounded-rate smooth profile

A second profile was designed to keep the peak rate closer to the linear baseline.

It used:

- quintic smooth velocity ramp-up,
- constant-rate cruise,
- quintic smooth velocity ramp-down.

For the 180° / 5 s segment:

```text
0.5 s acceleration ramp
4.0 s cruise
0.5 s deceleration ramp
```

For the 360° / 10 s segment:

```text
1.0 s acceleration ramp
8.0 s cruise
1.0 s deceleration ramp
```

Because the smooth velocity ramp has half the plateau velocity on average, the plateau rate became:

\[
40^\circ/\text{s}.
\]

This improved position tracking relative to the full quintic case but created a severe attitude problem near the inverted transition.

Results:

```text
tool RMS:
0.052297 m

tool max:
0.087932 m

attitude RMS:
6.051545 deg

attitude max:
20.839224 deg

mean solve:
1.1576 ms
```

The position error was comparable to the original linear run.

The attitude behavior was much worse.

---

# 24. Detailed bounded-rate diagnosis

The largest attitude error occurred near the end of the first entry maneuver:

```text
t ≈ 6.99 s

reference pitch:
≈ 180 deg

reference pitch rate:
≈ 0 deg/s

attitude error:
≈ 20.84 deg
```

The vehicle had slowed too early and had not reached the inverted attitude when the reference settled.

Representative behavior:

```text
t = 5.5 s
ref pitch ≈ 130°
actual pitch ≈ 130.9°

t = 6.3 s
ref pitch ≈ 162°
actual pitch ≈ 155.7°

t = 6.5 s
ref pitch ≈ 170°
actual pitch ≈ 159.0°

t = 6.9 s
ref pitch ≈ 179.9°
actual pitch ≈ 160.0°

t = 6.99 s
ref pitch ≈ 180°
actual pitch ≈ 159.2°
```

The controller then recovered during the +180° hold.

This showed that the near-inversion problem was not simply "too much speed."

The future actuator reference behavior inside the 2-second horizon also mattered.

---

# 25. Large-angle servo-reference branch behavior

Near the inverted region, the coarse refined actuator references can represent physically related rotor directions with very different scalar servo angles.

One example from the bounded-rate experiment showed reference branches for rotors 2 and 4 moving through the \(\pm180^\circ\) boundary.

The physical actuator remained on one branch while the numerical reference changed representation.

This motivated a question:

> Should the NMPC servo-reference cost itself be made periodic?

That led to the next controlled experiment.

---

# 26. Periodic servo-direction cost experiment

The original actuator residual was:

\[
e_\alpha
=
\alpha-\alpha_r.
\]

This treats:

```text
+179°
and
-179°
```

as approximately:

\[
358^\circ
\]

apart numerically.

A trial OCP replaced direct angle tracking with a periodic direction representation:

\[
e_{\sin}
=
\sin\alpha-\sin\alpha_r
\]

and:

\[
e_{\cos}
=
\cos\alpha-\cos\alpha_r.
\]

The separate thrust-magnitude term:

\[
f-f_r
\]

was retained.

The physical servo dynamics and constraints were **not** made periodic.

The motivation was reasonable:

- rotor direction is periodic,
- the servo state itself is not.

For small angular errors:

\[
(\Delta\sin\alpha)^2
+
(\Delta\cos\alpha)^2
\approx
(\Delta\alpha)^2,
\]

so the local cost scaling remained similar to the original \(Q_\alpha\).

---

# 27. Result of the periodic-cost experiment

The periodic-cost experiment used the linear 36°/s trajectory.

Results:

```text
tool position RMS:
0.053731 m

tool position max:
0.090195 m

attitude RMS:
4.012280 deg

attitude max:
9.857165 deg

mean solve:
1.0335 ms

solver failures:
0

deadline misses:
0
```

Compared with the original direct-angle linear baseline:

| Metric | Direct physical angle | Periodic direction |
|---|---:|---:|
| Tool RMS | `5.171 cm` | `5.373 cm` |
| Tool max | `9.005 cm` | `9.020 cm` |
| Attitude RMS | `3.661°` | `4.012°` |
| Attitude max | `9.829°` | `9.857°` |

There was no improvement.

The direct-angle cost remained better.

---

# 28. Why the periodic servo cost was rejected

The periodic cost removed information that turned out to matter.

Two tilt configurations can create nearly equivalent instantaneous rotor directions while representing very different **mechanical actuator configurations**.

At approximately:

```text
t = 12.15 s
pitch reference ≈ 102.6°
```

the original direct-angle controller had approximately:

```text
actual alpha rotor 1:
-57.7°

actual alpha rotor 3:
-56.0°

attitude error:
1.36°
```

The periodic-cost controller at the same time had approximately:

```text
actual alpha rotor 1:
+94.9°

actual alpha rotor 3:
-96.2°

attitude error:
6.83°
```

The two controllers were not merely choosing different numerical names for the same state.

They were occupying different physical servo configurations.

The conclusion is:

\[
\boxed{
\text{The servo state must remain a non-periodic mechanical state in the tracking cost.}
}
\]

The project therefore reverted to:

\[
\alpha-\alpha_r.
\]

A future reference generator can still become more branch-aware if a specific maneuver proves that necessary.

But the cost itself should preserve the physical actuator configuration.

---

# 29. Key lesson from the servo-cost experiment

The periodic experiment clarified an important modeling distinction.

The mapping:

\[
\alpha
\rightarrow
[\sin\alpha,\cos\alpha]
\]

is periodic in **instantaneous force direction**.

But the servo system has memory and finite motion:

\[
\dot{\alpha}
=
\frac{\alpha_c-\alpha}{\tau}.
\]

Therefore:

```text
rotor force direction
```

and:

```text
mechanical servo configuration
```

are not interchangeable control states.

A cost that collapses different mechanical branches into the same direction can make the future actuator dynamics less well structured.

This is especially important for large-angle tiltable-rotor systems.

---

# 30. Reframing the remaining problem

After rejecting the periodic cost, the original linear direct-angle result remained the strongest balanced baseline:

```text
tool RMS:
5.17 cm

attitude RMS:
3.66 deg
```

The remaining major visible issue was not a catastrophic branch failure.

It was the approximately 5 cm dynamic tool-position drift.

The controller was numerically healthy.

The attitude remained within roughly 10° maximum.

The actuator limits were not being violated beyond the existing modeled angle boundary.

Therefore the next question became:

> Is the controller simply willing to trade away too much end-effector translation because the tool-space position/velocity weights are too weak?

This led to a deliberately small OCP tuning change.

---

# 31. Stronger end-effector tracking weights

The generic body-centered weights were left untouched.

Only the end-effector OCP received stronger translation weights.

Previous end-effector position weights:

\[
Q_{p,E}
=
[300,\ 300,\ 400].
\]

New end-effector position weights:

\[
\boxed{
Q_{p,E}
=
[600,\ 600,\ 800]
}
\]

which is:

\[
2\times
\]

the previous value.

Previous end-effector velocity weights:

\[
Q_{v,E}
=
[10,\ 10,\ 10].
\]

New end-effector velocity weights:

\[
\boxed{
Q_{v,E}
=
[30,\ 30,\ 30]
}
\]

which is:

\[
3\times
\]

the previous value.

Everything else was kept unchanged:

```text
Q_attitude:
[300, 300, 600]

Q_omega:
[5, 5, 5]

Q_alpha:
2

R_thrust:
2

R_delta_alpha:
250

N:
20

horizon:
2.0 s

control rate:
100 Hz

servo model:
unchanged

allocator/reference refinement:
unchanged

physical state/input bounds:
unchanged
```

This was intentionally a controlled tuning experiment rather than a new architecture.

---

# 32. Stronger weights on the bounded-rate trajectory

The stronger end-effector tracking weights were first tested while the bounded-rate 40°/s trajectory was still active.

Results:

```text
tool RMS:
0.038565 m

tool max:
0.061799 m

attitude RMS:
6.107675 deg

attitude max:
21.180598 deg

mean solve:
1.0982 ms
```

Compared with the earlier bounded-rate run:

```text
tool RMS:
5.230 cm → 3.857 cm

tool max:
8.793 cm → 6.180 cm
```

This was a major improvement in tool position.

The attitude problem remained almost unchanged.

That was a useful separation of effects:

- stronger tool weights improve translation,
- the ~20° inverted attitude spike belongs mainly to the bounded-rate trajectory/reference behavior.

---

# 33. Final A/B experiment: stronger weights + original linear trajectory

The next run combined:

```text
original linear 36°/s paper-style trajectory
+
direct physical alpha-alpha_ref cost
+
stronger end-effector position/velocity weights
```

This is the strongest current fixed-pivot configuration.

Results:

```text
Run completed normally:
True

Final time:
31.010 s

Control updates:
3101

Solver failures:
0

10 ms deadline misses:
0
```

Solver timing:

```text
mean:
1.0720 ms

p95:
1.4300 ms

max:
2.4890 ms
```

Tracking:

```text
tool position RMS:
0.037851 m

tool position max:
0.064564 m

attitude RMS:
3.617178 deg

attitude max:
9.912260 deg
```

Actuator usage:

```text
max |actual alpha|:
180.481705 deg

max |alpha_cmd|:
180.481705 deg

max thrust:
16.133112 N
```

This is currently the preferred fixed-pivot pitch baseline.

---

# 34. Improvement relative to the original linear baseline

Original direct-angle / original weights:

```text
tool RMS:
5.171 cm

tool max:
9.005 cm

attitude RMS:
3.661 deg

attitude max:
9.829 deg
```

Current stronger-tool-tracking result:

```text
tool RMS:
3.785 cm

tool max:
6.456 cm

attitude RMS:
3.617 deg

attitude max:
9.912 deg
```

Therefore the translation improvement is approximately:

```text
tool RMS reduction:
26.8%

tool maximum-error reduction:
28.3%
```

while attitude RMS is essentially unchanged and slightly better numerically.

This is the most important tuning result in the current document.

It demonstrates that much of the earlier 5 cm tool drift was a cost tradeoff rather than a hard control-authority limitation.

---

# 35. Axis-by-axis analysis of the final run

The final end-effector error was reconstructed directly from:

\[
p_E^W
=
p_B^W
+
R_{WB}
\begin{bmatrix}
0\\
0\\
0.15
\end{bmatrix}.
\]

Against the fixed target:

\[
[0,\ 0,\ 1.5]^T,
\]

the RMS error components were approximately:

\[
\boxed{
e_{x,\mathrm{RMS}}
=
3.761\text{ cm}
}
\]

\[
\boxed{
e_{y,\mathrm{RMS}}
=
0.297\text{ cm}
}
\]

\[
\boxed{
e_{z,\mathrm{RMS}}
=
0.310\text{ cm}
}
\]

The corresponding maximum absolute axis errors were approximately:

```text
X:
6.452 cm

Y:
1.268 cm

Z:
0.809 cm
```

This is a very important observation:

\[
\boxed{
\text{Almost all remaining tool-position error is in the pitch-plane X direction.}
}
\]

The controller is not generally inaccurate in 3-D.

The dominant issue is the translational motion required for the CoG to orbit around the fixed tool point during pitch.

---

# 36. Error during the full 360° segment

During the central full rotation:

```text
10–20 s
+180° → -180°
-36°/s
```

the axis RMS errors were approximately:

```text
X:
4.585 cm

Y:
0.369 cm

Z:
0.384 cm
```

and the total tool-position RMS during that segment was approximately:

```text
4.616 cm
```

Again, the remaining error is overwhelmingly in X.

This suggests that the next useful analysis should investigate:

- phase lag in the ideal circular CoG motion,
- whether body X motion is delayed relative to the rotational reference,
- whether end-effector velocity weighting should change further,
- whether feed-forward body acceleration can be improved,
- rather than simply multiplying all position weights again.

---

# 37. Worst points in the final run

The largest tool-position error occurred at approximately:

```text
t:
19.95 s

pitch reference:
-178.2°

pitch rate:
-36°/s

tool error:
6.456 cm

attitude error:
3.956°
```

The error vector was approximately:

```text
X:
+6.452 cm

Y:
+0.079 cm

Z:
+0.232 cm
```

Therefore the worst tool error was almost purely X-direction translation.

The largest attitude error occurred at approximately:

```text
t:
14.73 s

pitch reference:
+9.72°

pitch rate:
-36°/s

attitude error:
9.912°

tool error:
4.997 cm
```

The largest attitude error was therefore **not** located at the ±180° inversion boundary in the final linear run.

This is another reason not to over-attribute the current residual error to a simple \(\pm\pi\) branch-cut problem.

---

# 38. Segment-level final-run behavior

Approximate tool RMS / attitude RMS by segment:

| Segment | Tool RMS | Attitude RMS |
|---|---:|---:|
| Initial 0° hold, `0–2 s` | `0.487 cm` | `4.25°` |
| Entry `0→+180°`, `2–7 s` | `4.276 cm` | `3.32°` |
| +180° hold, `7–10 s` | `2.365 cm` | `3.72°` |
| Full `+180→-180°`, `10–20 s` | `4.616 cm` | `4.22°` |
| -180° hold, `20–23 s` | `2.363 cm` | `3.75°` |
| Exit `-180→0°`, `23–28 s` | `4.416 cm` | `3.25°` |
| Final 0° hold, `28–31 s` | `1.625 cm` | `0.26°` |

The controller clearly recovers during holds.

The larger errors occur mainly while the CoG is actively moving around the fixed pivot.

---

# 39. Why the linear trajectory currently remains preferred

Three trajectory families were tested.

## Original linear

```text
peak moving rate:
36°/s
```

Results:

```text
tool RMS:
5.17 cm original weights
3.79 cm stronger weights

attitude RMS:
3.66° original weights
3.62° stronger weights
```

## Full quintic position profile

```text
peak:
67.5°/s
```

Results:

```text
tool RMS:
6.21 cm

attitude max:
14.21°
```

## Bounded-rate smooth velocity profile

```text
plateau:
40°/s
```

Results:

```text
tool RMS:
5.23 cm original weights
3.86 cm stronger weights

attitude max:
~21°
```

Therefore:

\[
\boxed{
\text{The original linear 36°/s profile currently gives the best overall balance.}
}
\]

The fact that it has discontinuous reference angular velocity at segment boundaries is less damaging in the current controller than the alternative profiles tested.

This should not be generalized into a rule that discontinuous references are always better.

It is simply the empirical result for the current 2-second-horizon NMPC, cost weights, allocator reference, and maneuver timing.

---

# 40. Consolidated experiment comparison

| Experiment | Tool RMS | Tool Max | Attitude RMS | Attitude Max | Mean Solve |
|---|---:|---:|---:|---:|---:|
| Free-space effector trajectory | `0.960 cm` | `2.149 cm` | `1.434°` | `3.161°` | `0.798 ms` |
| First generic 360 showcase | `4.736 cm` | `13.118 cm` | `141.606°` | `179.961°` | `1.144 ms` |
| Linear 36°/s, original weights | `5.171 cm` | `9.005 cm` | `3.661°` | `9.829°` | `1.123 ms` |
| Full quintic | `6.214 cm` | `14.400 cm` | `3.940°` | `14.214°` | `1.101 ms` |
| Bounded-rate 40°/s | `5.230 cm` | `8.793 cm` | `6.052°` | `20.839°` | `1.158 ms` |
| Linear + periodic alpha cost | `5.373 cm` | `9.020 cm` | `4.012°` | `9.857°` | `1.034 ms` |
| Bounded-rate + stronger tool weights | `3.857 cm` | `6.180 cm` | `6.108°` | `21.181°` | `1.098 ms` |
| **Linear + stronger tool weights** | **`3.785 cm`** | **`6.456 cm`** | **`3.617°`** | **`9.912°`** | **`1.072 ms`** |

Every listed paper-style 31-second run completed with:

```text
solver failures:
0

10 ms deadline misses:
0
```

The main failures observed during this phase were therefore **tracking-quality failures**, not solver-status failures.

---

# 41. Actuator authority in the large-angle experiments

The large-angle pitch experiments use far more actuator authority than the earlier body-centered multi-axis test.

Typical maximum thrust is approximately:

```text
16 N
```

versus the model limit:

```text
23 N
```

so there is still significant thrust headroom.

The tilt states reach approximately:

```text
180.48°
```

which is the current NMPC command/state bound:

\[
3.15\text{ rad}.
\]

The physical model parameter also retains a broader mechanical concept of:

```text
±225°
```

but the NMPC currently constrains the state and command to:

\[
\pm 3.15\text{ rad}
\approx
\pm180.48^\circ.
\]

The experiments therefore genuinely exercise the current optimization bound.

They are no longer mild ±3° tilt tests.

---

# 42. Solver timing under the harder task

Despite the much harder large-angle maneuver, the solver remains comfortably inside the 10 ms deadline.

For the final preferred run:

```text
mean:
1.072 ms

p95:
1.430 ms

max:
2.489 ms
```

This is a strong result.

Compared with the earlier mild body-centered run, solve time increased, but the controller still has substantial real-time margin.

At present there is no evidence that the end-effector formulation requires reducing the control rate or prediction horizon for computational reasons.

---

# 43. Servo-angle visualization

The rigid MuJoCo plant intentionally contains no mechanical tilt joints.

That means the actual rotor tilt states are not automatically visible as articulated geometry.

A lightweight visualization approach was therefore prepared.

The preferred approach is **viewer-only**:

```text
do not change XML dynamics
do not add fake joints
do not change the plant
```

Instead, at each viewer update:

1. read the actual physical \(\alpha_i\) states,
2. compute each rotor direction,
3. draw a thin blue disc at each rotor,
4. rotate the disc according to the actual servo angle,
5. optionally draw a direction marker/arrow.

This preserves the validated rigid-body dynamics while making actuator motion visible.

A helper was created:

```text
src/tiltable_quadrotor_rl/visualization/
    servo_overlay.py
```

with an `__init__.py` package file.

The overlay is intended to show:

- blue rotor-plane discs,
- yellow direction markers,
- the real current `alpha` state rather than only the command.

This should remain a visualization layer only.

It must not be used to reintroduce mechanical hinge dynamics into the nominal plant.

---

# 44. Current preferred end-effector OCP

The current preferred cost structure is:

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

Current end-effector-specific translation weights:

\[
Q_{p,E}
=
[600,\ 600,\ 800].
\]

\[
Q_{v,E}
=
[30,\ 30,\ 30].
\]

Unchanged weights:

\[
Q_q
=
[300,\ 300,\ 600]
\]

\[
Q_\omega
=
[5,\ 5,\ 5]
\]

\[
Q_\alpha
=
2
\]

\[
R_f
=
2
\]

\[
R_{\Delta\alpha}
=
250.
\]

This should be treated as the new fixed-pivot baseline unless a later controlled experiment produces a clear improvement.

---

# 45. Current preferred fixed-pivot trajectory

The current preferred pitch benchmark is:

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

20–23 s:
hold -180°

23–28 s:
-180° → 0°
+36°/s

28–31 s:
hold 0°
```

The reason for keeping this benchmark is experimental reproducibility.

Future changes to:

- weights,
- reference generation,
- allocator behavior,
- servo model,
- feed-forward terms,

should first be compared against this exact trajectory before moving to more visually elaborate maneuvers.

---

# 46. What is now considered validated

## Fixed end-effector geometry

Validated at the software/model level:

```text
body state
→ rigid tool pose/velocity
```

and:

```text
desired tool motion
→ required body motion
```

## Tool-space NMPC cost

Validated in closed loop.

The controller can directly regulate an offset end-effector instead of only the CoG.

## Free-space end-effector tracking

Validated with:

```text
< 1 cm RMS position
~1.4° attitude RMS
0 solver failures
```

for the initial trajectory.

## Fixed-pivot large-angle pitch

Validated as feasible.

A complete 360° rotation about a nominally fixed tool point can be executed without solver failure.

## Large tilt usage

The controller operates through approximately:

```text
±180.48°
```

tilt states within the current optimization bounds.

## Real-time computational feasibility

Still validated.

Even the harder fixed-pivot maneuver remains comfortably below the 10 ms control period.

## Stronger tool-space translation weighting

Validated as beneficial.

It reduced tool RMS error by approximately 27% on the linear benchmark without materially degrading attitude tracking.

---

# 47. What has been rejected or deprioritized

## Generic large-angle showcase trajectory

Rejected as the benchmark.

It mixed too many difficult rotations before the single-axis behavior was understood.

## Full quintic angle trajectory with unchanged segment durations

Rejected for now.

Its peak angular velocity increased to approximately 67.5°/s and worsened tracking.

## Bounded-rate 40°/s smooth profile

Rejected as the preferred benchmark.

It produced approximately 20–21° attitude spikes near the inverted transition.

## Periodic servo-direction tracking cost

Rejected.

It removed useful mechanical branch information and produced worse attitude/tracking behavior.

## Immediate external-wrench observer

Deferred.

## Immediate offset-free/integral augmentation

Deferred.

## Large robustness campaign

Deferred until the end-effector work reaches a more complete manipulation milestone.

---

# 48. Current software/file map since the previous document

## End-effector core

```text
src/tiltable_quadrotor_rl/control/
    end_effector.py
    nmpc_effector_reference.py
    nmpc_effector_ocp.py
    nmpc_effector_controller.py
    nmpc_effector_trajectory.py
```

## Paper-style pitch work

```text
src/tiltable_quadrotor_rl/control/
    nmpc_effector_paper_pitch_trajectory.py
```

## Experiments

```text
experiments/nmpc/
    build_effector_nmpc.py
    run_closed_loop_effector.py
    run_paper_pitch_experiment.py
    analyze_paper_pitch_reference.py
```

## Tests

```text
tests/
    test_end_effector.py
    test_paper_pitch_trajectory.py
```

## Visualization

```text
src/tiltable_quadrotor_rl/visualization/
    __init__.py
    servo_overlay.py
```

## Rigid plant / parameters updated

```text
src/tiltable_quadrotor_rl/robot/
    parameters.py

src/tiltable_quadrotor_rl/robot/mjcf/
    tiltable_quadrotor_rigid.xml
```

---

# 49. Important telemetry/logging improvements

The paper-style runtime logs:

- time,
- pitch reference,
- pitch-rate reference,
- solver time/status,
- deadline-miss status,
- tool error,
- attitude error,
- body position,
- quaternion,
- angular velocity,
- raw thrust references,
- raw tilt references,
- refined thrust references,
- refined tilt references,
- actual tilt angles,
- tilt commands,
- thrust commands.

This has been extremely valuable.

It allowed the project to distinguish:

```text
trajectory problem
vs
reference allocation problem
vs
servo-state behavior
vs
cost-function behavior
```

without guessing from the viewer.

The large-angle work would have been significantly harder to diagnose from visual observation alone.

---

# 50. Main engineering lessons from this phase

## 50.1 End-effector control does not require adding end-effector dynamic states

A rigid tool transform plus output-space cost is sufficient.

The 17-state vehicle model remains valid.

## 50.2 A fixed tool point creates real CoG translation requirements

Holding the end-effector still while rotating the body means the CoG must orbit around it.

The translation and rotation problems are therefore strongly coupled.

## 50.3 Smooth trajectory endpoints are not automatically better

The full quintic profile increased peak angular velocity and made the result worse.

Trajectory design must consider:

- endpoint smoothness,
- peak velocity,
- peak acceleration,
- future horizon behavior,
- actuator-reference behavior.

## 50.4 A periodic force direction is not the same as a periodic servo state

This was one of the most important findings.

The rotor direction may repeat every \(2\pi\), but the physical servo state has:

- bounded travel,
- finite dynamics,
- branch history.

The NMPC cost should retain that information.

## 50.5 The singularity-refined actuator reference is allowed to be imperfect

It is a coarse reference.

The NMPC is the actual dynamic optimizer.

Trying to force the reference layer itself to behave like the full controller can lead to unnecessary complexity.

## 50.6 Large-angle closed-loop quality must be evaluated from telemetry

Solver status alone is insufficient.

The first generic showcase had zero solver failures while being physically unacceptable.

## 50.7 End-effector weight tuning matters substantially

The original 5 cm RMS drift was not purely a dynamics/authority limitation.

A controlled increase in tool translation weights reduced it to 3.8 cm RMS.

## 50.8 The remaining fixed-pivot error is strongly directional

The dominant residual is body/world X motion in the pitch plane.

Y and Z tool tracking are already millimeter-scale RMS.

This provides a much narrower next debugging target.

---

# 51. Current unresolved technical questions

The project is now in a much better-defined state.

The remaining questions are no longer broad architecture questions.

They are specific.

## 51.1 Why does the CoG lag in X during fixed-pivot pitch?

Possible contributors to investigate:

- phase lag between desired circular body path and achieved path,
- end-effector velocity weighting,
- body acceleration feed-forward,
- actuator-reference regularization,
- finite servo lag,
- horizon tradeoffs.

This should be diagnosed before simply increasing \(Q_{p,E}\) again.

## 51.2 How does the same architecture behave for roll?

The reference paper demonstrates both pitch and roll fixed-pivot rotations.

A roll benchmark is the natural next symmetry test.

## 51.3 How does it behave for mixed-axis tool trajectories?

Once pitch and roll are individually understood, useful next demonstrations include:

- combined roll/pitch,
- diagonal-axis rotation,
- translation while rotating,
- tool-point path following.

## 51.4 What should "end-effector manipulation complete" mean before RL?

A useful practical milestone could be:

```text
free-space tool trajectory tracking
+
fixed-pivot pitch
+
fixed-pivot roll
+
one combined tool-space trajectory
+
basic contact-ready model
```

without yet adding a wrist or disturbance observer.

---

# 52. Recommended next development sequence

The recommended order from the current point is:

```text
1. preserve the current linear + stronger-weight configuration as baseline

2. integrate/verify the servo-angle viewer overlay

3. analyze X-direction phase lag in the fixed-pivot pitch run
   without immediately changing multiple weights

4. run the corresponding fixed-pivot roll experiment

5. compare pitch vs roll symmetry and actuator usage

6. add one mixed-axis / manipulation-style end-effector trajectory

7. verify the collidable tool-tip/contact pipeline at a basic level

8. document the end-effector milestone

9. move toward the planned RL phase

10. return later to disturbance observers, model mismatch, integral
    compensation, hardware identification, and broader robustness
```

This sequence follows the current project priority and avoids overengineering the nominal NMPC before the manipulation pipeline is complete.

---

# 53. Current quantitative milestone snapshot

## Previous body-centered nominal NMPC

```text
position RMS:
0.260 cm

attitude RMS:
0.362°

mean solve:
0.830 ms
```

## Initial free-space end-effector NMPC

```text
tool RMS:
0.960 cm

attitude RMS:
1.434°

mean solve:
0.798 ms
```

## Original fixed-pivot linear baseline

```text
tool RMS:
5.171 cm

tool max:
9.005 cm

attitude RMS:
3.661°

attitude max:
9.829°
```

## Current preferred fixed-pivot result

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

p95:
1.430 ms

max:
2.489 ms

solver failures:
0

deadline misses:
0
```

## Current residual-error direction

```text
tool X RMS:
3.761 cm

tool Y RMS:
0.297 cm

tool Z RMS:
0.310 cm
```

The remaining fixed-pivot translation problem is therefore primarily one-dimensional.

---

# 54. Current architecture after this phase

The complete nominal end-effector control path is now:

```text
desired tool trajectory
    ↓
EndEffectorReferencePoint
    ↓
rigid tool-to-body inverse kinematics
    ↓
desired body p, v, a, q, omega, omega_dot
    ↓
feed-forward body wrench
    ↓
pseudoinverse + singularity-aware actuator reference
    ↓
stagewise end-effector NMPC horizon
    ↓
17-state / 8-input acados SQP-RTI
    ↓
thrust + alpha_cmd
    ↓
first-order servo dynamics
    ↓
actual alpha
    ↓
analytical rotor wrench
    ↓
rigid six-DoF MuJoCo plant
    ↓
body state
    ↓
rigid forward kinematics
    ↓
actual end-effector state
    ↓
tool-space feedback cost
```

This is now a genuine effector-centric nonlinear control loop.

The end-effector does not merely appear in post-processing.

It is part of the NMPC optimization objective.

---

# 55. Current architecture decision

The preferred nominal manipulation architecture is now:

\[
\boxed{
\begin{aligned}
&\text{Rigid 6-DoF vehicle}\\
+&\text{explicit 4-state tilt-servo dynamics}\\
+&\text{fixed rigid tool transform}\\
+&\text{17-state nonlinear prediction model}\\
+&\text{8 physical control inputs}\\
+&\text{tool-space nonlinear least-squares tracking}\\
+&\text{body-space feed-forward dynamics}\\
+&\text{singularity-aware coarse actuator reference}\\
+&\text{direct physical alpha tracking}\\
+&\text{100 Hz acados SQP-RTI}
\end{aligned}
}
\]

The following alternatives are currently rejected for the nominal baseline:

```text
mechanical tilt hinge DoFs in MuJoCo
periodic alpha tracking cost
full-quintic 67.5°/s paper-style pitch
bounded-rate 40°/s pitch as the main benchmark
```

---

# 56. Current milestone table

| Area | Status |
|---|---|
| Rigid six-DoF plant | Validated |
| First-order servo model | Validated mathematically |
| Body-centered NMPC | Validated nominally |
| Fixed end-effector transform | Implemented |
| Tool forward kinematics | Implemented |
| Tool-to-body inverse motion transform | Implemented |
| Tangential/centripetal acceleration correction | Implemented |
| CasADi tool kinematics in NMPC cost | Implemented |
| End-effector-specific reference layer | Implemented |
| End-effector-specific OCP | Implemented |
| End-effector controller wrapper | Implemented |
| Fixed tool site in MJCF | Implemented |
| Collidable spherical tool tip | Implemented |
| Free-space end-effector closed loop | Passed |
| Generic multi-axis 360 showcase | Rejected as benchmark |
| Paper-style pitch reference diagnostic | Completed |
| Low-thrust singularity intervals identified | Completed |
| Paper-style linear 360 pitch | Passed |
| Full-quintic pitch profile | Tested / rejected |
| Bounded-rate smooth pitch profile | Tested / rejected as baseline |
| Periodic alpha cost | Tested / rejected |
| Stronger end-effector translation weights | Tested / beneficial |
| Current fixed-pivot pitch RMS | `3.785 cm` |
| Current fixed-pivot attitude RMS | `3.617°` |
| Current fixed-pivot solver failures | `0` |
| Current fixed-pivot deadline misses | `0` |
| Remaining dominant error | Pitch-plane X translation |
| Servo-angle viewer overlay | Prepared |
| Fixed-pivot roll benchmark | Next |
| Mixed-axis tool benchmark | Not yet done |
| Contact-manipulation test | Not yet done |
| Disturbance observer | Deferred |
| Offset-free/integral augmentation | Deferred |
| Robustness/model-mismatch campaign | Deferred |
| RL phase | Planned after end-effector milestone |

---

# 57. Final conclusion

The project has now moved from:

```text
"Can the nominal tiltable quadrotor NMPC fly?"
```

to:

```text
"Can the same controller regulate a physically meaningful tool point
through large-angle aerial-manipulation motion?"
```

The answer is now:

\[
\boxed{
\text{Yes, for the current nominal rigid end-effector model.}
}
\]

The end-effector-centric controller:

- reuses the validated 17-state vehicle dynamics,
- tracks the tool directly in the OCP,
- converts desired tool motion into physically correct CoG motion,
- includes centripetal and tangential offset acceleration,
- remains computationally fast,
- handles a complete 360° pitch maneuver,
- operates near the current ±180° tilt bound,
- and achieves substantially better fixed-pivot tracking after a controlled tool-space weight change.

The strongest current pitch benchmark achieves approximately:

\[
\boxed{
3.8\text{ cm RMS tool-position error}
}
\]

with:

\[
\boxed{
3.6^\circ\text{ attitude RMS}
}
\]

and:

```text
0 solver failures
0 deadline misses
```

The large-angle experiments also produced several important negative findings:

- smoother trajectory endpoints can worsen performance if they increase peak rate,
- the bounded-rate 40°/s profile can create large near-inversion attitude lag,
- treating servo-angle tracking as periodic loses useful mechanical branch information,
- and solver success alone does not imply acceptable physical tracking.

The remaining fixed-pivot position error is now strongly localized to the pitch-plane X direction.

That gives the project a clear next technical target rather than an ambiguous controller problem.

The recommended immediate path is therefore:

\[
\boxed{
\text{preserve current baseline}
\rightarrow
\text{inspect X phase lag}
\rightarrow
\text{roll benchmark}
\rightarrow
\text{mixed-axis tool motion}
\rightarrow
\text{basic contact manipulation}
\rightarrow
\text{RL}
}
\]

with disturbance observers, offset-free compensation, deliberate model mismatch, and hardware-identification refinements deferred until they solve a demonstrated need.

---

## Current preferred benchmark configuration

For clarity, the current baseline to preserve is:

```text
Tool offset:
[0, 0, 0.15] m

Tool orientation relative to body:
identity

NMPC state:
17

NMPC input:
8

Physics:
500 Hz

Control:
100 Hz

Prediction:
20 stages / 2.0 s

Trajectory:
linear paper-style 36°/s pitch sequence

End-effector position weights:
[600, 600, 800]

End-effector velocity weights:
[30, 30, 30]

Attitude weights:
[300, 300, 600]

Angular-rate weights:
[5, 5, 5]

Alpha tracking:
direct alpha - alpha_ref

Alpha weight:
2

Thrust-reference weight:
2

Alpha-command movement weight:
250

Servo tau:
0.048 s

Thrust limit:
23 N / rotor

Alpha / alpha_cmd optimization bounds:
±3.15 rad
```

This is the configuration that should be used as the reference point for the next end-effector experiments.
