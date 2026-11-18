# Tiltable Quadrotor — NMPC Robustness Characterization and Disturbance Findings

**Project:** `tiltable_quadrotor_rl`  
**Repository:** `~/Projects/tiltable_quadrotor_rl`  
**Working branch:** `feature/end-effector-nmpc`  
**Date:** 2026-10-14  
**Scope:** Progress after `TILTABLE_QUADROTOR_GENERAL_6DOF_END_EFFECTOR_NMPC_FINDINGS_2026-10-13.md`, covering controlled plant/model mismatch and external-disturbance experiments using the frozen general 6-DoF end-effector NMPC benchmark.

---

# 1. Purpose of this document

The previous milestone established that the nominal end-effector NMPC could track a general simultaneous:

\[
x,\ y,\ z,\ \phi,\ \theta,\ \psi
\]

tool trajectory with the selected empirical lateral rate-bias compensation.

The selected nominal full-amplitude result was:

```text
tool position RMS:
19.811 mm

tool position max:
37.211 mm

attitude RMS:
2.408 deg

attitude max:
3.879 deg

max thrust:
11.215 N

solver failures:
0
```

The nominal controller had therefore demonstrated adequate free-space end-effector tracking in a matched simulation.

The next question was no longer:

> Can the controller track the nominal trajectory?

It became:

\[
\boxed{
\text{What happens when the physical plant does not match the NMPC model?}
}
\]

The objective of this phase was to characterize that robustness before introducing:

- integral augmentation,
- disturbance estimation,
- offset-free NMPC,
- adaptive control,
- residual learning,
- robust NMPC,
- or RL-based compensation.

No robustness mechanism was added during the experiments documented here.

---

# 2. Main conclusion

The tests reveal a clear separation between two classes of uncertainty.

## 2.1 Parameter errors that are already tolerated well

The current controller is relatively insensitive to:

```text
±10% uniform inertia mismatch
±25% servo time-constant mismatch
```

These perturbations change transient tracking moderately or negligibly, but do not create significant steady-state pose error.

## 2.2 Errors that produce an unknown persistent net wrench

The current controller is much more sensitive to:

```text
mass mismatch
uniform thrust-effectiveness mismatch
single-rotor effectiveness mismatch
constant external force
constant external torque
```

All of these remain stable, but they create persistent position and/or attitude offsets.

The common failure mechanism is:

\[
\boxed{
\text{unknown constant wrench}
\rightarrow
\text{nonzero pose error required for feedback equilibrium}
}
\]

because the current architecture has:

```text
no integral state
no disturbance observer
no augmented constant-wrench state
no online parameter adaptation
```

The most important architectural conclusion is therefore:

\[
\boxed{
\text{The next phase should be disturbance-aware / offset-free NMPC.}
}
\]

---

# 3. Frozen controller used throughout the campaign

The robustness campaign preserved the same end-effector controller used in the final nominal 6-DoF benchmark.

Nominal rigid-body model:

```text
mass:
3.039 kg

inertia:
diag([0.063, 0.062, 0.095]) kg m^2

gravity:
9.81 m/s^2
```

Tilt servo model:

```text
nominal time constant:
0.048 s
```

Propulsion:

```text
max rotor thrust:
23 N

KQ / KT:
0.0165 m
```

Control:

```text
physics:
500 Hz

NMPC:
100 Hz

shooting interval:
0.1 s

prediction stages:
20

prediction horizon:
2.0 s
```

No NMPC weights were changed between robustness cases.

---

# 4. Benchmark trajectory

Every robustness experiment used the same general 6-DoF trajectory.

Tool center:

\[
p_{E,c}^{W}
=
[0,\ 0,\ 1.5]^T\text{ m}
\]

Translation amplitudes:

```text
X:
0.35 m

Y:
0.30 m

Z:
0.20 m
```

Translation periods:

```text
[8, 10, 6] s
```

Orientation amplitudes:

```text
roll:
40 deg

pitch:
30 deg

yaw:
60 deg
```

Orientation periods:

```text
[7, 9, 11] s
```

Schedule:

```text
0 - 2 s:
stationary

2 - 5 s:
smooth ramp-in

5 - 29 s:
full-amplitude simultaneous 6-DoF motion

29 - 32 s:
smooth ramp-out

32 - 35 s:
final stationary hold
```

The same rate-bias compensation was enabled for every case:

\[
k_b
=
0.000750
\frac{\text{m}}{\deg/\text{s}}.
\]

---

# 5. Experimental philosophy

The robustness experiments intentionally followed several rules.

## 5.1 One uncertainty at a time

The first campaign perturbed one mechanism per run.

This allows a measured degradation to be attributed to a specific model error.

## 5.2 The controller model remains nominal

For plant-mismatch cases:

\[
\text{NMPC model}
\neq
\text{MuJoCo plant}.
\]

The controller is not informed about the perturbation.

## 5.3 No retuning between cases

The same:

- NMPC weights,
- prediction horizon,
- actuator limits,
- trajectory,
- lateral compensation

were retained.

## 5.4 Final stationary hold is part of the test

The:

```text
32 - 35 s
```

final hold is particularly important because it separates:

```text
dynamic tracking error
```

from:

```text
persistent steady-state offset.
```

---

# 6. Robustness runner architecture

The general 6-DoF experiment runner was extended rather than creating a separate script for every test.

Main file:

```text
experiments/nmpc/run_6dof_effector_benchmark.py
```

The runner eventually supported named robustness cases and generic perturbation parameters including:

```text
--plant-mass-scale
--plant-inertia-scale
--servo-time-scale
--thrust-effectiveness-scale
--rotor-effectiveness
--reaction-torque-scale
--disturbance-force-world
--disturbance-torque-world
--disturbance-start-s
```

Named cases were used for reproducibility.

This is the preferred project pattern going forward:

\[
\boxed{
\text{one reusable experiment harness + explicit scenario configuration}
}
\]

rather than multiple one-off scripts.

---

# 7. Batch execution and GLFW issue

The first batch attempted to run multiple experiments sequentially using the interactive MuJoCo GLFW viewer.

The first two servo cases completed and wrote valid CSV/summary files, but the second process aborted during viewer teardown with:

```text
_glfwPlatformSetTls:
Assertion `tls->posix.allocated == GLFW_TRUE' failed.

Aborted (core dumped)
```

The simulation itself had already completed normally.

The shell loop contained:

```bash
|| break
```

so the GLFW teardown abort stopped the remaining batch.

The runner was therefore extended with:

```text
--headless
```

for automated robustness runs.

Headless mode:

- does not open GLFW,
- retains the same physics rate,
- retains the same NMPC virtual-time schedule,
- retains solve-time instrumentation,
- runs faster than real time,
- avoids viewer teardown failures.

This became the preferred mode for robustness sweeps.

---

# 8. Complete robustness matrix

The campaign contained the following configurations.

| Case | Perturbation |
|---|---|
| Nominal | matched plant/controller |
| Mass −10% | actual mass = 2.7351 kg |
| Mass +10% | actual mass = 3.3429 kg |
| Inertia −10% | all principal inertias ×0.90 |
| Inertia +10% | all principal inertias ×1.10 |
| Servo faster 25% | actual \(\tau_s=0.036\) s |
| Servo slower 25% | actual \(\tau_s=0.060\) s |
| Thrust −10% | realized thrust = 0.90 × command |
| Thrust +10% | realized thrust = 1.10 × command |
| R1 −10% | rotor 1 realized thrust = 0.90 × command |
| +X constant force | \(F_x=1.4906295\) N |
| +Z constant torque | \(\tau_z=0.4099231\) N m |

The force disturbance corresponds to:

\[
0.05mg.
\]

The torque disturbance corresponds to:

\[
0.05mgL
\]

with:

\[
L=0.275\text{ m}.
\]

Both disturbances begin at:

```text
t = 5 s
```

and remain active through the final hold.

---

# 9. Aggregate full-amplitude results

The most important comparison is the:

```text
5 - 29 s
```

full-amplitude interval.

| Case | Position RMS | Position max | Attitude RMS | Attitude max | Velocity RMS |
|---|---:|---:|---:|---:|---:|
| Nominal | 19.811 mm | 37.211 mm | 2.408° | 3.879° | 0.02028 m/s |
| Mass −10% | 110.429 mm | 117.500 mm | 3.624° | 4.730° | 0.02687 m/s |
| Mass +10% | 105.001 mm | 113.105 mm | 4.707° | 6.121° | 0.02150 m/s |
| Inertia −10% | 19.893 mm | 37.072 mm | 2.434° | 3.918° | 0.02036 m/s |
| Inertia +10% | 19.745 mm | 37.364 mm | 2.386° | 3.845° | 0.02023 m/s |
| Servo faster 25% | 23.379 mm | 43.470 mm | 2.710° | 4.327° | 0.02395 m/s |
| Servo slower 25% | 17.181 mm | 32.628 mm | 2.119° | 3.453° | 0.01751 m/s |
| Thrust −10% | 116.506 mm | 124.893 mm | 4.964° | 6.342° | 0.02241 m/s |
| Thrust +10% | 100.943 mm | 107.447 mm | 3.318° | 4.425° | 0.02631 m/s |
| R1 −10% | 65.200 mm | 100.124 mm | 6.971° | 10.304° | 0.03659 m/s |
| +X force | 57.284 mm | 85.230 mm | 2.798° | 4.932° | 0.02410 m/s |
| +Z torque | 42.640 mm | 69.737 mm | 12.846° | 15.136° | 0.04064 m/s |

Every case completed without solver failure.

---

# 10. Full-amplitude position-axis RMS

| Case | X RMS | Y RMS | Z RMS |
|---|---:|---:|---:|
| Nominal | 13.800 mm | 14.141 mm | 1.442 mm |
| Mass −10% | 18.312 mm | 21.679 mm | 106.720 mm |
| Mass +10% | 15.637 mm | 12.261 mm | 103.103 mm |
| Inertia −10% | 13.780 mm | 14.272 mm | 1.468 mm |
| Inertia +10% | 13.831 mm | 14.017 mm | 1.437 mm |
| Servo faster 25% | 17.144 mm | 15.843 mm | 1.300 mm |
| Servo slower 25% | 10.807 mm | 13.222 mm | 1.893 mm |
| Thrust −10% | 16.430 mm | 12.673 mm | 114.644 mm |
| Thrust +10% | 17.842 mm | 21.086 mm | 97.091 mm |
| R1 −10% | 43.395 mm | 35.954 mm | 32.789 mm |
| +X force | 55.332 mm | 14.416 mm | 3.466 mm |
| +Z torque | 28.160 mm | 31.937 mm | 2.277 mm |

This table makes the dominant direction of each mismatch clear.

---

# 11. Nominal reference case

Selected nominal full-amplitude result:

```text
position RMS:
19.811 mm

X/Y/Z RMS:
13.800 / 14.141 / 1.442 mm

attitude RMS:
2.408 deg

angular-velocity RMS:
0.041625 rad/s

max thrust:
11.215 N
```

The nominal result remains the comparison point for all robustness measurements.

---

# 12. Mass mismatch experiment

The controller model remained:

```text
3.039 kg
```

while the MuJoCo plant mass was changed only.

The plant inertia was deliberately left unchanged.

This isolates:

```text
translational / gravity model mismatch
```

from:

```text
rotational inertia mismatch.
```

---

# 13. −10% mass result

Actual plant:

\[
m=2.7351\text{ kg}.
\]

Full-amplitude result:

```text
position RMS:
110.429 mm

X/Y/Z RMS:
18.312 / 21.679 / 106.720 mm

attitude RMS:
3.624 deg
```

Final stationary hold:

```text
mean X error:
+0.194 mm

mean Y error:
-0.612 mm

mean Z error:
+99.131 mm

attitude RMS:
0.050 deg
```

The lighter vehicle therefore settles approximately:

\[
\boxed{99.1\text{ mm above the target}}
\]

while attitude converges almost exactly.

---

# 14. +10% mass result

Actual plant:

\[
m=3.3429\text{ kg}.
\]

Full-amplitude result:

```text
position RMS:
105.001 mm

X/Y/Z RMS:
15.637 / 12.261 / 103.103 mm

attitude RMS:
4.707 deg
```

Final stationary hold:

```text
mean X error:
+0.870 mm

mean Y error:
-0.774 mm

mean Z error:
-98.833 mm

attitude RMS:
0.112 deg
```

The heavier vehicle settles approximately:

\[
\boxed{98.8\text{ mm below the target}}.
\]

---

# 15. Mass-mismatch symmetry

A ±10% mass error corresponds to a gravity-force mismatch magnitude:

\[
|\Delta F_g|
=
0.3039(9.81)
\approx
2.9813\text{ N}.
\]

Measured final offsets:

```text
-10% mass:
+99.131 mm

+10% mass:
-98.833 mm
```

The response is nearly symmetric.

Approximate static vertical stiffness:

\[
k_{z,-10\%}
=
\frac{2.9813}{0.099131}
\approx
30.07\text{ N/m}
\]

and:

\[
k_{z,+10\%}
=
\frac{2.9813}{0.098833}
\approx
30.16\text{ N/m}.
\]

Thus:

\[
\boxed{
k_{z,\text{effective}}
\approx
30.1\text{ N/m}
}
\]

around the tested operating point.

---

# 16. Physical interpretation of mass mismatch

The controller still predicts nominal hover force:

\[
F_{\text{nom}}
=
3.039(9.81)
=
29.8126\text{ N}.
\]

For a lighter plant:

```text
nominal feedforward is too large
→ vehicle rises
→ positive Z error develops
→ feedback reduces commanded thrust
→ equilibrium is reached above target
```

For a heavier plant:

```text
nominal feedforward is too small
→ vehicle falls
→ negative Z error develops
→ feedback increases commanded thrust
→ equilibrium is reached below target
```

The controller can therefore find the correct force, but only by maintaining a nonzero position error.

This is the first direct evidence that the current NMPC is:

\[
\boxed{\text{not offset-free}}
\]

under constant force/model error.

---

# 17. Dynamic error after removing the mass-induced mean offset

The raw mass-mismatch RMS values are dominated by steady Z bias.

After subtracting the full-amplitude mean position-error vector, the remaining position-error RMS is approximately:

```text
nominal:
19.70 mm

mass -10%:
28.29 mm

mass +10%:
19.56 mm
```

This reveals an important asymmetry.

The +10% mass case retains nearly nominal **dynamic** position tracking once the constant offset is removed.

The −10% case additionally degrades dynamic X/Y tracking.

Therefore robustness should be analyzed using both:

```text
steady offset
```

and:

```text
offset-debiased dynamic error.
```

---

# 18. Uniform inertia mismatch

The mass remained exactly nominal while all three principal inertias were scaled together.

Tested:

```text
0.90 × nominal inertia
1.10 × nominal inertia
```

The ratios:

\[
I_x:I_y:I_z
\]

were preserved.

---

# 19. −10% inertia result

Actual inertia:

```text
[0.0567, 0.0558, 0.0855] kg m^2
```

Full-amplitude:

```text
position RMS:
19.893 mm

attitude RMS:
2.434 deg

angular-velocity RMS:
0.042020 rad/s
```

Final stationary position/attitude remained essentially nominal.

---

# 20. +10% inertia result

Actual inertia:

```text
[0.0693, 0.0682, 0.1045] kg m^2
```

Full-amplitude:

```text
position RMS:
19.745 mm

attitude RMS:
2.386 deg

angular-velocity RMS:
0.041326 rad/s
```

Again, there was no meaningful steady offset.

---

# 21. Inertia sensitivity conclusion

Relative to nominal position RMS:

```text
-10% inertia:
+0.41%

+10% inertia:
-0.33%
```

Relative to nominal attitude RMS:

```text
-10% inertia:
+1.08%

+10% inertia:
-0.91%
```

The point-by-point actual trajectory changed by only roughly:

```text
0.53 mm RMS in position
0.10 deg RMS in orientation
```

for either sign of perturbation.

Therefore:

\[
\boxed{
\pm10\%\text{ uniform inertia mismatch is practically negligible}
}
\]

for this trajectory and controller.

This test does **not** cover:

- asymmetric inertia error,
- products of inertia,
- shifted payload CoG,
- geometry-dependent payload inertia.

---

# 22. Servo time-constant mismatch

The controller continues to predict:

\[
\tau_s=0.048\text{ s}.
\]

Two plant values were tested:

```text
faster plant servo:
0.036 s
= 0.75 × nominal

slower plant servo:
0.060 s
= 1.25 × nominal
```

---

# 23. Faster servo result

Full-amplitude:

```text
position RMS:
23.379 mm

X/Y/Z:
17.144 / 15.843 / 1.300 mm

attitude RMS:
2.710 deg

attitude max:
4.327 deg
```

This is a modest degradation relative to nominal.

Final hold remained close to zero.

---

# 24. Slower servo result

Full-amplitude:

```text
position RMS:
17.181 mm

X/Y/Z:
10.807 / 13.222 / 1.893 mm

attitude RMS:
2.119 deg

attitude max:
3.453 deg
```

This was slightly **better** than nominal for the tested trajectory.

Final hold remained close to zero.

---

# 25. Interpretation of the servo result

The slower-servo result should **not** be interpreted as:

> slower tilt servos are inherently better.

The correct conclusion is:

\[
\boxed{
\text{this controller is not strongly sensitive to a ±25% servo-time error}
}
\]

for the general benchmark.

The 0.060 s plant happens to interact favorably with the current:

- trajectory,
- weights,
- command spectrum,
- actuator regularization.

The hardware servo time constant should still be identified and modeled accurately.

No persistent offset appears because the servo mismatch is dynamic rather than a constant missing wrench at steady state.

---

# 26. Uniform thrust-effectiveness mismatch

The next experiment kept mass and inertia nominal but changed realized rotor thrust.

Controller assumption:

\[
f_{\text{actual}}
=
f_{\text{commanded}}.
\]

Tested plant behavior:

\[
f_{\text{actual}}
=
0.9f_{\text{commanded}}
\]

and:

\[
f_{\text{actual}}
=
1.1f_{\text{commanded}}.
\]

---

# 27. −10% thrust effectiveness

Full-amplitude:

```text
position RMS:
116.506 mm

X/Y/Z:
16.430 / 12.673 / 114.644 mm

attitude RMS:
4.964 deg
```

Final stationary hold:

```text
mean Z error:
-109.828 mm

attitude RMS:
0.141 deg
```

The controller commands during final hold:

```text
total commanded thrust:
33.122842 N
```

The actual realized thrust is approximately:

\[
0.9(33.122842)
=
29.8106\text{ N},
\]

which is essentially the true vehicle weight.

Thus the controller reaches the correct physical force by settling below the target.

---

# 28. +10% thrust effectiveness

Full-amplitude:

```text
position RMS:
100.943 mm

X/Y/Z:
17.842 / 21.086 / 97.091 mm

attitude RMS:
3.318 deg
```

Final stationary hold:

```text
mean Z error:
+90.130 mm

attitude RMS:
0.045 deg
```

Final commanded total thrust:

```text
27.101227 N
```

Actual realized thrust:

\[
1.1(27.101227)
=
29.8113\text{ N}.
\]

Again, the actual physical force converges almost exactly to vehicle weight.

---

# 29. Why thrust error and mass error look similar

Both errors create a persistent mismatch between:

```text
predicted vertical acceleration
```

and:

```text
actual vertical acceleration.
```

Mass mismatch changes:

\[
\frac{F}{m}
\]

by changing:

\[
m.
\]

Thrust effectiveness mismatch changes the actual:

\[
F.
\]

From the controller's perspective, both appear as an approximately constant missing/excess force near hover.

Therefore both produce:

\[
\boxed{
\text{large vertical steady-state position error}
}
\]

with almost-correct final physical thrust.

---

# 30. Single weak rotor experiment

Uniform thrust scaling is a symmetric mismatch.

A more realistic actuator fault/calibration error was tested by reducing only rotor 1:

```text
R1:
90% effectiveness

R2:
100%

R3:
100%

R4:
100%
```

This is fundamentally different because it introduces both:

```text
force error
```

and:

```text
moment error.
```

---

# 31. R1 −10% result

Full-amplitude:

```text
position RMS:
65.200 mm

X/Y/Z:
43.395 / 35.954 / 32.789 mm

attitude RMS:
6.971 deg

attitude max:
10.304 deg

angular-velocity RMS:
0.075811 rad/s
```

Final stationary hold:

```text
mean X error:
+52.831 mm

mean Y error:
-0.428 mm

mean Z error:
-29.029 mm

attitude RMS:
7.058 deg
```

The final attitude error is predominantly:

```text
pitch:
~+7.05 deg

yaw:
~+0.31 deg

roll:
approximately zero
```

Thus:

\[
\boxed{
\text{one weak rotor produces a coupled force + moment equilibrium bias}
}
\]

rather than a simple altitude offset.

---

# 32. Meaning of the weak-rotor result

The controller remains stable.

It also redistributes rotor thrust:

```text
final commanded R1:
8.077 N

R2:
7.825 N

R3:
7.267 N

R4:
7.666 N
```

but because the NMPC still believes R1 produces its commanded force, its predicted wrench remains wrong.

The physical system therefore settles at a biased pose where nominal feedback generates enough compensating force and torque.

This is strong evidence that identifying the exact fault type is not necessary for the next robustness mechanism.

A general unknown-wrench estimator can represent the resulting residual wrench directly.

---

# 33. Constant world-X force disturbance

A constant external world-frame force was applied:

\[
F_d^W
=
[1.4906295,\ 0,\ 0]^T\text{ N}.
\]

This is:

\[
5\%
\]

of vehicle weight.

The disturbance begins at:

```text
5 s
```

and remains active through the final hold.

---

# 34. +X force result

Full-amplitude:

```text
position RMS:
57.284 mm

X/Y/Z:
55.332 / 14.416 / 3.466 mm

attitude RMS:
2.798 deg
```

Final stationary hold:

```text
mean X error:
+54.575 mm

mean Y error:
-1.147 mm

mean Z error:
+0.210 mm

attitude RMS:
1.178 deg
```

The dominant result is therefore a clean persistent X displacement.

---

# 35. Effective lateral stiffness

Using:

\[
F_x=1.4906295\text{ N}
\]

and:

\[
e_x=0.054575\text{ m},
\]

the approximate steady-state closed-loop stiffness is:

\[
k_{x,\text{effective}}
=
\frac{1.4906295}{0.054575}
\approx
27.3\text{ N/m}.
\]

This is close in scale to the approximately:

```text
30 N/m
```

vertical effective stiffness inferred from the mass-mismatch experiments.

This supports a consistent feedback-equilibrium interpretation.

---

# 36. Constant world-Z torque disturbance

A constant external yaw torque was applied:

\[
\tau_d^W
=
[0,\ 0,\ 0.409923]^T\text{ N m}.
\]

This corresponds to:

\[
0.05mgL.
\]

---

# 37. Yaw-torque result

Full-amplitude:

```text
position RMS:
42.640 mm

X/Y/Z:
28.160 / 31.937 / 2.277 mm

attitude RMS:
12.846 deg

attitude max:
15.136 deg

angular-velocity RMS:
0.079668 rad/s
```

Final stationary hold:

```text
position RMS:
2.291 mm

attitude RMS:
11.160 deg
```

The final attitude error is almost purely:

```text
yaw:
~11.159 deg
```

while:

```text
roll:
approximately zero

pitch:
approximately zero
```

and angular velocity is close to zero.

This is therefore a **steady biased rotational equilibrium**, not continuous spinning.

---

# 38. Effective yaw stiffness

Using:

\[
\tau_z=0.409923\text{ N m}
\]

and:

\[
e_\psi
\approx
11.159^\circ
=
0.19476\text{ rad},
\]

the approximate static rotational stiffness is:

\[
k_{\psi,\text{effective}}
=
\frac{0.409923}{0.19476}
\approx
2.10\text{ N m/rad}.
\]

The controller can generate the required opposing yaw torque, but needs a large persistent attitude error to do so.

---

# 39. Offset-debiased dynamic tracking

For cases with large constant offsets, raw RMS can exaggerate the true transient/dynamic degradation.

The full-amplitude position-error vector was therefore also analyzed after subtracting its own mean.

Approximate offset-debiased RMS:

| Case | Offset-debiased position RMS |
|---|---:|
| Nominal | 19.70 mm |
| Mass −10% | 28.29 mm |
| Mass +10% | 19.56 mm |
| Inertia −10% | 19.78 mm |
| Inertia +10% | 19.64 mm |
| Servo fast | 23.21 mm |
| Servo slow | 17.12 mm |
| Thrust −10% | 20.39 mm |
| Thrust +10% | 27.54 mm |
| R1 −10% | 42.30 mm |
| +X force | 20.40 mm |
| +Z torque | 41.79 mm |

This table is important.

Several large raw RMS values are primarily caused by steady bias rather than severe loss of dynamic tracking.

Examples:

```text
+X force:
57.3 mm raw
20.4 mm after mean removal

thrust -10%:
116.5 mm raw
20.4 mm after mean removal

mass +10%:
105.0 mm raw
19.6 mm after mean removal
```

The weak-rotor and yaw-torque cases still show substantial dynamic degradation after de-biasing.

---

# 40. Robustness classification

The complete campaign can now be divided into three groups.

## 40.1 Low-sensitivity dynamic-model mismatch

```text
uniform inertia mismatch
moderate servo time-constant mismatch
```

These do not create large steady offsets.

## 40.2 Primarily constant-force mismatch

```text
mass mismatch
uniform thrust-effectiveness mismatch
constant external force
```

These produce persistent translational offsets.

## 40.3 Coupled force-and-moment mismatch

```text
single weak rotor
external torque
```

These can produce persistent orientation offsets and larger dynamic degradation.

---

# 41. Why the controller stays stable

None of the tested cases caused:

- solver divergence,
- actuator saturation,
- uncontrolled growth,
- loss of simulation stability.

Even severe-offset cases remained bounded.

This means:

\[
\boxed{
\text{the nominal feedback structure has useful robustness}
}
\]

but:

\[
\boxed{
\text{it lacks zero steady-state error under persistent model/wrench error}
}
\]

These are different properties.

---

# 42. Actuator authority was not the main limitation

Representative maximum commanded thrust values remained well below:

```text
23 N
```

even in the disturbed cases.

Examples:

```text
nominal:
11.215 N

mass +10%:
12.013 N

thrust -10%:
12.105 N

R1 -10%:
12.047 N

+X force:
11.391 N

yaw torque:
11.205 N
```

Servo excursions also remained far below the modeled command-angle bound.

Therefore the persistent offsets are not explained by insufficient actuator authority.

---

# 43. Solver reliability

Across all tested robustness cases:

```text
solver failures:
0
```

Acados-only 10 ms deadline misses were also zero in all reported runs.

The perturbations therefore did not create a numerical-solver reliability problem.

---

# 44. Timing interpretation

Total Python controller time remained in approximately the same regime across the campaign.

Typical mean values:

```text
~8.4 - 8.9 ms
```

The number of 10 ms total-controller misses varied between runs, but this variation did not correlate meaningfully with robustness severity.

This reinforces the previous conclusion:

```text
controller runtime limitation:
implementation / Python reference path

robustness limitation:
persistent unmodeled wrench
```

These are separate issues.

---

# 45. Important finding: not every model mismatch deserves a robustness mechanism

The inertia tests demonstrate why robustness mechanisms should not be added simply because a parameter is uncertain.

The controller already tolerates realistic ±10% uniform inertia error with negligible performance change.

Likewise, ±25% servo-time error remains stable.

Adding adaptation for those quantities first would increase architecture complexity without addressing the dominant measured weakness.

The experimental data instead points toward **unknown-wrench compensation**.

---

# 46. Why ordinary integral action is not the preferred first solution

A conventional integral term could likely reduce several steady offsets.

For translation:

\[
\int e_p\,dt
\]

could increase/decrease commanded force until the position error disappears.

However, the robustness campaign shows that the error source is naturally six-dimensional.

For example, the weak-R1 case creates simultaneous:

- translational force error,
- pitch-moment error,
- smaller yaw coupling.

A general disturbance representation is therefore more natural than independent scalar integrators.

Additional issues with direct integral augmentation include:

- attitude integration on \(SO(3)\),
- anti-windup,
- coupling through the omnidirectional allocator,
- NMPC prediction still using the wrong physical model.

---

# 47. Recommended next architecture

The preferred next step is:

\[
\boxed{
\textbf{6-D disturbance-wrench estimation + disturbance-aware NMPC}
}
\]

Introduce an estimated unknown wrench:

\[
\hat d
=
\begin{bmatrix}
\hat F_d^W\\
\hat\tau_d^B
\end{bmatrix}.
\]

Translational model:

\[
m\dot v
=
R_{WB}F_B
+
mg
+
\hat F_d^W.
\]

Rotational model:

\[
I\dot\omega
=
\tau_B
+
\hat\tau_d^B
-
\omega\times(I\omega).
\]

---

# 48. Why a disturbance wrench matches the observed failures

For mass mismatch:

\[
\hat F_d^W
\]

can represent the gravity-model error.

For thrust-scale mismatch:

\[
\hat F_d^W
\]

can represent the missing/excess resultant force.

For a constant wind-like disturbance:

\[
\hat F_d^W
\]

directly represents the external force.

For yaw disturbance:

\[
\hat\tau_d
\]

directly represents the missing moment.

For a weak rotor:

\[
\begin{bmatrix}
\hat F_d\\
\hat\tau_d
\end{bmatrix}
\]

can represent the complete resultant wrench error without first identifying exactly which rotor is weak.

This makes the disturbance wrench an appropriate abstraction for all dominant failures found in this campaign.

---

# 49. Recommended first observer implementation

The current simulation has direct, exact access to full state.

Therefore the first version should **not** immediately introduce an EKF.

A simpler research progression is:

1. compute the nominal model-predicted state derivative / state transition,
2. compare against the measured MuJoCo state transition,
3. infer the unexplained translational force and rotational torque,
4. low-pass filter the wrench estimate,
5. feed the estimate into the NMPC prediction,
6. rerun the exact same robustness matrix.

This isolates the control architecture question:

> Does estimated-wrench compensation remove the offsets?

before introducing realistic sensing and estimation noise.

---

# 50. Success criteria for the disturbance-aware phase

The existing campaign gives concrete before/after targets.

## 50.1 Mass mismatch

Current:

```text
±10% mass:
~99 mm steady Z offset
```

Desired after compensation:

```text
few-mm-level steady Z error
```

## 50.2 Thrust-effectiveness mismatch

Current:

```text
-10% thrust:
~110 mm steady Z offset

+10% thrust:
~90 mm steady Z offset
```

Desired:

```text
few-mm-level steady error
```

## 50.3 Constant X force

Current:

```text
+1.49 N:
~54.6 mm steady X offset
```

Desired:

```text
few-mm-level X offset
```

## 50.4 Constant yaw torque

Current:

```text
+0.410 N m:
~11.16 deg steady yaw error
```

Desired:

```text
well below 1 deg
```

## 50.5 Weak rotor

Current:

```text
X:
~52.8 mm

Z:
~-29.0 mm

pitch:
~+7.05 deg
```

Desired:

```text
large reduction in all three without explicit fault identification
```

---

# 51. Tests not included in this campaign

Several possible uncertainties were deliberately excluded.

## 51.1 Sensor noise

The current benchmark uses ideal full state.

Noise belongs with the later estimator design.

## 51.2 State-estimation error

No EKF or observer is yet part of the loop.

## 51.3 Communication/control delay

This belongs to deployment/runtime robustness.

## 51.4 Motor thrust dynamics

No experimentally identified motor thrust time constant is currently available.

Inventing one would make the test arbitrary.

## 51.5 Aerodynamic drag uncertainty

No validated drag model has been established.

## 51.6 Tool-offset mismatch

This is important for aerial manipulation but represents a separate end-effector calibration/payload problem.

## 51.7 CoG shift and non-diagonal inertia

The uniform-inertia test does not cover payload-induced CoG movement or products of inertia.

These can be tested later if manipulation payloads make them relevant.

---

# 52. Findings accepted from this phase

## 52.1 The nominal NMPC is not fragile

Every tested case remained bounded and solver-stable.

## 52.2 ±10% uniform inertia mismatch is negligible

No robustness modification is justified specifically for this uncertainty.

## 52.3 ±25% servo time-constant mismatch is tolerable

Moderate servo identification error is not currently the dominant limitation.

## 52.4 Mass mismatch produces symmetric vertical steady-state error

Approximately:

```text
±10% mass
→ ∓99 mm Z equilibrium shift
```

## 52.5 Uniform thrust calibration error produces the same type of failure

The controller converges to the correct actual supporting force but at the wrong altitude.

## 52.6 A single weak rotor produces a coupled pose bias

The R1 −10% case creates large:

- X offset,
- Z offset,
- pitch offset.

## 52.7 Constant external force produces a translational equilibrium offset

A 1.49 N X disturbance creates approximately 54.6 mm X bias.

## 52.8 Constant external yaw torque produces a rotational equilibrium offset

A 0.410 N m yaw torque creates approximately 11.16° yaw bias.

## 52.9 Actuator saturation is not responsible

The controller has significant remaining thrust and tilt authority.

## 52.10 Solver failure is not responsible

All runs completed with zero NMPC solver failures.

## 52.11 Persistent unknown wrench is the common failure mechanism

This is the central robustness result.

## 52.12 More random parameter sweeps are not currently useful

The next project value comes from compensating the measured weakness.

---

# 53. Current project state after robustness characterization

The validated progression is now:

```text
rigid plant validation
↓
allocation validation
↓
NMPC model validation
↓
end-effector NMPC
↓
large-angle pitch/roll
↓
cartwheel / low-thrust handover
↓
predictive rotor pre-positioning
↓
general simultaneous 6-DoF tracking
↓
generalized rate-bias compensation
↓
model-mismatch characterization
↓
external-disturbance characterization
```

The project has now reached:

\[
\boxed{
\text{nominal control validation complete}
}
\]

and:

\[
\boxed{
\text{open-loop robustness characterization complete}
}
\]

for the current simulation stage.

---

# 54. Recommended next milestone

The next major milestone should be:

\[
\boxed{
\textbf{offset-free / disturbance-aware end-effector NMPC}
}
\]

Suggested progression:

```text
deterministic 6-D wrench residual estimation
↓
filter / observer design
↓
feed estimated wrench into NMPC prediction
↓
repeat identical robustness benchmark
↓
measure offset rejection and transient recovery
↓
add realistic state-estimation noise
↓
only then consider adaptive / learned residual methods
```

No controller-weight retuning should be used to hide the robustness problem before the disturbance-aware architecture is evaluated.

---

# 55. Experiment artifact index

## 55.1 Nominal optimized 6-DoF benchmark

```text
general_6dof_effector_closed_loop_bias_comp_20261013_140130.csv
general_6dof_effector_closed_loop_bias_comp_20261013_140130_summary.txt
```

## 55.2 Mass −10%

```text
general_6dof_effector_closed_loop_bias_comp_mass_scale_0p900_20261013_144337.csv
general_6dof_effector_closed_loop_bias_comp_mass_scale_0p900_20261013_144337_summary.txt
```

## 55.3 Mass +10%

```text
general_6dof_effector_closed_loop_bias_comp_mass_scale_1p100_20261013_145138.csv
general_6dof_effector_closed_loop_bias_comp_mass_scale_1p100_20261013_145138_summary.txt
```

## 55.4 Inertia −10%

```text
general_6dof_effector_closed_loop_bias_comp_inertia_scale_0p900_20261014_105719.csv
general_6dof_effector_closed_loop_bias_comp_inertia_scale_0p900_20261014_105719_summary.txt
```

## 55.5 Inertia +10%

```text
general_6dof_effector_closed_loop_bias_comp_inertia_scale_1p100_20261014_105816.csv
general_6dof_effector_closed_loop_bias_comp_inertia_scale_1p100_20261014_105816_summary.txt
```

## 55.6 Servo faster 25%

```text
general_6dof_effector_closed_loop_bias_comp_servo_fast_25_20261014_123039.csv
general_6dof_effector_closed_loop_bias_comp_servo_fast_25_20261014_123039_summary.txt
```

## 55.7 Servo slower 25%

```text
general_6dof_effector_closed_loop_bias_comp_servo_slow_25_20261014_123117.csv
general_6dof_effector_closed_loop_bias_comp_servo_slow_25_20261014_123117_summary.txt
```

## 55.8 Uniform thrust −10%

```text
general_6dof_effector_closed_loop_bias_comp_thrust_low_10_20261014_124214.csv
general_6dof_effector_closed_loop_bias_comp_thrust_low_10_20261014_124214_summary.txt
```

## 55.9 Uniform thrust +10%

```text
general_6dof_effector_closed_loop_bias_comp_thrust_high_10_20261014_124249.csv
general_6dof_effector_closed_loop_bias_comp_thrust_high_10_20261014_124249_summary.txt
```

## 55.10 Rotor 1 −10%

```text
general_6dof_effector_closed_loop_bias_comp_rotor1_low_10_20261014_124324.csv
general_6dof_effector_closed_loop_bias_comp_rotor1_low_10_20261014_124324_summary.txt
```

## 55.11 Constant X force

```text
general_6dof_effector_closed_loop_bias_comp_force_x_5pct_weight_20261014_124359.csv
general_6dof_effector_closed_loop_bias_comp_force_x_5pct_weight_20261014_124359_summary.txt
```

## 55.12 Constant yaw torque

```text
general_6dof_effector_closed_loop_bias_comp_yaw_torque_5pct_20261014_124434.csv
general_6dof_effector_closed_loop_bias_comp_yaw_torque_5pct_20261014_124434_summary.txt
```

---

# 56. Final milestone statement

The robustness campaign shows that the end-effector NMPC is **dynamically robust but not offset-free**.

It tolerates moderate uncertainty in:

\[
I
\]

and:

\[
\tau_s
\]

without meaningful loss of stability.

However, persistent errors that appear as an unmodeled six-dimensional wrench lead to biased equilibria.

The measured examples include:

\[
\boxed{
\pm10\%\text{ mass}
\Rightarrow
\sim99\text{ mm vertical offset}
}
\]

\[
\boxed{
\pm10\%\text{ thrust gain}
\Rightarrow
\sim90\text{–}110\text{ mm vertical offset}
}
\]

\[
\boxed{
1.49\text{ N lateral force}
\Rightarrow
54.6\text{ mm lateral offset}
}
\]

\[
\boxed{
0.410\text{ N m yaw torque}
\Rightarrow
11.16^\circ\text{ yaw offset}
}
\]

and:

\[
\boxed{
10\%\text{ weak rotor}
\Rightarrow
\text{coupled translational + rotational bias}
}
\]

These failures occur despite:

- zero solver failures,
- substantial actuator margin,
- bounded vehicle motion.

The controller therefore does not need a completely different nominal control law.

It needs a mechanism that lets the prediction model account for persistent unknown force and torque.

The recommended next architecture is:

\[
\boxed{
\textbf{estimated disturbance wrench + offset-free NMPC prediction}
}
\]

and the existing robustness suite should be preserved as the regression benchmark used to evaluate that architecture.
