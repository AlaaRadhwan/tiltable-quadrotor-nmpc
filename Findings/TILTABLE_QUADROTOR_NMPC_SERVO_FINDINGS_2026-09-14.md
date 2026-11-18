# Tiltable Quadrotor — NMPC, Plant, Servo, and Control Architecture Findings

**Project:** `tiltable_quadrotor_rl`  
**Repository:** `~/Projects/tiltable_quadrotor_rl`  
**Current branch:** `feature/nmpc-controller`  
**Date:** 2026-09-14  

## 1. Purpose

This document consolidates the findings, experiments, rejected approaches, validated models, and current architecture decisions reached while moving the tiltable-quadrotor project toward a paper-style nonlinear model predictive controller (NMPC).

The long-term project goal is broader than reproducing one controller. The target is a defensible simulation/control stack that can support:

- nonlinear model-based control,
- reinforcement learning,
- controller comparisons,
- future sim-to-real work,
- actuator identification from real hardware,
- and a clean distinction between measured physics, modeled physics, and assumptions.

The guiding principle is:

> **Use the most reliable model supported by evidence — not the most complicated model and not the simplest model by default.**

---

## 2. Current vehicle parameters

| Parameter | Value |
|---|---:|
| Total mass | `3.039 kg` |
| Gravity | `9.81 m/s²` |
| Weight | `29.81259 N` |
| Hover thrust per rotor | `7.4531475 N` |
| Arm length | `0.275 m` |
| Inertia | `diag(0.063, 0.062, 0.095) kg·m²` |
| Rotor thrust limit | `23 N` |
| Reaction-torque ratio `kq/kt` | `0.0165 m` |
| Rotor radius | `0.114 m` |
| Current nominal servo time constant | `0.048 s` |
| Commanded tilt limit | `±3.15 rad` |
| Mechanical tilt travel | approximately `±225°` |
| MuJoCo timestep | `0.002 s` (`500 Hz`) |
| Target controller period | `0.01 s` (`100 Hz`) |

Body frame:

- `+X`: forward
- `+Y`: left
- `+Z`: up

Rotor positions:

- R1: `[+l, 0, 0]`
- R2: `[0, +l, 0]`
- R3: `[-l, 0, 0]`
- R4: `[0, -l, 0]`

Reaction-torque signs alternate:

```text
R1 -
R2 +
R3 -
R4 +
```

---

## 3. Current MuJoCo plant architecture

The current MJCF contains:

- one free-flying base,
- four physical rotor bodies,
- four physical tilt hinge joints,
- four thrust actuators,
- no MuJoCo tilt-servo actuator,
- externally modeled tilt-servo dynamics in Python.

Compiled structure:

```text
nq    = 11
nv    = 10
nu    = 4
na    = 0
njnt  = 5
nbody = 6
```

The ten velocity DoFs are:

```text
6 free-base DoFs
+
4 tilt-hinge DoFs
```

The existing Python servo path is:

```text
alpha_cmd
    ↓
first-order servo update
    ↓
write alpha directly to tilt-joint qpos
    ↓
set tilt-joint qvel = 0
```

The nominal first-order model is:

\[
\dot{\alpha}
=
\frac{\alpha_c-\alpha}{\tau}
\]

with exact discrete update:

\[
\alpha_{k+1}
=
\alpha_c+
(\alpha_k-\alpha_c)e^{-\Delta t/\tau}.
\]

That exact implementation has already been validated to floating-point precision.

---

## 4. Static allocation model

For each rotor:

\[
h_i=f_i\sin\alpha_i
\]

\[
v_i=f_i\cos\alpha_i.
\]

Define:

\[
z=
[h_1,v_1,h_2,v_2,h_3,v_3,h_4,v_4]^T
\]

and body wrench:

\[
w=
[F_x,F_y,F_z,\tau_x,\tau_y,\tau_z]^T.
\]

Then:

\[
w=Az.
\]

The allocation matrix is `6 × 8`, full row rank, and has a two-dimensional null space.

Equivalent physical-actuator form:

\[
w=B(\alpha)f.
\]

### Validation status

The wrench model has been strongly validated:

- analytical `A` vs `B(alpha)`: approximately `3.6e-15`,
- model vs MuJoCo random cases: approximately `9e-9`,
- individual rotor force / moment-arm / reaction-torque tests: passed,
- hover equilibrium: passed,
- rotor tilt geometry: passed.

The actuator-wrench mapping is therefore not currently a suspected source of error.

---

## 5. Dynamic allocator work before the NMPC pivot

A one-step dynamic allocator and then a three-step receding-horizon allocator were developed to account for servo lag.

The motivating problem was:

> A static allocator computes thrust using the desired final tilt, but the real servo has not reached that angle yet. The transient wrench can therefore be badly wrong.

The three-step horizon reduced vertical-force transient error:

```text
old endpoint-preview Fz peak error:  ~7.90 N
three-step horizon Fz peak error:    ~2.97 N
improvement:                         ~62.4%
```

But moment behavior became worse:

```text
forward peak moment:
old:      ~0.363 N·m
horizon:  ~0.541 N·m
```

The return transient moment also worsened.

### Root cause

The allocator normalized wrench components using:

```text
force scale  = mg
torque scale = mgl
```

with:

\[
mgl\approx 8.20\text{ N·m}.
\]

That made substantial torque errors relatively cheap compared with force errors. The optimizer could therefore trade angular behavior for better force tracking.

This exposed a structural problem: a standalone wrench optimizer was becoming a miniature MPC without the rigid-body dynamics that naturally convert force and torque into future translational and rotational motion.

---

## 6. Reference-paper architecture

The main reference is:

**Li et al., “Effector-Centric NMPC of Tiltable-Multirotors for Offset-Free Omnidirectional Aerial Manipulation,” IEEE Transactions on Robotics, 2026.**

The key architectural realization is that the paper does **not** use a dynamic wrench allocator as the main controller.

Its architecture is approximately:

```text
desired trajectory / pose
        ↓
desired body force and torque
        ↓
pseudoinverse reference allocation
        ↓
reference thrusts and tilt angles
        ↓
full nonlinear MPC
        ↓
actual actuator commands
```

The allocation step is a **reference generator**, not the primary flight controller.

The NMPC directly reasons about:

- rigid-body dynamics,
- velocity,
- attitude,
- angular rate,
- actuator states,
- actuator limits,
- and future state evolution.

This also avoids the artificial need to choose a normalization between Newtons and Newton-meters. Force errors become translational state errors; torque errors become rotational state errors.

---

## 7. Initial NMPC model adopted for this project

The first implementation intentionally omits:

- end-effector coordinate transformation,
- external wrench estimator,
- integral/model-error compensation,
- singularity/cable heuristic.

These can be added after a stable baseline exists.

### State

\[
x=
[p^W,v^W,q^W_B,\omega^B,\alpha].
\]

Dimensions:

```text
position       3
velocity       3
quaternion     4
angular rate   3
tilt angles    4
----------------
total         17
```

Thus:

\[
n_x=17.
\]

### Input

\[
u=
[f_1,f_2,f_3,f_4,
\alpha_{c1},\alpha_{c2},\alpha_{c3},\alpha_{c4}].
\]

Thus:

\[
n_u=8.
\]

Important distinction:

```text
alpha      = actual tilt angle
alpha_cmd  = commanded tilt angle
```

The actuator wrench must use actual `alpha`, because that is the direction the rotor physically points at that instant.

---

## 8. Continuous CasADi model

The implemented continuous dynamics are:

\[
\dot p=v
\]

\[
\dot v=
\frac{1}{m}R(q)F_B+g_W
\]

\[
\dot q=
\frac12 q\otimes[0,\omega]^T
\]

\[
\dot\omega=
I^{-1}
\left(
\tau_B-\omega\times(I\omega)
\right)
\]

\[
\dot\alpha=
\frac{\alpha_c-\alpha}{\tau}.
\]

Body wrench:

\[
[F_B;\tau_B]=B(\alpha)f.
\]

### CasADi constructs used

`ca.SX.sym(...)`

- creates symbolic unknowns,
- used for state and input variables.

`ca.DM(...)`

- creates known numeric CasADi matrices,
- used for constants such as gravity and the fixed allocation matrix.

`ca.vertcat(...)`

- vertically stacks scalars/vectors,
- used to assemble state derivatives, wrenches, and symbolic vectors.

CasADi is used because acados needs symbolic dynamics and derivatives such as:

\[
\frac{\partial f}{\partial x},
\qquad
\frac{\partial f}{\partial u}.
\]

---

## 9. Continuous-model unit validation

The first NMPC-model test produced:

```text
========================================================================
NMPC CONTINUOUS MODEL VALIDATION
========================================================================
[PASS] State dimension = 17, input dimension = 8.
[PASS] Level equal-thrust hover is a continuous-time equilibrium.
[PASS] Servo derivative matches the first-order model.
[PASS] Quaternion derivative has the expected yaw-rate behavior.
[PASS] CasADi wrench model matches allocation model (max error 7.105e-15).

========================================================================
ALL NMPC CONTINUOUS MODEL TESTS PASSED
========================================================================
```

The wrench error of `7.105e-15` is effectively floating-point roundoff.

---

## 10. CasADi / acados toolchain

### CasADi

Working version:

```text
CasADi 3.7.2
```

CasADi `3.8.0` was initially installed, but the exact cloned acados Python interface declared:

```text
casadi <= 3.7.2
```

so it was deliberately pinned to `3.7.2`.

### Cython

Cython was installed successfully.

The full project `./tools/test` suite still passed afterward.

### acados

Source location:

```text
~/Projects/acados
```

Exact commit:

```text
3fec80b744be0d8e5329d23702f06ec432e105a6
```

Successfully built:

```text
libacados.so
libhpipm.so
libblasfeo.so
```

The official minimal pendulum OCP compiled and converged.

This validated:

```text
Python
→ CasADi
→ acados_template
→ Tera code generation
→ C compilation
→ acados
→ HPIPM / BLASFEO
→ successful OCP solve
```

---

## 11. CasADi-vs-MuJoCo derivative comparison

The next experiment compared instantaneous dynamics at randomized states.

Test cases included:

- arbitrary attitude,
- arbitrary position,
- arbitrary linear velocity,
- arbitrary tilt angles,
- arbitrary thrusts,
- and later arbitrary angular velocity.

The compared derivatives were:

\[
\dot p,\quad \dot v,\quad \dot q,\quad \dot\omega.
\]

### Initial result

For zero angular velocity:

```text
position_dot         RMS = 0.000e+00
velocity_dot         RMS = 7.751e-10
quaternion_dot       RMS = 0.000e+00
omega_dot            RMS = 8.393e-02
```

For nonzero angular velocity:

```text
position_dot         RMS = 0.000e+00
velocity_dot         RMS = 7.411e-10
quaternion_dot       RMS = 1.075e+00
omega_dot            RMS = 8.690e-02
```

Two separate issues were exposed:

1. quaternion comparison convention,
2. rotational acceleration mismatch.

---

## 12. Quaternion-convention diagnosis

The initial MuJoCo quaternion diagnostic used `mju_derivQuat` with the wrong angular-velocity frame for that utility.

The comparison was corrected by transforming the angular velocity consistently.

After the correction:

```text
quaternion_dot RMS = 1.646e-16
max                = 6.661e-16
```

Therefore the CasADi quaternion kinematics are correct.

Important lesson:

> Testing only at identity attitude can hide frame-convention errors. Random attitudes are necessary.

---

## 13. Tilt-hinge dynamic inconsistency

The remaining `omega_dot` mismatch was investigated using the MuJoCo generalized mass matrix:

\[
M(q)\ddot q=f.
\]

Because the plant has 10 velocity DoFs:

\[
M\in\mathbb{R}^{10\times10}.
\]

Conceptually:

\[
M=
\begin{bmatrix}
M_{bb} & M_{b\alpha}\\
M_{\alpha b} & M_{\alpha\alpha}
\end{bmatrix}.
\]

Two cases were compared.

### Normal articulated MuJoCo dynamics

Result:

```text
omega_dot_free RMS ≈ 8.4e-02 to 8.7e-02 rad/s²
```

### Hypothetical locked-tilt-acceleration diagnostic

The base acceleration was solved from:

\[
M_{bb}\ddot q_b=f_b
\]

which corresponds to imposing:

\[
\ddot\alpha=0.
\]

Result:

```text
omega_dot_locked RMS ≈ 6e-10 rad/s²
```

At the same time, MuJoCo predicted:

```text
tilt_qacc RMS ≈ 32–33 rad/s²
tilt_qacc max ≈ 77–81 rad/s²
```

### Conclusion

This result was decisive.

The CasADi rigid-body equations are correct.

The mismatch comes from the **simulation representation**:

```text
MuJoCo:
tilt angles are real passive mechanical hinge DoFs

Python:
tilt angles are also externally prescribed actuator states
```

Those two abstractions conflict.

---

## 14. Physical-servo experiment

A physical MuJoCo servo structure was tested:

```text
alpha_cmd
    ↓
first-order actuator filter
    ↓
internal reference / activation
    ↓
position servo
    ↓
joint torque
    ↓
actual alpha
```

The filter time constant was set to:

```text
0.048 s
```

and `kp` was swept.

Results:

```text
      kp     RMS err     max err    t63 [ms]    t90 [ms]    peak tau    alpha end
----------------------------------------------------------------------------------
   0.050     0.34465     0.57398      150.00      244.00     0.01178     0.96025
   0.100     0.26859     0.49464      122.00      198.00     0.01899     0.98653
   0.200     0.20587     0.41689      100.00      168.00     0.03032     0.99365
   0.500     0.14148     0.32188       80.00      144.00     0.05639     0.99621
   1.000     0.10488     0.25863       70.00      134.00     0.09044     0.99696
   2.000     0.07688     0.20422       64.00      126.00     0.14668     0.99737
   5.000     0.05033     0.14622       58.00      120.00     0.28524     0.99766
  10.000     0.03625     0.11226       56.00      118.00     0.49956     0.99779
```

For a pure first-order system with:

\[
\tau=0.048\text{ s},
\]

the nominal response is approximately:

\[
t_{63}=48\text{ ms}
\]

and:

\[
t_{90}\approx110.5\text{ ms}.
\]

Increasing `kp` makes the inner mechanical servo faster, causing the total two-stage system to approach the original first-order response.

### Interpretation

This architecture effectively becomes:

```text
48 ms outer filter
+
mechanical servo dynamics
```

If the 48 ms parameter already represents the observed overall actuator-angle response, adding another mechanical servo stage double-counts actuator lag.

Increasing `kp` simply makes the second stage negligible again.

This experiment therefore did **not** justify a torque-level servo model.

---

## 15. What the 48 ms number means

The reference value:

\[
\tau=0.048\text{ s}
\]

should **not** be treated as a universal property of tiltable rotors.

It should currently be treated as:

```text
nominal reference value
```

for the servo architecture being reproduced.

The correct long-term design is:

```text
first-order servo structure
+
configurable tau
```

not:

```text
hardcoded 48 ms forever
```

For example, future identified hardware might give:

```text
servo A → tau ≈ 25 ms
servo B → tau ≈ 48 ms
servo C → tau ≈ 80 ms
```

and the project should require only a parameter change.

For a first-order response:

\[
t_{63}\approx\tau
\]

\[
t_{90}\approx2.3\tau
\]

\[
t_{95}\approx3\tau.
\]

---

## 16. Long-term modeling principle

Two extremes should be avoided.

### Too simple

Example:

```text
alpha = alpha_cmd instantly
```

This removes an important actuator transient and is clearly unrealistic.

### Too detailed without evidence

Example:

```text
motor electrical dynamics
gear ratio
gear inertia
backlash
Coulomb friction
viscous friction
current limit
torque-speed curve
inner kp/kd
```

If these parameters are not measured or known, the model is more complicated but not necessarily more reliable.

### Preferred strategy

Use the simplest model that captures **known dominant dynamics**, while keeping the software architecture open to later refinement.

For the servo, the best-supported current model is:

\[
\boxed{
\dot\alpha=
\frac{\alpha_c-\alpha}{\tau}
}
\]

with configurable `tau`.

---

## 17. Recommended long-term plant/controller relationship

The NMPC prediction model and the simulator do not need to be mathematically identical.

A healthy architecture is:

```text
NMPC prediction model
    ↓
fast reduced-order model of dominant dynamics

Simulation plant
    ↓
may contain additional effects when they are physically justified
```

However, additional plant complexity must be evidence-based.

The current passive hinge behavior is not a deliberately modeled physical effect. It is an artifact of representing `alpha` simultaneously as:

- a MuJoCo mechanical joint,
- and an externally imposed state.

That architecture should not remain.

---

## 18. Current recommended nominal servo architecture

The present recommendation is:

```text
alpha_cmd
    ↓
configurable first-order actuator model
    ↓
alpha
    ↓
B(alpha) f
    ↓
6-DoF rigid-body dynamics
```

This preserves:

- actuator lag,
- an explicit actuator state,
- correct transient thrust direction,
- compatibility with NMPC,
- compatibility with RL,
- and a clear path to future hardware identification.

It avoids:

- instantaneous tilt,
- passive hinge artifacts,
- invented torque-level servo parameters.

The full control/simulation state remains:

\[
[p,v,q,\omega,\alpha].
\]

---

## 19. Why this is attractive for RL

The exact first-order update:

\[
\alpha_{k+1}
=
\alpha_c+
(\alpha_k-\alpha_c)e^{-\Delta t/\tau}
\]

is cheap and vectorizable.

For many parallel environments:

```text
alpha shape = [num_envs, 4]
```

can be updated efficiently.

The wrench:

\[
w=B(\alpha)f
\]

is also straightforward to vectorize.

Later domain randomization can include:

- `tau`,
- thrust scale,
- mass,
- inertia,
- reaction-torque ratio,
- sensor noise,
- latency,
- disturbances.

This can create robust policies without inventing unsupported servo internals.

---

## 20. When a richer physical servo model becomes justified

A detailed servo model should only be added after obtaining real actuator data such as:

- step response,
- no-load angular speed,
- loaded angular speed,
- stall torque,
- torque-speed curve,
- gearbox ratio,
- backlash,
- current limits,
- saturation behavior,
- response under rotor load.

At that point the architecture could become:

```text
ServoModel
├── FirstOrderServoModel
└── PhysicalServoModel
```

The first-order model would remain useful for:

- NMPC,
- fast RL,
- baseline testing.

The physical model could then be used for:

- high-fidelity sim-to-real validation,
- robustness testing,
- hardware-specific studies.

---

## 21. Current NMPC plan

The immediate target remains a basic hover NMPC.

Development sequence:

```text
continuous symbolic model
        ↓
validate model
        ↓
finalize nominal plant-side actuator representation
        ↓
short-horizon integration validation
        ↓
create AcadosModel
        ↓
define cost
        ↓
define constraints
        ↓
set horizon
        ↓
first hover OCP
        ↓
closed-loop MuJoCo test
```

The paper provides the initial reference architecture:

```text
N = 20
controller period = 0.01 s
prediction step = 0.1 s
```

and uses:

- CasADi,
- acados,
- HPIPM,
- partial condensing,
- explicit RK,
- warm start,
- SQP-RTI.

These should initially be treated as reference choices rather than permanent final tuning.

---

## 22. Why standalone dynamic allocation should remain a baseline, not the main controller

The dynamic allocator remains useful as:

- a diagnostic,
- a baseline,
- a transient actuator experiment,
- a comparison tool.

But the primary controller should be NMPC.

A wrench allocator asks:

> Which actuators best reproduce this requested force and torque?

NMPC asks:

> Which actuator commands produce the best future vehicle motion while respecting dynamics and constraints?

For a tiltable multirotor with significant servo dynamics, the second question is more appropriate.

---

## 23. Validation philosophy

Each layer should be validated independently.

### Layer 1 — actuator geometry

Already validated:

```text
f + alpha → body wrench
```

### Layer 2 — nominal servo dynamics

Already validated:

```text
alpha_cmd → alpha
```

for the first-order model.

### Layer 3 — rigid-body dynamics

CasADi vs MuJoCo with unintended tilt acceleration removed:

```text
position_dot   ~ machine precision
velocity_dot   ~ 1e-9
quaternion_dot ~ 1e-16
omega_dot      ~ 1e-9
```

### Layer 4 — finite-time integration

Still to validate.

### Layer 5 — OCP / NMPC

Still to build.

### Layer 6 — deliberate model mismatch / robustness

Later.

---

## 24. Key engineering lessons

### 24.1 More detailed is not automatically more realistic

An invented electromechanical model may be less reliable than a simpler identified model.

### 24.2 Controller and plant models do not need exact equality

The controller needs a sufficiently accurate model of dominant dynamics, not a perfect simulator clone.

### 24.3 Hidden mechanical DoFs matter

Adding hinge DoFs changes the generalized mass matrix and can feed reaction dynamics back into the base.

### 24.4 Static validation is not enough

The plant passed:

- mass/inertia checks,
- geometry checks,
- individual rotor wrench checks,
- hover checks,

yet randomized derivative comparison still found a rotational inconsistency.

### 24.5 Random-attitude tests are necessary

Quaternion/frame bugs can remain invisible at identity attitude.

### 24.6 Dominant identified parameters matter more than unsupported detail

For the servo, the most defensible parameter today is the overall response time constant `tau`.

---

## 25. Open questions

1. **Final nominal plant implementation of `alpha`**
   - The current passive-hinge-plus-qpos-overwrite architecture should be replaced.

2. **Nominal servo time constant**
   - Current baseline: `48 ms`.
   - Must remain configurable.
   - Future value should come from hardware identification.

3. **Servo rate limit**
   - Physically reasonable, but should only be added once a defensible value is known.

4. **Torque-level servo model**
   - Deferred until hardware data exists.

5. **Quaternion norm handling inside NMPC**
   - Still to finalize.

6. **Reference pseudoinverse allocation**
   - To be added after the initial hover NMPC.

7. **External wrench estimation**
   - Deferred.

8. **End-effector-centric NMPC**
   - Deferred until the center-of-gravity baseline works.

---

## 26. Current recommended next step

Before building the full acados OCP, finalize the plant-side tilt representation.

The target must satisfy:

```text
1. alpha remains a real state.
2. alpha_cmd does not instantly become alpha.
3. tau is configurable.
4. no passive hinge dynamics are created and then overwritten.
5. no unsupported torque-level parameters are required.
6. the implementation remains efficient for RL.
7. NMPC can use the same nominal first-order actuator structure.
8. richer actuator physics can be added later without redesigning everything.
```

After that:

```text
plant tests
→ derivative comparison
→ finite-time integration comparison
→ AcadosModel
→ hover NMPC
```

---

## 27. Current architecture decision

Based on all evidence so far:

\[
\boxed{
\text{Use a configurable first-order tilt actuator model as the nominal servo model.}
}
\]

Nominal current parameter:

```text
tau = 0.048 s
```

but this is a **reference value**, not a permanent truth.

Do not:

- hardcode 48 ms throughout the project,
- assume instantaneous tilt,
- retain passive hinge dynamics that are immediately overwritten,
- invent torque-level servo parameters without hardware data.

Do:

- centralize `tau` in configuration,
- keep the actuator model modular,
- preserve `alpha` as an explicit state,
- replace `tau` with measured hardware values later,
- optionally randomize it for RL robustness.

---

## 28. Summary

Strongly validated so far:

```text
vehicle mass/inertia
rotor geometry
thrust/reaction-torque model
allocation matrix
first-order servo equation
CasADi rigid-body equations
quaternion convention
CasADi vs MuJoCo translational dynamics
CasADi vs MuJoCo rotational dynamics when unintended hinge acceleration is removed
CasADi/acados software toolchain
```

The major issue discovered is not an error in the NMPC equations.

It is the current tilt mechanism representation:

```text
physical passive hinge dynamics
+
externally imposed tilt state
```

Those abstractions conflict.

The most reliable current long-term direction is therefore:

```text
configurable first-order servo state
+
rigid-body vehicle dynamics
+
analytical rotor wrench from actual alpha
+
modular path to a future hardware-identified servo model
```

The next major milestone after finalizing that representation is:

\[
\boxed{\text{first closed-loop paper-style NMPC hover}}
\]

with state:

\[
[p,v,q,\omega,\alpha]
\]

and control:

\[
[f,\alpha_c].
\]
