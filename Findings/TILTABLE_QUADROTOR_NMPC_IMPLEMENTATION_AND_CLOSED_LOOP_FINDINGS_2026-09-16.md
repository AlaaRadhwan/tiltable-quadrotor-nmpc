# Tiltable Quadrotor — Rigid Plant, NMPC Implementation, and Closed-Loop Findings

**Project:** `tiltable_quadrotor_rl`  
**Repository:** `~/Projects/tiltable_quadrotor_rl`  
**Branch:** `feature/nmpc-controller`  
**Date:** 2026-09-16  
**Scope:** Work completed after the previous dynamic-allocation and NMPC/servo findings documents

---

## 1. Purpose of this document

This document records the major engineering progress made after the two previous project documents:

- `TILTABLE_QUADROTOR_DYNAMIC_ALLOCATOR_FINDINGS_UPDATED.md`
- `TILTABLE_QUADROTOR_NMPC_SERVO_FINDINGS_2026-09-14.md`

Those documents ended with two major unresolved items:

1. the nominal MuJoCo plant still represented tilt angles as mechanical hinge DoFs while also externally imposing them as actuator states, and
2. the project had not yet built or tested the full acados NMPC closed loop.

Since then, both of those issues have been resolved at the nominal-model level.

The project now has:

- a rigid six-DoF MuJoCo plant with no hidden tilt-joint dynamics,
- a separately modeled first-order tilt actuator state,
- instantaneous and finite-time agreement between the CasADi model and MuJoCo,
- a paper-style 17-state / 8-input acados NMPC,
- general trajectory-reference generation,
- stagewise feed-forward actuator references,
- successful acados code generation and compilation,
- and a full 30-second closed-loop multi-axis simulation at 100 Hz NMPC / 500 Hz physics.

The most important milestone is:

\[
\boxed{
\text{The general nonlinear NMPC is now operational in closed loop on the validated nominal plant.}
}
\]

This is no longer a hover-only or single-axis controller experiment.

---

# 2. Executive summary of what changed

The previous NMPC/servo document concluded that the articulated MuJoCo tilt-joint representation was structurally inconsistent with the reduced-order NMPC model.

That diagnosis was correct.

The project has now replaced that plant architecture.

The current nominal control/simulation chain is:

```text
trajectory
    ↓
reference generation
    ↓
feed-forward wrench
    ↓
pseudoinverse / reference allocation
    ↓
NMPC
state: [p, v, q, omega, alpha]
input: [f1..f4, alpha_cmd1..4]
    ↓
first-order servo model
    ↓
actual alpha
    ↓
analytical rotor wrench B(alpha) f
    ↓
rigid 6-DoF MuJoCo body
```

The older three-step dynamic allocator is retained as an experimental baseline, but it is no longer the planned primary flight-control architecture.

The NMPC directly predicts:

- rigid-body motion,
- attitude,
- angular velocity,
- servo-angle states,
- actuator limits,
- and future actuator effects.

This removes the need for a standalone wrench allocator to decide force-versus-moment tradeoffs without seeing vehicle motion.

---

# 3. Current physical and control parameters

The centralized nominal parameters remain:

| Parameter | Value |
|---|---:|
| Mass | `3.039 kg` |
| Gravity | `9.81 m/s²` |
| Weight | `29.81259 N` |
| Hover thrust / rotor | `7.4531475 N` |
| Inertia | `diag(0.063, 0.062, 0.095) kg·m²` |
| Arm length | `0.275 m` |
| Maximum thrust / rotor | `23 N` |
| Reaction torque ratio `kq/kt` | `0.0165 m` |
| Nominal servo time constant | `0.048 s` |
| NMPC tilt / tilt-command bound | `±3.15 rad` |
| MuJoCo timestep | `0.002 s` |
| MuJoCo rate | `500 Hz` |
| NMPC control period | `0.010 s` |
| NMPC update rate | `100 Hz` |
| NMPC shooting stages | `20` |
| NMPC shooting interval | `0.100 s` |
| Prediction horizon | `2.0 s` |

The 48 ms servo time constant is still treated as a nominal configurable reference value rather than a measured hardware truth.

---

# 4. Finalized nominal plant architecture

## 4.1 Why the previous articulated plant was rejected

The previous MuJoCo vehicle contained:

```text
free base
+
4 physical tilt hinge joints
```

while Python simultaneously imposed the tilt state using the external first-order actuator model.

Even with tilt-joint `qpos` overwritten and tilt-joint `qvel` set to zero, MuJoCo still treated those hinges as genuine mechanical generalized coordinates.

The generalized mass matrix therefore contained base/tilt coupling.

A diagnostic showed:

```text
CASE A: omega = 0

velocity_dot RMS       ≈ 7.75e-10
omega_dot_free RMS     ≈ 8.39e-02
omega_dot_locked RMS   ≈ 6.20e-10
tilt_qacc RMS          ≈ 3.22e+01 rad/s²
```

and with nonzero base angular velocity:

```text
velocity_dot RMS       ≈ 7.41e-10
omega_dot_free RMS     ≈ 8.69e-02
omega_dot_locked RMS   ≈ 5.93e-10
tilt_qacc RMS          ≈ 3.32e+01 rad/s²
```

The decisive result was:

> When hypothetical tilt acceleration was removed from the generalized dynamics, the base angular acceleration matched the CasADi rigid-body model to approximately \(10^{-9}\).

Therefore the NMPC equations were not the problem.

The mechanical hinge representation was.

---

## 4.2 New rigid MuJoCo model

A new nominal MJCF was created:

```text
src/tiltable_quadrotor_rl/robot/mjcf/tiltable_quadrotor_rigid.xml
```

The nominal plant now contains:

- one rigid base body,
- one MuJoCo free joint,
- no mechanical rotor-tilt joints,
- no hidden tilt inertial DoFs,
- visual rotor/arm geometry only,
- a body IMU,
- six internal wrench actuators.

Compiled structure:

```text
nq    = 7
nv    = 6
nu    = 6
na    = 0
njnt  = 1
nbody = 2
```

Thus the MuJoCo plant has exactly the intended six mechanical degrees of freedom:

```text
3 translation
+
3 rotation
```

The four tilt angles are no longer mechanical MuJoCo states.

They are actuator states maintained outside the rigid-body generalized coordinates.

---

## 4.3 Meaning of the six MuJoCo actuators

The six MuJoCo controls are an **internal physics interface**:

```text
Fx, Fy, Fz, Tx, Ty, Tz
```

They are not the controller's physical inputs.

The real control input remains:

\[
u=
[f_1,f_2,f_3,f_4,
\alpha_{c1},\alpha_{c2},\alpha_{c3},\alpha_{c4}]
\]

The internal chain is:

\[
(f,\alpha)
\rightarrow
B(\alpha)f
\rightarrow
[F_B,\tau_B]
\rightarrow
\text{MuJoCo rigid-body wrench}
\]

This lets MuJoCo integrate the rigid-body motion while preserving the physical rotor command interface used by the controller.

---

# 5. Servo and actuator representation

The nominal servo model remains:

\[
\dot{\alpha}
=
\frac{\alpha_c-\alpha}{\tau}
\]

with exact discrete propagation:

\[
\alpha_{k+1}
=
\alpha_c+
(\alpha_k-\alpha_c)e^{-\Delta t/\tau}.
\]

The nominal value is:

\[
\tau=0.048\text{ s}.
\]

The full actuator chain is now conceptually clean:

```text
alpha_cmd
    ↓
first-order actuator dynamics
    ↓
alpha
    ↓
rotor direction
    ↓
B(alpha) f
    ↓
body wrench
```

No extra mechanical servo dynamics are silently added by the simulator.

This directly resolves the architectural conflict identified in the 2026-09-14 findings document.

---

# 6. Rigid-plant implementation

Two important modules were added.

## 6.1 `rigid_actuation.py`

Location:

```text
src/tiltable_quadrotor_rl/robot/rigid_actuation.py
```

Responsibilities:

```text
thrust + physical alpha
    ↓
analytical rotor-wrench model
    ↓
body force / torque
    ↓
MuJoCo internal six-axis wrench actuators
```

It reuses the validated allocation geometry rather than duplicating a second wrench model.

---

## 6.2 `rigid_plant.py`

Location:

```text
src/tiltable_quadrotor_rl/robot/rigid_plant.py
```

Main responsibilities:

- combine MuJoCo base state with the external four-element servo state,
- expose the full 17-state NMPC state,
- propagate the first-order servo model,
- apply the corresponding rotor wrench,
- advance the rigid MuJoCo plant.

The 17-state vector is:

\[
x=
[p^W,v^W,q_B^W,\omega^B,\alpha].
\]

The plant step uses a midpoint servo angle for body-wrench application:

```text
alpha(t)
    ↓
exact first-order half-step
    ↓
alpha_mid
    ↓
B(alpha_mid) f
    ↓
MuJoCo 2 ms rigid-body step
    ↓
exact full servo endpoint
```

The midpoint wrench is used because the servo angle moves continuously during the 2 ms physics interval.

The servo endpoint itself remains the exact first-order solution.

---

# 7. Rigid-plant regression tests

A dedicated rigid-plant regression test was added:

```text
tests/test_rigid_plant.py
```

It verifies:

- model dimensions,
- mass,
- inertia,
- gravity behavior,
- hover equilibrium,
- body torque response.

The rigid plant test passed.

The complete project test suite also remained passing after the architecture change.

---

# 8. Instantaneous CasADi vs MuJoCo validation

A direct randomized comparison was performed:

```text
experiments/nmpc/rigid_model_vs_mujoco.py
```

The test sampled arbitrary:

- positions,
- attitudes,
- linear velocities,
- angular velocities,
- servo angles,
- thrust values.

The CasADi derivatives were compared with the new rigid MuJoCo plant.

Results over 100 randomized cases:

```text
position_dot
  RMS = 0
  max = 0

velocity_dot
  RMS = 1.502e-15
  max = 1.066e-14

quaternion_dot
  RMS = 1.676e-16
  max = 6.661e-16

omega_dot
  RMS = 1.192e-14
  max = 4.263e-14
```

These are effectively floating-point roundoff.

Therefore:

\[
\boxed{
\text{CasADi and MuJoCo represent the same instantaneous rigid-body dynamics.}
}
\]

This closed the major model-consistency issue from the previous document.

---

# 9. Finite-time integration validation

Instantaneous derivative agreement is necessary but not sufficient.

A second experiment compared the models over a nontrivial time interval:

```text
experiments/nmpc/finite_time_model_vs_mujoco.py
```

Configuration:

- duration: `0.4 s`,
- nominal MuJoCo timestep: `2 ms`,
- high-resolution CasADi RK4 reference,
- nontrivial initial position,
- nontrivial attitude,
- nonzero linear and angular velocity,
- nonzero initial servo angles,
- deterministic changing actuator commands.

## 9.1 Nominal 2 ms MuJoCo result

```text
position final/max error:
1.666711e-03 m

velocity final/max error:
1.502140e-03 m/s

attitude final/max error:
1.087593e-05 rad

omega final error:
3.757702e-05 rad/s

omega max error:
4.261581e-05 rad/s

alpha final error:
6.195843e-13 rad

alpha max error:
2.774639e-12 rad
```

## 9.2 Temporary 1 ms convergence test

MuJoCo was temporarily reduced to:

```text
1 ms
```

and the high-resolution reference to:

```text
0.125 ms
```

Results:

```text
position final/max error:
8.332369e-04 m

velocity final/max error:
7.536995e-04 m/s

attitude final/max error:
2.719207e-06 rad

omega final error:
9.395407e-06 rad/s

omega max error:
1.065515e-05 rad/s

alpha final error:
3.894662e-14 rad
```

Halving the MuJoCo timestep approximately:

- halved translational error,
- quartered attitude/angular-rate error.

This is the expected convergence behavior for numerical integration.

Therefore the remaining finite-time discrepancy is numerical rather than structural.

The project retained the nominal:

```text
2 ms / 500 Hz
```

MuJoCo timestep.

---

# 10. Long visual plant validation

A 45-second viewer experiment was added:

```text
experiments/visual/rigid_plant_validation_viewer.py
```

It included nine reset-separated scenarios:

1. pure hover,
2. vertical force,
3. body-X force,
4. body-Y force,
5. roll torque,
6. pitch torque,
7. yaw torque,
8. combined 6-D wrench,
9. final hover.

The viewer was also upgraded to automatically save:

```text
artifacts/validation/
    rigid_plant_validation_<timestamp>.csv
    rigid_plant_validation_<timestamp>_summary.txt
```

at 100 Hz telemetry, eliminating the need to manually copy terminal output.

---

# 11. Important interpretation of the X/Y visual behavior

During the open-loop X/Y force tests, the vehicle initially moved in the requested direction and then curved sideways.

This was investigated from the 100 Hz log.

The result was not an axis-sign bug.

For the body-X case, the maneuver ended with approximately:

\[
\phi\approx0.409^\circ
\]

of residual roll.

Returning to a nominal body-frame wrench:

\[
F_B=[0,0,mg]
\]

does not produce a world-vertical force when the body is rolled.

The resulting lateral acceleration is approximately:

\[
g\sin(0.409^\circ)
\approx0.070\text{ m/s}^2.
\]

The measured lateral drift closely matched this prediction.

The Y-force case showed the symmetric pitch/X-direction behavior.

This established that:

- X/Y axis mapping was correct,
- static final-geometry allocation plus servo lag created small transient torque,
- residual attitude transformed nominal body-frame hover thrust into a world-frame lateral force,
- the real solution is closed-loop full-state control.

This became a direct motivation for proceeding to the NMPC rather than further patching the open-loop allocator.

---

# 12. NMPC model finalized

The NMPC continuous model is implemented in:

```text
src/tiltable_quadrotor_rl/control/nmpc_model.py
```

State:

\[
x=
[p^W,v^W,q_B^W,\omega^B,\alpha].
\]

Dimension:

\[
n_x=17.
\]

Input:

\[
u=
[f_1,f_2,f_3,f_4,
\alpha_{c1},\alpha_{c2},\alpha_{c3},\alpha_{c4}].
\]

Dimension:

\[
n_u=8.
\]

Dynamics:

\[
\dot p=v
\]

\[
\dot v=
\frac1m R_{WB}(q)F_B+g_W
\]

\[
\dot q=
\frac12q\otimes[0,\omega]^T
\]

\[
\dot\omega=
I^{-1}
\left(
\tau_B-\omega\times I\omega
\right)
\]

\[
\dot\alpha=
\frac{\alpha_c-\alpha}{\tau}.
\]

Body wrench:

\[
[F_B;\tau_B]
=
B(\alpha)f.
\]

The important modeling choice remains:

> Actual `alpha` determines the physical wrench. `alpha_cmd` only drives the servo dynamics.

---

# 13. General reference-generation architecture

A new module was added:

```text
src/tiltable_quadrotor_rl/control/nmpc_reference.py
```

The reference system is intentionally general rather than hover-specific.

## 13.1 `ReferencePoint`

A reference point contains the desired motion at one instant:

```text
position_W
velocity_W
acceleration_W
quaternion_WB
omega_B
omega_dot_B
```

Thus the trajectory can describe full translational and rotational motion.

---

## 13.2 `ReferenceTrajectory`

`ReferenceTrajectory` is a protocol/interface:

```text
sample(time_s) -> ReferencePoint
```

It does not itself define a path.

Its purpose is to decouple:

```text
trajectory source
```

from:

```text
NMPC controller
```

Any future source can implement the same interface:

- fixed setpoint,
- waypoint trajectory,
- spline,
- circle,
- joystick,
- planner,
- autonomous mission.

The controller only asks:

```text
"What should the vehicle be doing at time t?"
```

---

## 13.3 Feed-forward rigid-body wrench

From the desired motion, a nominal body wrench is calculated.

Translationally:

\[
F_B=
R_{WB}^T m(a_r-g_W).
\]

Rotationally:

\[
\tau_B=
I\dot\omega_r+
\omega_r\times(I\omega_r).
\]

Therefore:

```text
ReferencePoint
    ↓
inverse rigid-body dynamics
    ↓
desired feed-forward body wrench
```

---

## 13.4 Reference allocation

The desired wrench is then mapped to a coarse physical actuator reference.

The static virtual model remains:

\[
w=Az.
\]

A pseudoinverse reference is obtained:

\[
z_r=A^\dagger w_r.
\]

Then:

\[
z_r
\rightarrow
f_r,\alpha_r.
\]

These values are **not commands**.

They are soft references and initial guesses for NMPC.

The controller remains free to deviate from them whenever transient dynamics make that beneficial.

---

## 13.5 Low-thrust singularity refinement

The corrected implementation also includes the low-thrust reference refinement from the target paper.

A nominal threshold of:

\[
1.5\text{ N}
\]

is used for the current geometry.

If one or more reference rotors lie near the poorly conditioned low-thrust region:

- one low-thrust rotor angle is adjusted,
- that rotor's two virtual components are fixed,
- the remaining six virtual components are solved from a \(6\times6\) reduced system,
- the result is checked for numerical conditioning and wrench consistency.

If the reduced solve is poor, the original pseudoinverse reference is retained.

This refinement affects only the actuator **reference**, not the physical control law.

---

## 13.6 Quaternion and tilt-reference continuity

Two continuity issues are handled.

### Quaternion double cover

Because:

\[
q
\]

and:

\[
-q
\]

represent the same orientation, reference quaternion sign is chosen to stay closest to the previous stage.

### Tilt `atan2` branch

Equivalent tilt angles near the \(\pm\pi\) branch are selected to remain closest to the previous reference angle where allowed by the actuator bounds.

This avoids artificial approximately \(2\pi\) reference jumps.

---

# 14. `StageReference`

The reference generator creates one `StageReference` for every prediction node.

It contains:

```text
parameters
x_reference
u_reference
wrench_reference_B
```

### `parameters`

The 21 stagewise acados parameters are:

\[
[
p_r,
v_r,
q_r,
\omega_r,
\alpha_r,
f_r
].
\]

Dimensions:

```text
position       3
velocity       3
quaternion     4
angular rate   3
alpha ref      4
thrust ref     4
----------------
total         21
```

### `x_reference`

The 17-state reference:

\[
x_r=
[p_r,v_r,q_r,\omega_r,\alpha_r]
\]

is used as a useful NLP state-trajectory initialization.

### `u_reference`

The 8-input reference:

\[
u_r=
[f_r,\alpha_r]
\]

is used as an input initialization.

At perfect steady tracking:

\[
\alpha_c=\alpha_r=\alpha.
\]

---

# 15. NMPC optimal-control problem

The acados OCP is defined in:

```text
src/tiltable_quadrotor_rl/control/nmpc_ocp.py
```

It is one general trajectory-tracking OCP.

There are no separate:

```text
hover controller
X controller
Y controller
yaw controller
```

Different flight behaviors are different reference trajectories supplied to the same NMPC.

---

# 16. NMPC error vector and cost

The nonlinear least-squares residual is:

\[
e=
\begin{bmatrix}
p-p_r\\
v-v_r\\
V(q_r^*\otimes q)\\
\omega-\omega_r\\
\alpha-\alpha_r\\
f-f_r\\
\alpha_c-\alpha
\end{bmatrix}.
\]

The path residual therefore contains:

```text
position error                  3
velocity error                  3
quaternion-vector error         3
angular-rate error              3
actual tilt - tilt reference    4
thrust - thrust reference       4
tilt command - actual tilt      4
---------------------------------
total                          24
```

The terminal cost excludes input terms because there is no terminal control \(u_N\).

The initial weights follow the paper baseline:

```text
Q_position:
x,y = 300
z   = 400

Q_velocity:
x,y,z = 10

Q_attitude:
x,y = 300
z   = 600

Q_omega:
x,y,z = 5

Q_alpha:
2

R_thrust:
2

R_delta_alpha:
250
```

The term:

\[
\alpha_c-\alpha
\]

is important.

It penalizes commanding the servo far from its current physical state, while:

\[
\alpha-\alpha_r
\]

still guides the actual actuator toward the desired reference geometry.

---

# 17. Quaternion tracking error

The controller does not use raw quaternion subtraction.

Instead:

\[
q_e=q_r^*\otimes q
\]

and uses the vector component:

\[
V(q_e).
\]

This avoids treating \(q\) and \(-q\) as different physical orientations.

No hard unit-quaternion equality constraint has been added to the OCP.

The measured quaternion is normalized before every solve, and the continuous dynamics preserve quaternion norm analytically.

A hard nonlinear norm equality was deliberately avoided in the first implementation because it would increase optimization complexity without evidence that it is currently necessary.

---

# 18. NMPC constraints

## 18.1 Input constraints

For every rotor:

\[
0\le f_i\le23\text{ N}.
\]

Tilt command:

\[
|\alpha_{c,i}|\le3.15\text{ rad}.
\]

## 18.2 State constraints

World velocity:

\[
|v_i|\le1\text{ m/s}.
\]

Body angular velocity:

\[
|\omega_i|\le6\text{ rad/s}.
\]

Physical tilt state:

\[
|\alpha_i|\le3.15\text{ rad}.
\]

The same physical state constraints are also applied at the terminal shooting node.

---

# 19. NMPC discretization and solver

The current OCP configuration is:

```text
N_horizon = 20
tf        = 2.0 s
stage dt  = 0.1 s
```

The controller itself is called every:

```text
0.01 s = 100 Hz
```

This distinction is important.

Every 10 ms, a new 2-second prediction is solved.

At time \(t\), the trajectory is sampled approximately at:

```text
t
t + 0.1
t + 0.2
...
t + 2.0
```

Then only the first optimized input is applied.

Solver configuration:

```text
integrator:
ERK

Runge-Kutta stages:
4

QP solver:
PARTIAL_CONDENSING_HPIPM

nonlinear solver:
SQP_RTI

Hessian:
GAUSS_NEWTON
```

---

# 20. Warm-start design and correction

An earlier implementation idea manually shifted the previous optimized trajectory by one shooting node between NMPC updates.

That would have been incorrect because:

```text
shooting-node spacing = 100 ms
controller period      = 10 ms
```

Shifting one whole node every 10 ms would advance the warm start ten times too far.

The corrected implementation does not perform that manual 100 ms shift.

Instead:

- the previous acados NLP iterate is kept,
- stage 0 is replaced by the newly measured state,
- stagewise reference parameters are refreshed,
- acados / HPIPM warm-start support is enabled.

On the first solve, the state and input trajectories are initialized from the generated reference horizon.

If a later warm-started solve fails, the controller performs one retry after reinitializing from the current reference trajectory.

If that retry also fails, the controller reports failure.

---

# 21. Runtime controller wrapper

The runtime wrapper is:

```text
src/tiltable_quadrotor_rl/control/nmpc_controller.py
```

One call to:

```text
controller.solve(...)
```

performs:

```text
1. validate and normalize measured state
2. build the N+1 reference horizon
3. update acados stage parameters
4. initialize the NLP iterate if needed
5. enforce x0 = measured state
6. solve SQP-RTI
7. retry once from a clean reference initialization if needed
8. extract u0
9. return thrust[4] + alpha_cmd[4]
10. report status and solve time
```

Only:

\[
u_0
\]

is applied.

The remaining optimized input sequence is a prediction/warm-start object, not an open-loop command sequence sent to the plant.

---

# 22. acados solver generation

The general solver was successfully generated and compiled using:

```text
CasADi 3.7.2
acados commit:
3fec80b744be0d8e5329d23702f06ec432e105a6
```

Build output included successful generation of:

```text
tiltable_quadrotor_nmpc_expl_ode_fun
tiltable_quadrotor_nmpc_expl_vde_forw
tiltable_quadrotor_nmpc_expl_vde_adj

tiltable_quadrotor_nmpc_cost_y_fun
tiltable_quadrotor_nmpc_cost_y_e_fun
```

The final result was:

```text
GENERAL NMPC SOLVER BUILD PASSED
```

acados reported:

```text
acados was compiled without OpenMP
```

This has not been a practical problem because measured NMPC solve times are far below the 10 ms control deadline.

---

# 23. General trajectory implementation

A concrete trajectory provider was added:

```text
src/tiltable_quadrotor_rl/control/nmpc_trajectory.py
```

It implements:

```text
QuinticPositionYawTrajectory
```

through timed waypoints.

Each segment uses:

\[
s(r)=10r^3-15r^4+6r^5.
\]

This gives:

\[
s'(0)=s'(1)=0
\]

and:

\[
s''(0)=s''(1)=0.
\]

Therefore each waypoint is joined with zero velocity and acceleration at the segment boundaries.

The first runtime trajectory exercises:

- X motion,
- Y motion,
- Z motion,
- yaw motion,
- simultaneous multi-axis movement,
- return to the starting pose.

The current trajectory generator keeps desired roll and pitch at zero.

This is a trajectory-generator choice, not an NMPC limitation.

The NMPC reference/OCP architecture supports arbitrary quaternion references.

---

# 24. First full closed-loop runtime

A complete closed-loop experiment was added:

```text
experiments/nmpc/run_closed_loop_general.py
```

Runtime architecture:

```text
MuJoCo rigid plant:
500 Hz / 2 ms

NMPC:
100 Hz / 10 ms

physics steps per NMPC command:
5
```

The NMPC command is therefore zero-order held for exactly five MuJoCo steps.

The 30-second reference sequence was:

| Time | Position `[x,y,z]` m | Yaw |
|---:|---|---:|
| 0 s | `[0.0, 0.0, 1.5]` | `0°` |
| 2 s | `[0.0, 0.0, 1.5]` | `0°` |
| 7 s | `[1.0, 0.7, 1.8]` | `30°` |
| 12 s | `[-0.4, 1.0, 1.3]` | `-20°` |
| 17 s | `[-0.8, -0.5, 1.7]` | `45°` |
| 22 s | `[0.4, -0.7, 1.4]` | `-30°` |
| 27 s | `[0.0, 0.0, 1.5]` | `0°` |
| 30 s | `[0.0, 0.0, 1.5]` | `0°` |

The first and final holds are parts of the same trajectory, not separate hover-specific controller tests.

---

# 25. Closed-loop automatic logging

The runtime writes:

```text
artifacts/nmpc/
    general_nmpc_closed_loop_<timestamp>.csv
    general_nmpc_closed_loop_<timestamp>_summary.txt
```

The CSV includes 100 Hz controller data:

- actual position,
- reference position,
- actual velocity,
- reference velocity,
- actual quaternion,
- reference quaternion,
- Euler angles,
- attitude error,
- angular velocity,
- angular-rate reference,
- actual servo angles,
- thrust commands,
- servo-angle commands,
- solver status,
- solve time,
- deadline-miss flag.

The summary reports:

- completion status,
- solver failures,
- 10 ms deadline misses,
- solve-time statistics,
- position RMS/max error,
- velocity RMS error,
- attitude RMS/max error,
- actuator maxima.

---

# 26. First 30-second closed-loop result

The first complete run finished normally.

Summary:

```text
Run completed normally: True

Final simulation time:
30.010 s

Control updates:
3001

Solver failures:
0

10 ms deadline misses:
0
```

This is the first full end-to-end confirmation that the complete stack works together:

```text
trajectory
→ reference generation
→ acados stage parameters
→ SQP-RTI
→ physical thrust + servo commands
→ servo dynamics
→ rigid MuJoCo plant
→ state feedback
→ repeat
```

---

# 27. NMPC solver timing result

Measured solve times:

```text
mean:
0.8298 ms

p50:
0.8020 ms

p95:
0.9630 ms

max:
2.7020 ms
```

Controller deadline:

```text
10.000 ms
```

Therefore:

- the median solve uses roughly `8%` of the 10 ms control period,
- the 95th-percentile solve uses roughly `10%`,
- the worst observed solve uses roughly `27%`,
- no observed solve missed the 100 Hz deadline.

This provides substantial computational margin on the current machine.

At this stage, rebuilding acados for OpenMP is not justified by timing results.

---

# 28. Closed-loop tracking result

Measured tracking:

```text
position RMS:
0.002596 m

position max:
0.005155 m

velocity RMS:
0.004703 m/s

attitude RMS:
0.362142 deg

attitude max:
0.658064 deg
```

So the nominal controller tracked the multi-axis trajectory with:

- approximately `2.6 mm` RMS position error,
- approximately `5.2 mm` worst position error,
- approximately `4.7 mm/s` RMS velocity error,
- sub-degree attitude error.

The controller was not merely holding a fixed pose during these measurements.

It was simultaneously tracking changing X, Y, Z, and yaw references.

---

# 29. Final-state behavior

At the end of the 30-second trajectory, the vehicle returned essentially exactly to the final reference.

Approximate final position:

\[
[
4.9\times10^{-7},
-5.8\times10^{-7},
1.49999996
]\text{ m}
\]

against:

\[
[0,0,1.5]\text{ m}.
\]

Final position error was approximately:

\[
7.6\times10^{-7}\text{ m}.
\]

Final attitude error was approximately:

\[
3\times10^{-5}\text{ deg}.
\]

Thus there was no significant accumulated drift in the nominal closed loop.

---

# 30. Actuator usage during the closed-loop run

The actuator commands remained far from the modeled limits.

Maximum rotor thrust:

\[
7.580422\text{ N}.
\]

Thrust limit:

\[
23\text{ N}.
\]

Maximum commanded tilt magnitude:

\[
3.295737^\circ.
\]

The actual physical tilt remained similarly small.

Therefore this first trajectory is dynamically mild relative to the available modeled authority.

That is useful for validating the architecture, but it also means this run should not yet be interpreted as a test of operation near actuator saturation.

---

# 31. Servo behavior under NMPC

The maximum command-to-actual tilt separation during the successful run was small.

Approximate maximum:

\[
|\alpha_c-\alpha|
\approx0.17^\circ.
\]

Approximate RMS separation:

\[
0.064^\circ.
\]

This indicates that the current NMPC:

- does not aggressively slam the servo reference,
- respects the servo transient model,
- and is strongly influenced by the \(R_{\Delta\alpha}\) term.

This is qualitatively different from the old standalone dynamic allocator, which could demand extreme transient servo commands when optimizing a short wrench horizon.

---

# 32. Attitude behavior during translation

The vehicle is intended to be omnidirectional.

During the multi-axis translation test, the NMPC did not rely on large body roll/pitch to create lateral motion.

Observed maximum body tilt was approximately:

```text
max |roll|  ≈ 0.63 deg
max |pitch| ≈ 0.59 deg
```

Yaw tracking was particularly accurate:

```text
yaw RMS error:
≈ 0.026 deg

yaw max error:
≈ 0.08 deg
```

Most of the total attitude error therefore came from small roll/pitch deviations during translation.

This is consistent with the intended tiltable-rotor behavior.

---

# 33. Why the earlier X/Y drift problem disappeared

The open-loop X/Y visual test had shown:

```text
requested body-X/Y force
    ↓
servo transient
    ↓
small unintended attitude error
    ↓
return to nominal body-frame mg thrust
    ↓
lateral world-frame acceleration
    ↓
curved trajectory
```

The NMPC closes that loop.

It continuously sees:

\[
p,\quad
v,\quad
q,\quad
\omega,\quad
\alpha
\]

and predicts the servo and rigid-body dynamics.

A residual roll/pitch error is therefore not ignored.

The controller immediately trades thrust and tilt commands to reduce:

- translational error,
- attitude error,
- angular-rate error,
- actuator-reference error.

This is exactly the class of problem that motivated the transition from open-loop/static allocation to full-state NMPC.

---

# 34. Current status of the old dynamic allocator

The dynamic allocator work remains valuable as:

- an engineering baseline,
- a servo-transient study,
- a comparison architecture,
- a diagnostic tool,
- a potential future controller benchmark.

However, it is no longer the preferred main flight-control path.

The previous dynamic-allocation document ended with a plan to investigate:

```text
attainable wrench normalization
+
explicit wrench priority
+
secondary actuator objectives
```

That work is now deprioritized for the main controller.

The NMPC naturally converts:

```text
force errors
→ translational-state consequences

moment errors
→ rotational-state consequences
```

inside one optimization problem.

It therefore avoids much of the artificial Newton-versus-Newton-meter normalization problem that affected the standalone wrench allocator.

The allocator remains useful if a future experiment specifically compares:

```text
classical control allocation
vs
full NMPC
vs
RL
```

but it does not need to be perfected before progressing with the NMPC stack.

---

# 35. Current project file map

## Robot / plant

```text
src/tiltable_quadrotor_rl/robot/parameters.py

src/tiltable_quadrotor_rl/robot/mjcf/
    tiltable_quadrotor_rigid.xml

src/tiltable_quadrotor_rl/robot/
    tilt_servo.py
    rigid_actuation.py
    rigid_plant.py
```

## Allocation

```text
src/tiltable_quadrotor_rl/control/
    allocation.py
    dynamic_allocation.py
```

## NMPC

```text
src/tiltable_quadrotor_rl/control/
    nmpc_model.py
    nmpc_reference.py
    nmpc_ocp.py
    nmpc_controller.py
    nmpc_trajectory.py
```

## Plant / NMPC experiments

```text
experiments/nmpc/
    rigid_model_vs_mujoco.py
    finite_time_model_vs_mujoco.py
    build_general_nmpc.py
    run_closed_loop_general.py
```

## Visual validation

```text
experiments/visual/
    rigid_plant_validation_viewer.py
```

## Regression

```text
tests/
    test_nmpc_model.py
    test_rigid_plant.py
```

The dynamic-allocation experiments from the previous work remain available under:

```text
experiments/allocation/
```

---

# 36. What is now considered validated

## Actuator geometry

Validated:

```text
f + alpha
    ↓
body wrench
```

The analytical wrench model is no longer a current suspect.

## Servo model

Validated:

```text
alpha_cmd
    ↓
first-order servo dynamics
    ↓
alpha
```

for the nominal mathematical model.

## Rigid-body model

Validated both instantaneously and over finite time:

```text
CasADi reduced-order rigid dynamics
≈
MuJoCo rigid plant
```

to numerical integration accuracy.

## Full nominal controller loop

Demonstrated:

```text
general trajectory
→ reference generation
→ acados NMPC
→ first-order servo
→ rigid MuJoCo plant
→ state feedback
```

for a 30-second multi-axis trajectory.

## Real-time computational feasibility on the current machine

Demonstrated with:

```text
0 deadline misses in 3001 controller updates
```

and worst measured solve time:

```text
2.702 ms
```

against:

```text
10 ms
```

available.

---

# 37. What is NOT yet validated

The strong nominal result should not be mistaken for complete controller validation.

The NMPC and MuJoCo plant currently share the same core model assumptions.

The following are not yet established.

## 37.1 Robustness to model mismatch

Not yet systematically tested:

- wrong mass,
- wrong inertia,
- wrong servo time constant,
- wrong thrust scale,
- wrong reaction-torque coefficient,
- center-of-mass offset.

## 37.2 External disturbances

Not yet systematically tested:

- wind,
- external force,
- external moment,
- impulse disturbance,
- payload interaction.

## 37.3 State-estimation imperfections

The current simulation effectively provides exact state information.

Not yet included:

- IMU noise,
- velocity-estimation noise,
- position-estimation noise,
- quaternion noise,
- bias,
- latency,
- packet/update jitter.

## 37.4 Actuator mismatch

Thrust is still treated as instantaneous.

Not yet modeled:

- rotor/motor thrust dynamics,
- thrust slew limit,
- motor lag,
- servo deadband,
- backlash,
- saturation-rate behavior.

## 37.5 Hardware validity of the 48 ms servo value

The structure is justified.

The exact number is not yet hardware-identified.

## 37.6 End-effector / manipulation formulation

The current controller is body/CoG-centered.

The paper's end-effector-centric transform has not yet been added.

## 37.7 Disturbance estimator and model-error compensation

The baseline omits:

- external wrench observer,
- integral/model-error compensation.

These are likely the next paper-derived features to become relevant once deliberate plant/controller mismatch is introduced.

---

# 38. Important interpretation of the current tracking quality

The first closed-loop tracking errors are extremely small.

That is encouraging, but it is also expected to some degree because:

```text
NMPC model
and
nominal simulation plant
```

have been deliberately made internally consistent.

Therefore the current result proves:

- correct system integration,
- correct controller plumbing,
- correct reference flow,
- correct dynamics usage,
- numerically healthy optimization,
- strong nominal tracking,
- sufficient solver speed.

It does **not** yet prove real-world robustness.

The next stage should intentionally break model equality.

---

# 39. Current recommended next development stage

The next major work should be a **robustness / model-mismatch campaign**, not more basic X/Y/hover demonstrations.

A useful progression is:

```text
1. mass mismatch
2. inertia mismatch
3. servo-tau mismatch
4. thrust-scale mismatch
5. external constant force
6. external constant moment
7. impulse disturbance
8. combined mismatch + disturbance
```

For each case, measure:

- RMS / max position error,
- RMS / max attitude error,
- settling behavior,
- actuator saturation,
- solver failures,
- solve time.

This will answer the next meaningful question:

> How much mismatch can the baseline NMPC tolerate before disturbance estimation / integral compensation becomes necessary?

---

# 40. Disturbance-estimation work after robustness baseline

Once baseline robustness is characterized, the next paper-style extensions should be considered.

Likely architecture:

```text
measured state
    ↓
disturbance / model-error estimation
    ↓
estimated external wrench or correction
    ↓
NMPC prediction / reference compensation
```

This should be introduced only after quantifying what problem it is solving.

The project should avoid adding estimator complexity merely because it exists in the reference paper.

---

# 41. Hardware-identification path

Before real hardware control, the project should identify at least:

```text
servo step response
thrust vs command
motor/thrust lag
maximum sustainable thrust
reaction torque coefficient
mass
inertia / approximate inertia
center of mass
```

The existing architecture supports this.

For example, the first-order servo model can keep the same code while replacing:

```text
tau = 0.048 s
```

with an experimentally fitted value.

Likewise, thrust dynamics can be inserted as an additional actuator state later if measurements justify it.

---

# 42. RL relevance

The project remains suitable for future RL comparisons.

The current nominal model is especially convenient because:

- the actuator state is explicit,
- the first-order servo model is cheap,
- the rigid-body state is compact,
- the wrench calculation is analytical,
- parameters can be randomized.

Future domain randomization can include:

```text
mass
inertia
tau
thrust scale
kq/kt
center of mass
sensor noise
latency
external disturbances
```

The same plant can therefore support:

```text
NMPC robustness studies
RL training
NMPC vs RL comparison
hybrid model-based / learned control
```

without returning to the problematic mechanical-hinge abstraction.

---

# 43. Major engineering lessons since the previous documents

## 43.1 The simulator architecture matters as much as the controller equations

The previous rotational mismatch came from hidden simulated mechanical DoFs, not from the NMPC equations.

## 43.2 Matching instantaneous derivatives is not enough

Finite-time convergence testing was needed to establish that the remaining discrepancy behaved like numerical integration error.

## 43.3 Open-loop wrench correctness does not imply trajectory correctness

A nominal body-frame hover wrench can still cause strong world-frame motion when attitude is not level.

## 43.4 Full-state NMPC resolves problems that are awkward at the allocation layer

Servo transient torque, residual attitude, and force-frame coupling become natural state-prediction problems rather than isolated wrench-weighting problems.

## 43.5 Reference allocation is useful without being the controller

The pseudoinverse provides:

- an actuator reference,
- a good initial guess,
- a geometric starting point.

But NMPC remains free to deviate from it.

## 43.6 Good warm starting requires respecting timescales

A 100 ms shooting interval must not be blindly shifted every 10 ms controller update.

## 43.7 The first successful nominal result should trigger robustness testing, not endless axis-by-axis validation

The current system has already demonstrated general multi-axis nominal control.

The next information gain comes from deliberate mismatch.

---

# 44. Current architecture decision

The nominal control stack is now:

\[
\boxed{
\begin{aligned}
&\text{Rigid 6-DoF MuJoCo plant}\\
+&\text{explicit first-order tilt states}\\
+&\text{analytical rotor wrench}\\
+&\text{17-state nonlinear prediction model}\\
+&\text{8 physical NMPC inputs}\\
+&\text{reference pseudoinverse allocation}\\
+&\text{100 Hz acados SQP-RTI controller}
\end{aligned}
}
\]

The old architecture:

```text
mechanical passive tilt hinges
+
externally imposed alpha
```

is rejected for the nominal plant.

The standalone dynamic allocator is retained as a baseline rather than the main flight controller.

---

# 45. Current milestone status

| Area | Status |
|---|---|
| Static allocation | Validated |
| Rotor wrench model | Validated |
| First-order servo model | Validated mathematically |
| Old articulated plant | Rejected |
| Rigid six-DoF plant | Implemented |
| Rigid plant unit tests | Passed |
| CasADi vs rigid MuJoCo instantaneous dynamics | Passed |
| Finite-time model comparison | Passed / numerically convergent |
| Visual rigid-plant validation | Passed |
| Automatic plant telemetry logging | Implemented |
| 17-state CasADi NMPC model | Implemented |
| General reference system | Implemented |
| Feed-forward wrench generation | Implemented |
| Pseudoinverse actuator references | Implemented |
| Low-thrust reference refinement | Implemented |
| Quaternion/tilt reference continuity | Implemented |
| acados OCP | Implemented |
| OCP constraints | Implemented |
| SQP-RTI / HPIPM configuration | Implemented |
| Warm-start handling | Corrected and implemented |
| acados code generation | Passed |
| acados C compilation/linking | Passed |
| Smooth multi-axis trajectory generator | Implemented |
| 500 Hz plant / 100 Hz controller runtime | Implemented |
| Automatic NMPC CSV + summary logging | Implemented |
| 30 s nominal closed-loop run | Passed |
| NMPC solver failures in first run | `0` |
| 10 ms deadline misses in first run | `0` |
| Model-mismatch robustness campaign | Not yet done |
| Disturbance estimator | Deferred |
| Integral/model-error compensation | Deferred |
| Sensor/state-estimation model | Deferred |
| Hardware actuator identification | Deferred |
| End-effector-centric formulation | Deferred |

---

# 46. Quantitative milestone snapshot

## Plant consistency

```text
instantaneous randomized comparison:

position_dot RMS:
0

velocity_dot RMS:
1.502e-15

quaternion_dot RMS:
1.676e-16

omega_dot RMS:
1.192e-14
```

## Finite-time nominal 2 ms integration

```text
position error:
~1.67 mm

velocity error:
~1.50 mm/s

attitude error:
~1.09e-05 rad

omega error:
~4.26e-05 rad/s max
```

## First general NMPC run

```text
duration:
30.010 s

control updates:
3001

solver failures:
0

deadline misses:
0
```

## Solver timing

```text
mean:
0.8298 ms

p95:
0.9630 ms

max:
2.7020 ms
```

## Tracking

```text
position RMS:
2.596 mm

position max:
5.155 mm

velocity RMS:
4.703 mm/s

attitude RMS:
0.362 deg

attitude max:
0.658 deg
```

## Actuator usage

```text
max thrust:
7.580 N

max commanded tilt:
3.296 deg
```

---

# 47. Final conclusion

The project has moved beyond the architecture-selection stage.

The major plant inconsistency identified in the previous findings document has been removed.

The project now has a nominal model/control stack in which:

```text
1. the mechanical plant has only the intended rigid-body DoFs,
2. servo tilt is an explicit actuator state rather than a hidden hinge dynamic,
3. CasADi and MuJoCo agree to numerical precision at the derivative level,
4. finite-time disagreement converges as a numerical-integration effect,
5. the full paper-style NMPC problem can be generated and compiled by acados,
6. the controller runs comfortably within the 100 Hz timing budget,
7. one controller tracks a smooth X/Y/Z/yaw trajectory in closed loop,
8. nominal tracking is millimeter-level in position with sub-degree attitude error,
9. the old open-loop X/Y residual-attitude problem is naturally corrected by full-state feedback.
```

The current engineering question is therefore no longer:

> Can the nominal NMPC architecture work?

The answer is now clearly:

\[
\boxed{\text{Yes, in the current nominal simulation.}}
\]

The next engineering question is:

> How robust is it when the controller model is deliberately wrong and the vehicle is disturbed?

That is the recommended focus of the next project phase.

---

## Final current status

**Nominal rigid plant:** validated.  
**First-order actuator model:** integrated and validated.  
**CasADi prediction model:** validated.  
**acados NMPC:** generated, compiled, and operational.  
**General reference architecture:** operational.  
**Closed-loop multi-axis tracking:** successful.  
**100 Hz real-time budget on current machine:** comfortably met.  
**Primary next milestone:** deliberate model mismatch and disturbance robustness testing.

