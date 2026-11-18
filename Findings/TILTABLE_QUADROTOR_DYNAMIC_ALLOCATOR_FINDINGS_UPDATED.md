# Tiltable Quadrotor RL — Dynamic Allocator Findings and Development Notes

**Project:** `tiltable_quadrotor_rl`  
**Repository:** `~/Projects/tiltable_quadrotor_rl`  
**Current branch target:** `feature/dynamic-actuator-allocation`  
**Document scope:** Dynamic actuator allocation work performed after the validated static allocation baseline  
**Status:** Three-step / 30 ms receding-horizon dynamic allocator implemented and transient-tested. Force transients improved substantially, but moment transients and actuator-usage behavior exposed a normalization / priority problem. Literature review completed; next step is attainable-authority normalization and a cleaner separation of normalization, control priority, and secondary actuator objectives.

---

## 1. Purpose of this work

The original static allocator could solve the geometric control-allocation problem well when rotor tilt angles were treated as if they could change instantaneously.

That assumption is not physically valid.

The tilt servos have first-order dynamics:

\[
\dot{\alpha}
=
\frac{\alpha_c-\alpha}{\tau}
\]

with nominal time constant:

\[
\tau = 0.048 \text{ s}
\]

where:

- \(\alpha\) = actual physical rotor tilt angle,
- \(\alpha_c\) = commanded tilt angle,
- \(\tau\) = servo time constant.

The exact discrete-time servo update is:

\[
\alpha_{k+1}
=
\alpha_c
+
(\alpha_k-\alpha_c)e^{-\Delta t/\tau}
\]

The static allocator could compute an actuator geometry that exactly realizes a desired wrench, but during the transient the actual rotor angles lagged behind the commanded angles. This produced large wrench errors.

The goal of the new dynamic allocator is therefore:

> Choose thrusts and tilt commands while explicitly accounting for where the servos are now and how far they can physically move during the prediction interval.

---

## 2. Relevant validated plant model

The vehicle is a tiltable quadrotor with four independently tilting thrust units.

Nominal parameters currently used:

- Total mass: \(m = 3.039\,\text{kg}\)
- Gravity: \(g = 9.81\,\text{m/s}^2\)
- Weight: \(mg = 29.81259\,\text{N}\)
- Hover thrust per rotor: \(mg/4 = 7.4531475\,\text{N}\)
- Arm length: \(l = 0.275\,\text{m}\)
- Rotor reaction torque ratio: \(k_q/k_t = 0.0165\,\text{m}\)
- Maximum thrust per rotor: \(0 \le f_i \le 23\,\text{N}\)
- Commanded servo tilt limit: \(|\alpha_{c,i}| \le 3.15\,\text{rad}\)
- Mechanical servo joint limit: approximately \(\pm225^\circ\)

The instantaneous wrench for fixed physical tilt angles is:

\[
w = B(\alpha)f
\]

where:

\[
w =
\begin{bmatrix}
F_x & F_y & F_z & \tau_x & \tau_y & \tau_z
\end{bmatrix}^T
\]

and:

\[
f =
\begin{bmatrix}
f_1 & f_2 & f_3 & f_4
\end{bmatrix}^T
\]

The function `thrust_allocation_matrix(alpha)` constructs \(B(\alpha)\).

This analytical model was previously validated against MuJoCo. During the new dynamic transient experiment, the maximum analytical-vs-MuJoCo wrench discrepancy was:

\[
1.528\times10^{-10}
\]

which confirms that the actuator wrench model is not the source of the transient problems observed below.

---

## 3. Why the previous architecture was insufficient

The earlier servo-aware architecture conceptually did:

```text
desired wrench
    ↓
static allocator computes final rotor geometry
    ↓
servo begins moving toward that geometry
    ↓
current physical alpha is observed
    ↓
thrust is repaired using B(current alpha)
```

This approach improved some transients but retained an important structural flaw:

> the desired tilt geometry still came from a static allocator that assumed the final geometry was already available.

The servo lag was treated after the desired tilt was chosen rather than being included directly in the control-allocation decision.

This motivated a new architecture in which the optimizer jointly selects thrust and tilt command while predicting the physical servo state.

---

## 4. One-step dynamic allocator

The new allocator was implemented in:

```text
src/tiltable_quadrotor_rl/control/dynamic_allocation.py
```

The intended interface is:

```text
desired wrench
current physical servo angles
previous thrust commands
previous tilt commands
        ↓
one-step dynamic optimization
        ↓
new thrust commands
new tilt commands
```

The decision vector is:

\[
x =
\begin{bmatrix}
f_1,
f_2,
f_3,
f_4,
\alpha_{c1},
\alpha_{c2},
\alpha_{c3},
\alpha_{c4}
\end{bmatrix}^T
\]

So the optimizer chooses all eight physical actuator commands jointly.

### 4.1 Servo prediction inside the optimizer

For every candidate command vector, the allocator predicts:

\[
\alpha_{\text{pred}}
=
\alpha_c
+
(\alpha_k-\alpha_c)e^{-\Delta t/\tau}
\]

The predicted wrench is then:

\[
w_{\text{pred}}
=
B(\alpha_{\text{pred}})f
\]

This is the key conceptual improvement over the static allocator. The optimizer no longer assumes that the servos instantly reach \(\alpha_c\).

---

## 5. Optimization objective

The first implementation uses `scipy.optimize.least_squares`.

The residual contains three components:

1. normalized wrench error,
2. thrust-command change penalty,
3. tilt-command change penalty.

Conceptually:

\[
J =
\|W_w(w_{\text{pred}}-w_{\text{des}})\|^2
+
\lambda_f\|f-f_{\text{prev}}\|^2
+
\lambda_\alpha\|\alpha_c-\alpha_{c,\text{prev}}\|^2
\]

with nominal regularization:

\[
\lambda_f = 10^{-4}
\]

\[
\lambda_\alpha = 10^{-4}
\]

These regularization terms were intentionally kept weak so that wrench tracking dominates. They are currently interpreted primarily as tie-breakers and continuity aids rather than strong smoothing constraints.

---

## 6. Wrench normalization

The wrench residual uses dimensionless normalization.

For force components:

\[
F_{\text{scale}} = mg
\]

For moment components:

\[
\tau_{\text{scale}} = mg\,l
\]

So the normalized wrench error has the form:

\[
\begin{bmatrix}
F_x/(mg) \\
F_y/(mg) \\
F_z/(mg) \\
\tau_x/(mgl) \\
\tau_y/(mgl) \\
\tau_z/(mgl)
\end{bmatrix}
\]

The default component weights are currently equal.

This is intentionally different from imposing a permanent priority such as vertical body force first or torque first. The current design keeps the allocator body-axis-neutral because the long-term vehicle is intended for omnidirectional flight, where no body-fixed force direction should be assumed universally privileged.

---

## 7. First implementation error discovered

The first run exposed a simple wiring bug in `dynamic_allocation.py`.

The code accidentally used:

```python
B = thrust_allocation_matrix(
    predict_tilt_angles
)
```

which passed the Python function itself rather than the computed numerical angle vector.

This produced:

```text
TypeError: float() argument must be a string or a real number, not 'function'
```

The correct call is:

```python
B = thrust_allocation_matrix(
    predicted_alpha
)
```

This also exposed a weakness in the dynamic regression suite: the initial dynamic test only exercised the servo-prediction function and did not actually call the allocator.

A permanent smoke test was therefore recommended for:

```text
tests/test_allocation_dynamic.py
```

to ensure that the allocator can be called successfully and returns bounded, finite commands.

---

## 8. First isolated one-step test

The first numerical test used:

\[
w_{\text{des}} =
\begin{bmatrix}
20 \\
0 \\
29.81259 \\
0 \\
0 \\
0
\end{bmatrix}
\]

with initial conditions:

\[
\alpha = 0,
\qquad
\alpha_c = 0,
\qquad
f_i = 7.4531475\,\text{N}
\]

The initial prediction interval was:

\[
\Delta t = 10\,\text{ms}
\]

corresponding to the intended 100 Hz controller period.

### 8.1 Result

The allocator returned:

```text
New thrust commands [N]:
[ 1.17179816 16.71855186  1.17179817 17.51425789]
```

Tilt commands:

```text
[+180.4817, +180.4817, +180.4817, -180.4817] deg
```

Predicted physical angles after 10 ms:

```text
[+33.9420, +33.9420, +33.9420, -33.9420] deg
```

Predicted wrench:

```text
Fx = 19.11403 N
Fy ≈ 0
Fz = 30.343887 N
Tx = 0.133849 N m
Ty ≈ 0
Tz = 0.198839 N m
```

Desired wrench:

```text
Fx = 20.00000 N
Fz = 29.81259 N
moments = 0
```

Wrench error:

```text
Fx error = -0.88597 N
Fz error = +0.53130 N
Tx error = +0.13385 N m
Tz error = +0.19884 N m
```

Error norm:

\[
1.060506
\]

The optimizer converged successfully in 11 function evaluations.

---

## 9. Interpretation of the 10 ms result

The extreme \(\pm180^\circ\) commands were not immediately interpreted as a failure.

For:

\[
\tau = 48\,\text{ms}
\]

and:

\[
\Delta t = 10\,\text{ms}
\]

we have:

\[
e^{-0.01/0.048} \approx 0.812
\]

Therefore the servo closes only about:

\[
1-0.812 = 18.8\%
\]

of the command error during the first 10 ms.

The optimizer was effectively saying:

> If I only care about what happens 10 ms from now, the fastest way to create useful horizontal authority is to command maximum tilt immediately.

This indicated possible short-horizon myopia rather than an obviously bad model.

---

## 10. Prediction-horizon sweep

To isolate the effect of look-ahead time, the same one-step optimization problem was run with identical weights and regularization, but with prediction intervals of:

- 10 ms,
- 20 ms,
- 30 ms,
- 50 ms.

The controller update rate was conceptually still 100 Hz. The only quantity being varied was how far ahead the servo state was predicted when evaluating a candidate command.

Experiment:

```text
experiments/allocation/dynamic_allocator_horizon_sweep.py
```

---

## 11. Horizon sweep results

### 11.1 10 ms prediction

```text
Thrust [N]:
[ 1.171798 16.718552  1.171798 17.514258]

Tilt command [deg]:
[+180.482, +180.482, +180.482, -180.482]

Predicted physical tilt [deg]:
[+33.942, +33.942, +33.942, -33.942]

Predicted wrench:
Fx = 19.11403
Fz = 30.343887
Tx = 0.133849
Tz = 0.198839

Error norm = 1.060506
```

### 11.2 20 ms prediction

```text
Thrust [N]:
[ 2.329288 15.995786  2.329288 16.183308]

Tilt command [deg]:
[ +0.814, +121.825, +0.814, -104.141]

Predicted physical tilt [deg]:
[ +0.277, +41.513, +0.277, -35.487]

Predicted wrench:
Fx = 19.996567
Fz = 29.813461
Tx = 0.000099
Tz = 0.000043

Error norm = 0.003544
```

### 11.3 30 ms prediction

```text
Thrust [N]:
[ 4.168081 14.497847  4.168081 14.879000]

Tilt command [deg]:
[ +0.887, +98.177, +0.887, -86.835]

Predicted physical tilt [deg]:
[ +0.412, +45.627, +0.412, -40.356]

Predicted wrench:
Fx = 19.997614
Fz = 29.813146
Tx = 0.000066
Tz = 0.000019

Error norm = 0.002451
```

### 11.4 50 ms prediction

```text
Thrust [N]:
[ 5.780136 13.273546  5.780136 13.825376]

Tilt command [deg]:
[ +0.416, +77.312, +0.416, -69.987]

Predicted physical tilt [deg]:
[ +0.269, +50.031, +0.269, -45.291]

Predicted wrench:
Fx = 19.998353
Fz = 29.812872
Tx = 0.000046
Tz = 0.000004

Error norm = 0.001672
```

---

## 12. Main finding from the horizon sweep

The aggressive \(\pm180^\circ\) commands were primarily caused by the very short 10 ms prediction horizon.

As the prediction interval increased:

- commanded tilt magnitude decreased,
- thrust distribution became more balanced,
- unwanted moments nearly disappeared,
- endpoint wrench tracking improved dramatically.

| Prediction horizon | Largest tilt command | Approx. largest predicted physical tilt | Wrench error norm |
|---:|---:|---:|---:|
| 10 ms | 180.5° | 33.9° | 1.060506 |
| 20 ms | 121.8° | 41.5° | 0.003544 |
| 30 ms | 98.2° | 45.6° | 0.002451 |
| 50 ms | 77.3° | 50.0° | 0.001672 |

This provided strong evidence that the basic dynamic allocation formulation was working.

It also indicated that the optimizer benefits from considering servo evolution beyond a single 10 ms control period.

---

## 13. Why 30 ms was selected as the next baseline

The 20 ms solution was already very accurate, but still required relatively aggressive tilt commands.

The 50 ms solution produced gentler commands, but looks farther into the future and may become less responsive to rapidly changing wrench requests.

The 30 ms solution was selected as a useful middle ground for the next experiment because it gave:

- near-perfect predicted wrench,
- near-zero moment error,
- less extreme commands than 10 or 20 ms,
- a modest look-ahead duration.

At this stage this was an experimental choice, not yet a final design parameter.

---

## 14. Important distinction: controller period vs preview horizon

The controller period remained:

\[
10\,\text{ms}
\]

or:

\[
100\,\text{Hz}
\]

The 30 ms value was not a slower controller period.

Instead, the intended logic was:

```text
t = 0 ms:
observe current alpha
predict candidate command effect at t + 30 ms
optimize
apply command

t = 10 ms:
observe new real alpha
again predict to t + 30 ms
optimize again
apply new command
```

So the moving prediction window is:

```text
at t = 0 ms   → look toward 30 ms
at t = 10 ms  → look toward 40 ms
at t = 20 ms  → look toward 50 ms
...
```

This was referred to as a **receding-preview allocator**.

However, it was not yet true finite-horizon MPC because each optimization still selected one constant command and judged only the wrench at one future endpoint.

---

## 15. Receding-preview transient experiment

The next experiment tested the 30 ms preview allocator over time.

Experiment file:

```text
experiments/allocation/receding_preview_allocator_transient.py
```

Simulation rates:

- servo / plant update: 500 Hz,
- physics timestep: 2 ms,
- allocator update: 100 Hz,
- control interval: 10 ms,
- allocator preview: 30 ms.

The rigid vehicle body was intentionally held fixed.

The purpose was to isolate:

- servo dynamics,
- thrust allocation,
- wrench generation,
- optimizer behavior.

This avoided mixing body motion and outer-loop flight dynamics into the allocator diagnosis.

---

## 16. Transient wrench schedule

The desired wrench profile was:

```text
0.0–0.5 s:
hover

0.5–1.5 s:
+20 N body-X force
while maintaining hover support

1.5–2.0 s:
return to hover
```

So:

\[
w_{\text{hover}} =
\begin{bmatrix}
0\\0\\mg\\0\\0\\0
\end{bmatrix}
\]

and:

\[
w_{\text{forward}} =
\begin{bmatrix}
20\\0\\mg\\0\\0\\0
\end{bmatrix}
\]

---

## 17. Forward transition results

Immediately before the step:

```text
t = 0.490 s
Fx = 0
Fz = 29.813 N
moments = 0
alpha = 0°
```

At:

```text
t = 0.500 s
```

the desired wrench changes to +20 N forward.

The allocator chooses:

```text
alpha_cmd:
[ +0.887, +98.177, +0.887, -86.835 ] deg
```

and thrust:

```text
[ 4.1681, 14.4978, 4.1681, 14.8790 ] N
```

The physical rotors are still at zero tilt at that exact instant. Therefore the actual current wrench is:

```text
Fx = 0.000 N
Fz = 37.713 N
Tx = -0.1048 N m
Tz = +0.3472 N m
```

Yet the allocator predicts, at +30 ms:

```text
Fx ≈ 19.9976 N
Fz ≈ 29.8131 N
moments ≈ 0
```

This exposed the most important limitation of the current preview formulation.

---

## 18. Forward transition time history

| Time [s] | Fx desired [N] | Fx actual [N] | Fz actual [N] | Tx [N m] | Tz [N m] | max physical tilt |
|---:|---:|---:|---:|---:|---:|---:|
| 0.500 | 20 | 0.000 | 37.713 | -0.1048 | 0.3472 | 0.00° |
| 0.510 | 20 | 8.360 | 36.974 | -0.0301 | 0.1657 | 18.46° |
| 0.520 | 20 | 13.050 | 35.304 | -0.0176 | 0.0734 | 30.49° |
| 0.530 | 20 | 15.721 | 33.779 | -0.0160 | 0.0296 | 38.42° |
| 0.550 | 20 | 18.248 | 31.756 | -0.0123 | 0.0004 | 47.26° |
| 0.600 | 20 | 19.764 | 30.113 | -0.0025 | -0.0033 | 53.67° |
| 0.750 | 20 | 19.999 | 29.814 | ≈0 | -0.0001 | 54.80° |
| 1.000 | 20 | 20.000 | 29.813 | ≈0 | ≈0 | 54.80° |

The final settled tracking is excellent.

However, the initial transient contains a large vertical-force overshoot.

---

## 19. Vertical-force overshoot during forward transition

At the instant the forward command is introduced:

\[
F_z = 37.713\,\text{N}
\]

instead of:

\[
29.81259\,\text{N}
\]

The vertical-force error is:

\[
7.9004\,\text{N}
\]

For the 3.039 kg vehicle, if the body were free, the approximate vertical acceleration error would be:

\[
a_z
=
\frac{37.713-29.81259}{3.039}
\approx
2.6\,\text{m/s}^2
\]

upward.

This is not acceptable as a long-term dynamic allocation behavior.

---

## 20. Return-to-hover transition

The second important test occurs at:

```text
t = 1.500 s
```

Immediately before the command change, the forward solution is essentially perfectly settled.

The physical rotor angles are approximately:

```text
[-0.22, +54.80, -0.22, -50.17] deg
```

The desired wrench suddenly becomes hover again.

The allocator commands:

```text
[-0.743, -58.687, -0.743, +62.169] deg
```

with thrust approximately:

```text
[4.7431, 10.1700, 4.7431, 10.1702] N
```

The current actual wrench becomes:

```text
Fx = 16.12084 N
Fz = 21.86210 N
Tx = 0.08664 N m
Tz = -0.07991 N m
```

But the predicted +30 ms wrench is almost perfect hover:

```text
Fx ≈ 0.00374 N
Fz ≈ 29.813 N
moments ≈ 0
```

Again, the optimizer is satisfied by a good future endpoint while allowing a poor immediate transient.

---

## 21. Return-to-hover time history

| Time [s] | Fx actual [N] | Fz actual [N] | Tx [N m] | Tz [N m] | max physical tilt |
|---:|---:|---:|---:|---:|---:|
| 1.500 | 16.121 | 21.862 | 0.0866 | -0.0799 | 54.80° |
| 1.510 | 10.543 | 26.862 | 0.0622 | -0.0396 | 33.46° |
| 1.520 | 6.483 | 28.750 | 0.0401 | -0.0198 | 20.75° |
| 1.530 | 3.904 | 29.434 | 0.0248 | -0.0103 | 13.19° |
| 1.550 | 1.391 | 29.765 | 0.0090 | -0.0031 | 6.01° |
| 1.600 | 0.104 | 29.812 | 0.0007 | -0.0002 | 2.38° |
| 1.750 | 0.000 | 29.813 | ≈0 | ≈0 | 2.10° |

The return transient still clearly shows that endpoint-only prediction is insufficient.

---

## 22. Quantitative metrics from the transient simulation

Measured results:

```text
Settled forward Fx RMS error:
0.028558 N
```

```text
Settled forward Fz RMS error:
0.036523 N
```

```text
Peak forward-transition Fx error:
20.000000 N
```

The 20 N value is unavoidable at the exact first instant because the rotors physically cannot produce body-X thrust before they tilt.

More important is:

```text
Peak forward-transition Fz error:
7.900419 N
```

```text
Peak moment norm during forward segment:
0.362649 N m
```

```text
Peak moment norm after return-to-hover:
0.117867 N m
```

The analytical model consistency remained excellent:

```text
Maximum B(alpha) vs MuJoCo error:
1.528e-10
```

The nonlinear optimizer required at most:

```text
32 function evaluations
```

in this experiment.

---

## 23. Core diagnosis

The one-step / endpoint-preview allocator has now demonstrated two important properties.

### 23.1 What works

The allocator:

- correctly models servo lag,
- jointly chooses thrust and tilt command,
- respects actuator bounds,
- finds very accurate future wrench solutions,
- converges reliably in the tested cases,
- matches MuJoCo extremely well through the analytical model,
- achieves excellent steady-state tracking,
- avoids the false assumption that commanded tilt is immediately achieved.

### 23.2 What does not work

The current 30 ms preview objective only evaluates:

\[
w(t+30\text{ ms})
\]

It does not directly penalize the wrench at:

\[
t+10\text{ ms}
\]

or:

\[
t+20\text{ ms}
\]

As a result, it can choose commands that produce:

- large temporary \(F_z\) overshoot,
- temporary forward-force error,
- temporary body moments,

as long as the 30 ms endpoint becomes excellent.

This is now the dominant structural limitation.

---

## 24. Why arbitrary weight tuning is not the next step

It would be possible to reduce some transient errors by manually increasing:

- \(F_z\) weight,
- torque weights,
- angle regularization,
- thrust regularization.

However, doing that now would risk hiding the real issue.

The current evidence shows that the main problem is not simply insufficient weight on vertical force or attitude moments.

The fundamental issue is:

> the optimizer only cares about one future endpoint rather than the full transient trajectory.

Therefore the next step should be a structural fix before any major weight tuning.

---

## 25. Implemented next architecture: 3-step receding-horizon allocator

The proposed finite-horizon allocator was implemented in:

```text
src/tiltable_quadrotor_rl/control/dynamic_allocation.py
```

The new allocator plans three future 10 ms control intervals:

\[
0\rightarrow10\text{ ms},
\qquad
10\rightarrow20\text{ ms},
\qquad
20\rightarrow30\text{ ms}
\]

while the allocator itself still runs at:

\[
100\,\text{Hz}
\]

The horizon therefore remains:

\[
30\,\text{ms}
\]

but now contains three independently optimized actuator commands rather than one constant command held conceptually across the whole preview.

For each interval \(j\), the optimizer chooses:

\[
f_j\in\mathbb{R}^4
\]

and:

\[
\alpha_{c,j}\in\mathbb{R}^4
\]

giving:

\[
3\times 8 = 24
\]

decision variables.

The planned command sequence is:

\[
(f_0,\alpha_{c,0}),
\quad
(f_1,\alpha_{c,1}),
\quad
(f_2,\alpha_{c,2})
\]

Only the first pair:

\[
(f_0,\alpha_{c,0})
\]

is actually applied. After 10 ms, the real actuator state is measured again and the full horizon is re-optimized.

This is the receding-horizon principle.

---

## 26. Wrench evaluation at both ends of every interval

An important refinement was made before implementing the horizon allocator.

It is not sufficient to score only:

\[
w_{10}, w_{20}, w_{30}
\]

because thrust is assumed to change immediately while the tilt servo moves gradually.

For each interval, the allocator therefore evaluates the wrench at both the start and the end.

For interval \(j\):

\[
w_{j,\text{start}}
=
B(\alpha_j)f_j
\]

Then the servo state is propagated for one controller interval:

\[
\alpha_{j+1}
=
g(\alpha_j,\alpha_{c,j})
\]

using the exact first-order servo model.

The end wrench is:

\[
w_{j,\text{end}}
=
B(\alpha_{j+1})f_j
\]

For a three-step horizon this gives six wrench samples:

```text
interval 0 start
interval 0 end
interval 1 start
interval 1 end
interval 2 start
interval 2 end
```

This specifically prevents a future-perfect solution from completely hiding a poor instantaneous wrench immediately after changing thrust.

---

## 27. New horizon objective

The finite-horizon residual contains:

- six wrench-error samples,
- three thrust-command changes,
- three tilt-command changes.

The wrench part is conceptually:

\[
J_w
=
\sum_{j=0}^{2}
\left(
\|e_{w,j,\text{start}}\|^2
+
\|e_{w,j,\text{end}}\|^2
\right)
\]

The continuity terms remain:

\[
J_f
=
\lambda_f
\sum_{j=0}^{2}
\|f_j-f_{j-1}\|^2
\]

and:

\[
J_\alpha
=
\lambda_\alpha
\sum_{j=0}^{2}
\|\alpha_{c,j}-\alpha_{c,j-1}\|^2
\]

where the actually applied previous command is used for \(j=0\).

The implementation uses sample-count normalization so simply increasing the number of horizon samples does not automatically increase the total wrench cost relative to the regularization terms.

For three intervals, the optimizer has:

\[
24
\]

decision variables and:

\[
60
\]

least-squares residual components.

---

## 28. First isolated 3-step optimization

Experiment:

```text
experiments/allocation/multi_step_dynamic_allocator.py
```

The same initial forward-wrench request was used:

\[
w_{\text{des}}
=
\begin{bmatrix}
20 & 0 & 29.81259 & 0 & 0 & 0
\end{bmatrix}^T
\]

starting from zero tilt and hover thrust.

The optimizer converged successfully in:

```text
18 function evaluations
```

with cost:

```text
0.0400569994
```

The first command to apply was:

```text
Thrust [N]:
[0.00000, 16.39776, 0.00000, 16.38264]

Tilt command [deg]:
[+0.006, +180.482, +0.006, -162.315]
```

So the optimizer still used aggressive servo commands, but the important change was the planned wrench trajectory.

### Interval 0: 0–10 ms

Physical tilt:

```text
start:
[0, 0, 0, 0] deg

end:
[~0, +33.942, ~0, -30.526] deg
```

Wrench at interval start:

```text
Fx = 0.000 N
Fz = 32.7804 N
Tx = 0.00416 N m
Tz = 0.54088 N m
```

Wrench at interval end:

```text
Fx = 17.47686 N
Fz = 27.71568 N
Tx = 0.14855 N m
Tz = 0.22778 N m
```

### Interval 1: 10–20 ms

The plan reached:

```text
start:
Fx = 19.14858 N
Fz = 30.36687 N

end:
Fx = 19.98292 N
Fz = 29.76624 N
```

with near-zero moment error at the end.

### Interval 2: 20–30 ms

The plan was essentially exact:

```text
start:
Fx = 19.99942 N
Fz = 29.81306 N

end:
Fx = 20.00050 N
Fz = 29.81217 N
```

This confirmed that scoring the transient trajectory materially changed the optimizer behavior.

---

## 29. Full receding-horizon transient experiment

The full closed-loop actuator-layer experiment was then repeated using the 3-step allocator.

Experiment:

```text
experiments/allocation/receding_horizon_allocator_transient.py
```

Configuration:

- servo simulation: 500 Hz,
- physics / servo step: 2 ms,
- allocator: 100 Hz,
- horizon: 3 steps,
- horizon duration: 30 ms,
- vehicle rigid body fixed,
- thrust still assumed instantaneous.

The desired-wrench schedule remained identical:

```text
0.0–0.5 s:
hover

0.5–1.5 s:
+20 N body-X force while maintaining mg vertical force

1.5–2.0 s:
return to hover
```

This made the result directly comparable with the previous 30 ms endpoint-preview allocator.

---

## 30. Receding-horizon transient result

The analytical wrench model remained extremely consistent with MuJoCo:

\[
\max |B(\alpha)f-w_{\text{MuJoCo}}|
=
3.669\times10^{-10}
\]

Maximum nonlinear optimizer effort was:

```text
21 function evaluations
```

### 30.1 Forward transition

At the instant of the command step:

```text
t = 0.500 s

Fx = 0.000 N
Fz = 32.780 N
Tx = 0.0042 N m
Tz = 0.5409 N m
```

After only 10 ms:

```text
t = 0.510 s

Fx = 19.149 N
Fz = 30.367 N
Tx = 0.1623 N m
Tz = 0.2498 N m
```

After 20 ms:

```text
t = 0.520 s

Fx = 19.999 N
Fz = 29.813 N
moments ≈ 0
```

By 30 ms the wrench was effectively exact.

This is a major improvement in force response compared with the endpoint-preview allocator.

### 30.2 Settled forward condition

The steady solution became approximately:

```text
physical tilt:
R1 ≈ 0°
R2 ≈ +37.31°
R3 ≈ 0°
R4 ≈ -30.44°

thrust:
R1 ≈ 0.01 N
R2 ≈ 17.97 N
R3 ≈ 0.01 N
R4 ≈ 17.97 N
```

and the desired wrench was reproduced essentially exactly.

This also exposed a new actuator-usage behavior: the optimizer strongly prefers one rotor pair while almost unloading the other pair.

### 30.3 Return to hover

At:

```text
t = 1.500 s
```

the allocator returned almost immediately to the requested force:

```text
Fx = 0.23469 N
Fz = 29.77853 N
Tx = -0.10594 N m
Tz = -0.41428 N m
```

At:

```text
t = 1.510 s
```

the force error was already tiny:

```text
Fx = 0.0115 N
Fz = 29.8153 N
```

and by:

```text
t = 1.520 s
```

the wrench was essentially exact.

However, the return transition contained a significant moment transient.

---

## 31. Quantitative comparison with endpoint-preview baseline

### Force tracking

Previous 30 ms endpoint-preview allocator:

```text
Peak forward Fz error:
7.900419 N
```

New 3-step receding-horizon allocator:

```text
Peak forward Fz error:
2.967812 N
```

Improvement:

\[
62.43\%
\]

The settled forward errors became numerically negligible:

```text
Settled Fx RMS error:
0.000000 N

Settled Fz RMS error:
0.000000 N
```

Return-to-hover force errors were also very small:

```text
Peak return Fx error:
0.234686 N

Peak return Fz error:
0.048913 N
```

### Moment tracking

The moment result moved in the wrong direction.

Previous endpoint-preview forward peak moment norm:

```text
0.362649 N m
```

New horizon allocator:

```text
0.540893 N m
```

which is approximately:

```text
49.15% worse
```

Previous return-to-hover peak moment norm:

```text
0.117867 N m
```

New horizon allocator:

```text
0.427608 N m
```

which is approximately:

```text
262.79% worse
```

Therefore the horizon architecture solved a large part of the force-transient problem but exposed an important force-versus-moment tradeoff in the objective.

---

## 32. New actuator-configuration issue

The horizon experiment exposed another behavior that should not be confused with wrench tracking.

During the forward solution, approximately two rotors carry nearly all of the thrust.

After returning to hover, the optimizer finds another exact wrench solution in which a different rotor pair carries most of the load.

Some low-thrust / unused rotors remain tilted rather than returning to a nominal configuration.

This is mathematically reasonable under the current objective.

If a rotor produces almost no thrust, its orientation has almost no effect on the primary wrench. Moving the servo would incur an angle-change penalty, so the optimizer can prefer leaving it where it is.

This reveals a separate **secondary allocation problem**:

- preferred actuator configuration,
- rotor-load distribution,
- servo centering,
- power / energy use,
- distance from saturation,
- singularity margin.

These should not be solved by distorting the primary wrench-tracking weights.

---

## 33. Re-evaluation of the original \(mg/mgl\) normalization

Up to this point, the wrench residual was normalized using:

\[
s_F = mg
\]

for all force axes, and:

\[
s_\tau = mgl
\]

for all moment axes.

For this vehicle:

\[
mg \approx 29.8126\,\text{N}
\]

and:

\[
mgl \approx 8.1985\,\text{N\,m}
\]

The normalization therefore made the forward-transition yaw moment:

\[
0.5409\,\text{N\,m}
\]

appear as only approximately:

\[
\frac{0.5409}{8.1985}
\approx 0.066
\]

normalized error.

The initial vertical-force error:

\[
32.7804-29.8126
=
2.9678\,\text{N}
\]

appeared as:

\[
\frac{2.9678}{29.8126}
\approx 0.100
\]

normalized error.

So the optimizer rationally preferred reducing the vertical-force error even if that required a significant yaw moment.

This does **not** mean the optimizer is malfunctioning.

It means the normalization defines a tradeoff that is not sufficiently grounded in the actual vehicle's available control authority.

---

## 34. Inertia-based normalization was considered, but not adopted

One possible alternative considered was to normalize torque using an inertia-based characteristic angular acceleration:

\[
\tau_{\text{scale},i}
=
I_i\frac{g}{l}
\]

This would give smaller torque scales and therefore penalize moment error more strongly.

Although this is physically interpretable, it still requires choosing:

\[
\frac{g}{l}
\]

as a characteristic angular acceleration.

There is no universal reason that \(g/l\) is the uniquely correct angular-acceleration scale for this allocator.

Therefore this approach is **not being adopted as the current baseline**.

The literature review below suggests a cleaner separation.

---

# 35. Literature review: what should normalization mean?

A literature review was performed specifically to answer:

> How should force, moment, and actuator quantities be normalized in a redundant / dynamic control-allocation problem without hiding arbitrary priorities inside unit conversions?

The main finding is that three concepts should be treated separately:

1. numerical / physical normalization,
2. control-objective priority,
3. secondary actuator-use objectives.

Our current implementation partially mixes these concepts.

---

## 36. Prioritized WLS control allocation: priorities should be explicit

Smeur, Höppener, and De Wagter use Weighted Least Squares control allocation for a saturated quadrotor and explicitly use weighting to prioritize control objectives.

For their conventional quadrotor, they argue for a priority order of:

```text
roll / pitch
then thrust
then yaw
```

The important lesson for this project is **not** to copy that fixed priority order.

Our vehicle is omnidirectional and should not permanently inherit the assumptions of a conventional fixed-rotor quadrotor.

The useful principle is:

> If one wrench component is more important than another, that should be represented explicitly as a control-priority choice, not hidden inside dimensional normalization.

Therefore a future formulation should conceptually separate:

\[
S_w
\]

for normalization from:

\[
P
\]

for control priority.

A generic normalized wrench residual becomes:

\[
r_w
=
P\,S_w\,(w-w_{\text{des}})
\]

where:

- \(S_w\) defines comparable dimensionless scales,
- \(P\) expresses deliberate task / safety priority.

The neutral baseline should use:

\[
P = I_6
\]

until evidence justifies a different priority structure.

---

## 37. Attainable-control normalization

Recent control-allocation work by Hafner, Myschik, and Holzapfel describes normalization of both the actuator inputs and pseudo-control space.

Their control-effectiveness data are normalized with respect to:

- input limits,
- maximum attainable pseudo-control.

This gives normalized actuator limits of approximately:

\[
[-1,1]
\]

and an attainable pseudo-control set whose axis extremes are also normalized to approximately:

\[
[-1,1].
\]

This provides a strong conceptual direction for our wrench allocator.

Instead of using:

\[
mg
\]

and:

\[
mgl
\]

as convenient dimensional scales, define wrench scales from the **actual attainable wrench authority of the vehicle**.

Conceptually:

\[
S_w
=
\operatorname{diag}
\left(
\frac{1}{s_{F_x}},
\frac{1}{s_{F_y}},
\frac{1}{s_{F_z}},
\frac{1}{s_{\tau_x}},
\frac{1}{s_{\tau_y}},
\frac{1}{s_{\tau_z}}
\right)
\]

where each \(s_i\) is derived from a physically attainable range for that wrench component.

Then:

```text
0.10 normalized Fx error
```

means approximately:

> 10% of representative available Fx authority.

Likewise:

```text
0.10 normalized yaw-moment error
```

means approximately:

> 10% of representative available yaw-moment authority.

That is a clearer neutral comparison than equating newtons and newton-metres through \(mgl\).

---

## 38. Tilt-rotor literature: normalize actuator motion using actuator limits

Cuniato et al., in *Allocation for Omnidirectional Aerial Robots: Incorporating Power Dynamics*, specifically address allocation for tilt-rotor aerial robots.

Their work highlights several points directly relevant to this project:

- geometric allocation neglects actuator dynamics and limits,
- differential allocation can exploit system redundancy,
- actuator dynamics should be incorporated into allocation,
- actuator-motion variables can be normalized using their actual limits,
- normalized actuator variables improve numerical conditioning and reduce dependence on heuristic weighting.

In their formulation, actuator velocities are normalized relative to minimum and maximum actuator-rate limits so the normalized variables lie approximately in:

\[
[-1,1].
\]

This is especially relevant because tilt-arm rates and propeller acceleration can differ by orders of magnitude numerically.

For this project, the corresponding lesson is:

> actuator regularization should be based on actual actuator capability, especially rate / dynamic capability, rather than arbitrary raw units.

Our current thrust-change normalization:

\[
\frac{\Delta f}{f_{\max}}
\]

is already partly aligned with this idea.

Our current servo-change normalization:

\[
\frac{\Delta\alpha_c}{\alpha_{\max}}
\]

is less satisfactory for dynamic allocation because the scarce transient resource is primarily **how fast the physical servo can move**, not simply its total angle range.

This should later be revisited using an identified or modeled servo-rate capability.

---

## 39. Dynamic control-allocation literature: actuator dynamics belong in the allocator

Several recent dynamic control-allocation works reinforce the architectural direction already taken here.

### 39.1 Dynamic Control Allocation for Dual-Tilt UAV Platforms

Sorge et al. formulate a hierarchical structure with:

- a high-level controller generating a desired wrench,
- a dynamic control allocator mapping that wrench to physical actuators,
- explicit actuator dynamics,
- actuator saturation,
- redundancy used to optimize actuator state.

This supports the design choice made in this project to keep:

```text
desired wrench
```

as the high-level interface while placing servo dynamics inside the allocator.

### 39.2 Closed-loop incremental dynamic control allocation

Jing and Ma formulate dynamic allocation while accounting for actuators with limited bandwidth and both magnitude and rate constraints.

This further supports treating actuator dynamics and constraints as first-class parts of allocation rather than post-processing a static solution.

### 39.3 Receding-horizon nullspace optimization

Pretto et al. propose receding-horizon, actuator-aware allocation for omnidirectional UAVs.

Their method forward-simulates actuator dynamics over a prediction horizon and uses redundancy / nullspace optimization to improve actuator behavior while preserving the desired body wrench.

The architectural similarity to our current direction is important:

```text
predict actuator behavior over time
+
preserve the primary wrench objective
+
use redundancy for secondary actuator objectives
```

This strongly supports separating primary wrench tracking from secondary goals such as load balancing, command smoothness, and preferred actuator configuration.

---

## 40. Relation to the target tiltable-quadrotor paper

Li et al., *Effector-Centric NMPC of Tiltable-Multirotors for Offset-Free Omnidirectional Aerial Manipulation*, provide an especially relevant reference platform.

Their work demonstrates:

- a four-rotor one-DoF-per-arm tiltable configuration,
- full six-degree-of-freedom wrench generation,
- importance of null-space redundancy,
- singularity traversal under physical constraints,
- onboard NMPC operation at 100 Hz.

The key project-level lesson is that a full predictive controller can reason about the effect of actuation on the vehicle trajectory itself.

Our present allocator is intentionally narrower:

```text
desired wrench
    ↓
dynamic control allocator
    ↓
physical actuator commands
```

Therefore we still need an explicit way to compare force and moment tracking errors inside the allocator.

Attainable-authority normalization is currently considered the best neutral baseline for that narrower problem.

Longer term, a full NMPC or RL controller that includes rigid-body state evolution can express motion priorities at the state / task level rather than relying entirely on wrench-error weighting.

---

# 41. Revised design principle: separate three different layers

The current best design principle is:

### Layer A — normalization

Answer:

> What does one unit of normalized error mean?

Use physical / attainable capability.

For wrench:

\[
S_w
\]

should be based on representative attainable wrench authority.

For actuator motion, normalization should be based on physical actuator limits / rates.

### Layer B — control priority

Answer:

> If not all desired wrench components can be achieved simultaneously, which matter more?

Use an explicit matrix or hierarchy:

\[
P
\]

Do not hide this inside normalization.

Initial neutral baseline:

\[
P=I_6
\]

### Layer C — secondary actuator objective

Answer:

> Among actuator solutions that give essentially the same primary wrench performance, which physical configuration do we prefer?

Examples:

- distribute rotor load,
- reduce power,
- avoid actuator saturation,
- avoid singular configurations,
- center unused servos,
- preserve future wrench authority,
- reduce command motion.

These should be solved as secondary objectives rather than by corrupting the primary wrench error.

---

# 42. Current best plan: characterize the attainable wrench set first

The next experiment should **not** immediately alter the dynamic allocator weights.

The next step is to characterize the control authority of the actual modeled vehicle.

We should build an offline script that solves for positive and negative extrema of:

\[
F_x,\;
F_y,\;
F_z,\;
\tau_x,\;
\tau_y,\;
\tau_z
\]

under the physical actuator constraints.

At minimum:

\[
0\le f_i\le23\,\text{N}
\]

and:

\[
|\alpha_i|\le3.15\,\text{rad}
\]

for the command-reachable static angle range.

This requires 12 scalar optimizations:

```text
max +Fx
max -Fx

max +Fy
max -Fy

max +Fz
max -Fz

max +Tx
max -Tx

max +Ty
max -Ty

max +Tz
max -Tz
```

The script should report the full maximizing actuator configuration for each case, not only the scalar value.

---

## 43. Do not blindly use one global maximum without inspecting it

Because:

\[
B(\alpha)
\]

is state dependent, a single global axis maximum may be achieved at an extreme rotor geometry that is unrepresentative of normal operation.

Therefore the authority audit should include at least two views.

### 43.1 Global static authority

Search the full allowed static actuator range.

This tells us the theoretical envelope of the platform.

### 43.2 Representative / nominal authority

Evaluate authority around representative operating configurations such as:

- zero-tilt hover,
- the settled forward-allocation configuration,
- potentially several evenly sampled common tilt / operating configurations.

This tells us whether the global extrema are representative enough to serve as normalization constants.

If the authority varies strongly with geometry, a future state-dependent or horizon-dependent normalization may be worth investigating.

However, time-varying normalization must be treated carefully because changing the scales also changes the effective objective while the vehicle moves.

Therefore the first implementation should prefer a **fixed, documented nominal authority scale** unless strong evidence supports dynamic scaling.

---

# 44. Proposed attainable-authority wrench normalization

After the authority study, define:

\[
s_i
\]

for each wrench component.

A candidate fixed scale is:

\[
s_i
=
\max
\left(
|w_{i,\max}|,
|w_{i,\min}|
\right)
\]

if the positive and negative authority are reasonably symmetric.

If the authority is significantly asymmetric, retain and report that asymmetry instead of silently hiding it.

Then define:

\[
S_w
=
\operatorname{diag}
\left(
1/s_1,\dots,1/s_6
\right)
\]

and test the allocator with:

\[
P=I_6.
\]

This creates the clean experimental question:

> If force and moment errors are normalized by actual vehicle authority, does the large moment transient decrease naturally without sacrificing the improved force transient?

---

# 45. Exact next A/B experiment

Once the fixed authority scales are obtained:

1. keep the current 3-step / 30 ms horizon,
2. keep the same 100 Hz allocator rate,
3. keep the same servo model,
4. keep the same weak regularization,
5. replace only the wrench normalization,
6. keep:
   \[
   P=I_6
   \]
7. rerun:
   ```text
   experiments/allocation/multi_step_dynamic_allocator.py
   ```
8. rerun:
   ```text
   experiments/allocation/receding_horizon_allocator_transient.py
   ```
9. compare directly against the current \(mg/mgl\) baseline.

The main comparison metrics should remain:

- peak forward \(F_z\) error,
- forward \(F_x\) rise time,
- peak moment norm during the forward transition,
- return-to-hover force errors,
- peak return moment norm,
- command saturation,
- rotor-load distribution,
- solver evaluations / computation time.

This isolates normalization from every other design change.

---

# 46. Decision rule after the normalization experiment

If authority normalization reduces the large moment transient while preserving the good force response, it becomes the new baseline.

If large moments remain, then the problem is no longer primarily one of units / normalization.

It becomes an explicit **control-priority question**.

Only then should we consider:

- task-dependent \(P\),
- lexicographic / hierarchical wrench priorities,
- hard bounds on selected wrench errors,
- outer-loop-informed priority scheduling.

This is preferable to manually increasing torque weights until one test looks good.

---

# 47. Secondary actuator objective comes after primary normalization

The pairwise rotor usage and stranded servo angles should be treated separately after the primary wrench objective is well defined.

A future secondary objective may include:

\[
J_{\text{secondary}}
=
J_{\text{load}}
+
J_{\text{servo}}
+
J_{\text{margin}}
+
J_{\text{power}}
\]

Possible terms include:

- equal or desirable rotor loading,
- servo centering where appropriate,
- distance from thrust saturation,
- singularity / conditioning margin,
- predicted future wrench authority,
- power consumption,
- actuator motion.

The preferred long-term architecture is hierarchical:

```text
PRIMARY:
achieve the best feasible wrench trajectory

SECONDARY:
among solutions with essentially the same primary performance,
choose the best actuator configuration
```

Implementation options later include:

- null-space optimization when a suitable local null space is available,
- sequential / lexicographic optimization,
- constrain primary cost to remain within a small tolerance, then minimize secondary cost.

The current weak regularization can remain as a numerical continuity aid until this secondary layer is designed deliberately.

---

# 48. Actuator-side normalization plan

The literature also suggests improving actuator normalization.

### Thrust

Current:

\[
\frac{\Delta f}{f_{\max}}
\]

is a reasonable first dimensionless scale because thrust lies in:

\[
[0,f_{\max}].
\]

Later, if motor acceleration / thrust slew limits are identified, the dynamic allocator should use those rate capabilities directly.

### Tilt servo

Current:

\[
\frac{\Delta\alpha_c}{\alpha_{\max}}
\]

normalizes position-command change by total travel.

For dynamic allocation, a better long-term quantity is likely based on servo motion capability, for example:

\[
\frac{\dot\alpha}{\dot\alpha_{\max}}
\]

or an equivalent quantity obtained from the identified first-order model and command limits.

Because the present servo model has no independent hard constant rate limit, this should not be invented arbitrarily.

It should come from:

- manufacturer data,
- identification,
- or a clearly defined reachable-rate envelope from the first-order model.

---

# 49. What should remain unchanged for now

Until the authority study is complete, do **not** simultaneously modify:

- horizon length,
- torque priority,
- force priority,
- regularization magnitude,
- preferred servo angles,
- rotor balancing,
- singularity cost.

The current 3-step allocator is useful precisely because it is now a controlled experimental baseline.

Changing several design choices at once would make it difficult to understand which modification caused an improvement or regression.

---

# 50. Updated file / experiment map

Core static allocation:

```text
src/tiltable_quadrotor_rl/control/allocation.py
```

Dynamic allocation:

```text
src/tiltable_quadrotor_rl/control/dynamic_allocation.py
```

Servo dynamics:

```text
src/tiltable_quadrotor_rl/robot/tilt_servo.py
```

Regression:

```text
tests/test_allocation_dynamic.py
```

Dynamic-allocation experiments now include:

```text
experiments/allocation/one_step_dynamic_allocator.py
experiments/allocation/dynamic_allocator_horizon_sweep.py
experiments/allocation/receding_preview_allocator_transient.py
experiments/allocation/multi_step_dynamic_allocator.py
experiments/allocation/receding_horizon_allocator_transient.py
```

Next proposed experiment:

```text
experiments/allocation/attainable_wrench_authority.py
```

---

# 51. Current architecture status

### Validated

- static allocation matrix and inverse relationships,
- \(B(\alpha)\) against MuJoCo,
- explicit first-order servo model,
- servo-aware one-step prediction,
- 3-step servo propagation,
- receding-horizon execution logic,
- start/end wrench prediction within each interval.

### Demonstrated improvements

- endpoint-only transient failure identified,
- 3-step horizon greatly reduced forward \(F_z\) transient,
- forward wrench reaches target in approximately 20 ms in the fixed-body actuator experiment,
- return-to-hover force transient is dramatically improved.

### Current problems

- peak moment transient increased,
- current \(mg/mgl\) normalization embeds a questionable force-vs-moment tradeoff,
- optimizer can concentrate thrust on a rotor pair,
- unused tilt angles can remain stranded,
- no explicit secondary actuator-configuration objective,
- thrust dynamics are still omitted,
- angle winding and singularity handling remain unresolved.

---

# 52. Best current development plan

The current best sequence is:

1. **Freeze the present 3-step allocator as a reproducible baseline.**
2. **Implement an attainable-wrench-authority audit.**
3. **Measure positive and negative force / moment authority from the actual vehicle model.**
4. **Inspect global versus representative authority before choosing fixed scales.**
5. **Define a documented capability-based wrench normalization \(S_w\).**
6. **Keep control priority neutral initially with \(P=I_6\).**
7. **Repeat the exact same isolated and 2-second transient experiments.**
8. **Decide whether remaining moment error is normalization or true control-priority behavior.**
9. **Only then design an explicit priority strategy if required.**
10. **After the primary wrench objective is stable, add a secondary actuator objective for rotor loading, servo configuration, saturation margin, and singularity avoidance.**
11. **Later incorporate thrust dynamics and realistic actuator-rate limits.**
12. **Eventually compare the classical dynamic allocator with RL-based alternatives.**

This sequence keeps each experiment interpretable and avoids tuning around symptoms.

---

# 53. Literature references used for the current re-evaluation

### Smeur, Höppener, De Wagter — prioritized control allocation

E. J. J. Smeur, D. C. Höppener, and C. De Wagter,  
**“Prioritized Control Allocation for Quadrotors Subject to Saturation,”**  
International Micro Air Vehicle Conference and Flight Competition, 2017.

Key relevance:

- Weighted Least Squares allocation,
- explicit prioritization between control objectives,
- demonstrates that priority is a design choice separate from actuator feasibility.

Reference:
https://repository.tudelft.nl/record/uuid:54c4659a-3a6d-4d44-873d-aec6ae5e2376

---

### Hafner, Myschik, Holzapfel — normalized control-allocation spaces

S. Hafner, S. Myschik, and F. Holzapfel,  
**“Accelerating Sequential Least Squares Active Set Control Allocation,”**  
Control Engineering Practice, Vol. 166, Article 106621, 2026.

DOI:

```text
10.1016/j.conengprac.2025.106621
```

Key relevance:

- actuator-input normalization using actuator limits,
- pseudo-control normalization using maximum attainable pseudo-control,
- normalized attainable-control geometry,
- clear separation of primary pseudo-control error and secondary actuator objective.

---

### Cuniato et al. — omnidirectional tilt-rotor allocation with actuator dynamics

E. Cuniato, M. Allenspach, T. Stastny, H. Oleynikova, R. Siegwart, and M. Pantic,  
**“Allocation for Omnidirectional Aerial Robots: Incorporating Power Dynamics,”**  
IEEE Transactions on Robotics, Vol. 42, pp. 2166–2177, 2026.

DOI:

```text
10.1109/TRO.2026.3688101
```

arXiv:

```text
2412.16107
```

Key relevance:

- allocation specifically for tilt-rotor omnidirectional aerial robots,
- actuator dynamics included in allocation,
- actuator velocities normalized using physical limits,
- improved numerical conditioning,
- reduction of heuristic weighting for actuator balancing.

---

### Sorge et al. — dynamic allocation for tilting UAVs

M. Sorge, F. Ciresola, G. Michieletto, and A. Cenedese,  
**“Dynamic Control Allocation for Dual-Tilt UAV Platforms,”**  
arXiv, 2026.

arXiv:

```text
2604.05677
```

Key relevance:

- high-level desired wrench plus dynamic actuator allocator,
- explicit actuator dynamics,
- saturation,
- redundancy used for actuator-state optimization.

---

### Jing and Ma — closed-loop incremental DCA

H. Jing and K. Ma,  
**“Closed-loop incremental dynamic control allocation with actuator constraints and application to attitude control of aircraft,”**  
Proceedings of the Institution of Mechanical Engineers, Part G: Journal of Aerospace Engineering, Vol. 240, Issue 7, 2026.

DOI:

```text
10.1177/09544100251409477
```

Key relevance:

- actuator bandwidth included,
- magnitude and rate constraints,
- dynamic allocation formulated as constrained optimization.

---

### Pretto et al. — receding-horizon actuation-aware allocation

R. Pretto, M. Hamandi, A. M. Ali, G. Alcan, A. Tzes, and F. Abu-Dakka,  
**“Receding-Horizon Nullspace Optimization for Actuation-Aware Control Allocation in Omnidirectional UAVs,”**  
arXiv, 2026.

arXiv:

```text
2603.06832
```

Key relevance:

- receding-horizon forward simulation,
- explicit actuator dynamics,
- redundancy / nullspace used for actuator optimization,
- preservation of primary wrench while improving actuator behavior.

---

### Li et al. — target tiltable-multirotor NMPC work

J. Li, Y. Chen, J. Kübel, H. Liu, J. Sugihara, and M. Zhao,  
**“Effector-Centric NMPC of Tiltable-Multirotors for Offset-Free Omnidirectional Aerial Manipulation,”**  
2026.

arXiv:

```text
2608.17819
```

Key relevance:

- four-rotor one-DoF tiltable configuration,
- six-DoF wrench generation,
- null-space redundancy,
- singularity traversal,
- physical constraints,
- onboard 100 Hz predictive control.

---

# 54. Updated conclusion

The dynamic allocator has progressed beyond the original endpoint-preview prototype.

The project now has a working:

\[
\boxed{
3\text{-step},\ 30\text{ ms},\ 100\text{ Hz receding-horizon dynamic allocator}
}
\]

that explicitly predicts servo motion and scores the wrench through the horizon.

This architecture substantially improves force transients.

However, the latest experiment shows that a good horizon formulation alone is not enough.

The present:

\[
mg/mgl
\]

wrench scaling creates a force-versus-moment tradeoff that is not sufficiently grounded in the actual control authority of the vehicle.

The literature review indicates that the cleaner architecture is:

\[
\boxed{
\text{capability normalization}
+
\text{explicit control priority}
+
\text{separate secondary actuator objective}
}
\]

rather than attempting to make one weight matrix perform all three jobs.

Therefore the immediate next milestone is **not more manual weight tuning**.

It is:

\[
\boxed{
\text{measure the attainable wrench authority of the modeled vehicle}
}
\]

and use that information to build the next normalization baseline.

Only after that A/B test should the project decide whether an explicit wrench-priority hierarchy is needed.

---

## Final status

**Static allocation:** validated.  
**Servo model:** validated.  
**One-step dynamic allocation:** validated as a useful prototype.  
**Prediction-horizon behavior:** characterized.  
**30 ms endpoint-preview allocator:** tested; endpoint-only weakness identified.  
**3-step / 30 ms receding-horizon allocator:** implemented and transient-tested.  
**Force-transient behavior:** substantially improved.  
**Moment-transient behavior:** currently worse than endpoint-preview baseline.  
**Current normalization:** under re-evaluation; \(mg/mgl\) retained only as the historical baseline.  
**Literature direction:** capability normalization + explicit priority + secondary actuator optimization.  
**Immediate next experiment:** attainable-wrench-authority characterization.
