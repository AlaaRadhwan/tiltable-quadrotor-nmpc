# Tiltable Quadrotor: End-Effector NMPC for Omnidirectional Flight

A quadrotor whose four rotors each tilt on a servo can push in any direction without changing its attitude. This project models that vehicle in MuJoCo and controls it with a nonlinear MPC that tracks a **tool point** instead of the body: the drone can roll, pitch or spin through a full 360° while the tip of its end-effector stays almost in place.

The controller runs closed-loop against a rigid-body plant with first-order servo dynamics, and the results are rendered with a CAD visualization of the airframe and spinning propellers.

> Simulation only; nothing has been flown. Reinforcement learning is future work: **no RL policy has been trained**, and the NMPC is not an offset-free controller (there is no disturbance estimation).

## Demonstrations

**360° roll, 360° pitch and cartwheel about the end-effector**

<!-- VIDEO-1: GitHub attachment URL to be added after upload -->

Complete 360° roll and 360° pitch maneuvers followed by an excerpt of a cartwheel (about a third of its full rotation), all holding the tool point in place. These are three separate simulation runs joined by straight cuts, so the time stamp restarts at each cut. Real-time playback; the orange marker is the controller's tool point and the green one is the target.

**Simultaneous 6-DoF tracking**

<!-- VIDEO-2: GitHub attachment URL to be added after upload -->

One uncut 35 s run tracking simultaneous translation (±0.35 / ±0.30 / ±0.20 m) and rotation (roll ±40°, pitch ±30°, yaw ±60°) of the tool point.

<sub>Rendered with the repository's simplified geometry, so the propellers, motors and servos are generic stand-ins (see [Attribution](#documentation-attribution-and-licenses)).</sub>

## What I built

- **Plant and servo model**: a MuJoCo rigid-body vehicle with four tilt servos modeled as first-order lags.
- **Actuator allocation**: a 6×8 map from four thrusts and four tilt angles to force and torque, with static and servo-aware receding-horizon allocators.
- **End-effector NMPC**: 17 states and 8 actuator inputs, solved with acados at a 10 ms control period over 2 ms physics, with the tool point 0.15 m above the body origin.
- **Maneuver library**: 360° roll, pitch and yaw, a vertical cartwheel, and a simultaneous 6-DoF benchmark.
- **CAD visualization**: my own Onshape design exported to meshes, with propellers animated from the commanded thrust. It is display-only and never feeds back into the simulation.

## Architecture

```text
 reference (tool position + orientation)
        │
        ▼
 end-effector NMPC  (acados + CasADi, 17 states / 8 inputs, 10 ms)
        │  4 rotor thrusts + 4 tilt-angle commands
        ▼
 servo model (first order)  ──►  rigid-body plant (MuJoCo, 2 ms)
                                        │  state
                                        ├──► controller
                                        └──► renderer (visual only)
```

Source is in `src/tiltable_quadrotor_rl/` (`robot/`, `control/`, `visualization/`) and the runners are in `experiments/nmpc/`.

## Selected results

Simulation, nominal parameters and the specific maneuvers below; these are not general performance guarantees.

| Maneuver | Tool-position error (RMS) | Notes |
|---|---|---|
| Simultaneous 6-DoF tracking, 35 s | **19.8 mm** (37.2 mm max) | Attitude 2.41° RMS, 0 solver failures |
| 360° pitch / roll about a fixed tool point | **about 8 mm** | Attitude about 2.8° RMS |
| Cartwheel | **about 12 mm** | Attitude about 1.1° RMS |

The 6-DoF and the pitch/roll results use an empirical, angular-rate-dependent position correction; without it the 6-DoF error was 27.3 mm. I re-ran all four demonstrations with the public geometry and the position and attitude figures matched the earlier reports (the 6-DoF and cartwheel numbers exactly; the pitch/roll runs gave 7.7–7.8 mm).

## Engineering investigations

Write-ups are in [`Findings/`](Findings):

- **Servo dynamics and allocation**: how tilt-servo lag limits wrench tracking, and the servo-aware allocator it led to. [NMPC servo findings](Findings/TILTABLE_QUADROTOR_NMPC_SERVO_FINDINGS_2026-09-14.md), [dynamic allocator](Findings/TILTABLE_QUADROTOR_DYNAMIC_ALLOCATOR_FINDINGS_UPDATED.md)
- **End-effector-centric control**: moving the control point to the tool tip and the rate-bias correction for 360° pitch and roll. [Update, Sep 22](Findings/TILTABLE_QUADROTOR_END_EFFECTOR_NMPC_UPDATE_2026-09-22.md)
- **Cartwheel and low-thrust transitions**: error concentrates where rotors pass through near-zero thrust; a predictive pre-positioning experiment helped but was not adopted. [Cartwheel findings](Findings/TILTABLE_QUADROTOR_END_EFFECTOR_NMPC_CARTWHEEL_AND_PREDICTIVE_FINDINGS_2026-09-26.md)
- **6-DoF benchmark and timing**: [6-DoF findings](Findings/TILTABLE_QUADROTOR_GENERAL_6DOF_END_EFFECTOR_NMPC_FINDINGS_2026-10-13.md)
- **Robustness**: [parameter and disturbance sweep](Findings/TILTABLE_QUADROTOR_NMPC_ROBUSTNESS_CHARACTERIZATION_FINDINGS_2026-10-14.md)

## Limitations and status

| Component | State |
|---|---|
| End-effector NMPC, allocation, maneuvers, visualization | Working in simulation |
| Disturbance-estimating (offset-free) NMPC | Not implemented |
| Reinforcement-learning controller | Not started as a trained policy |
| Hardware | Not attempted |

- **Timing:** the full Python control loop averaged 8.7 ms in the benchmark, with 29 of 3501 steps over the 10 ms budget. It is not demonstrated as hard real-time; the acados solve alone averages 0.8 ms.
- **Robustness:** in one-at-a-time tests, ±10 % mass or thrust-effectiveness error left steady offsets of roughly 100 mm, and a constant yaw torque left a 12.8° attitude error, because there is no disturbance estimation.
- **Coverage:** the 6-DoF run does not exercise the low-thrust regime that the cartwheel does.
- **Tests:** the unit tests run as scripts (`tools/test`). `tests/test_paper_pitch_trajectory.py` currently fails an assertion at import; it has not been fixed. Not every test has been run.

## Documentation, attribution and licenses

Running the experiments needs Python 3.13, MuJoCo, NumPy, SciPy, CasADi and [acados](https://docs.acados.org) built from source with its Python interface (`ACADOS_SOURCE_DIR` and `LD_LIBRARY_PATH` set). `tools/python` runs a script with `src/` on the path using the interpreter in `TILTABLE_PYTHON`. Generate the NMPC solver once with `experiments/nmpc/build_effector_nmpc.py`; commands for the maneuvers and rendering are in [`docs/CAD_VISUALIZATION.md`](docs/CAD_VISUALIZATION.md). This is an engineering portfolio, so a fresh-clone install has not been tested.

**Research.** This is an independent implementation. The effector-centric formulation, the paper-style maneuvers and the RL motivation follow published work from the DRAGON Lab at the University of Tokyo; the results here come from my own simulation, not reproductions of their experiments, and I am not affiliated with the authors.

- J. Li, Y. Chen, J. Kübel, H. Liu, J. Sugihara, M. Zhao, "Effector-Centric NMPC of Tiltable-Multirotors for Offset-Free Omnidirectional Aerial Manipulation," arXiv:2608.17819 [cs.RO], 2026, doi:10.1109/TRO.2026.3731683.
- W. Zhang, Z. Ma, J. Li, H. Wang, H. Liu, J. Sugihara, C. Chen, Y. Chen, E. Cuniato, M. Pantic, M. Zhao, "Learning Agile and Robust Omnidirectional Aerial Motion on Overactuated Tiltable-Quadrotors," arXiv:2602.21583 [cs.RO], 2026. (This is the title of the first arXiv version; the current v2 listing is titled "Learning End-to-End Control for Omnidirectional Aerial Motion on Overactuated Tilt-rotor Quadrotors.")

**CAD.** The base, arms, mounts and end-effector are my own design, modeled in Onshape. My private detailed model also used three downloaded components (an APC 8×8E propeller and a T-Motor AT2814 motor, both from GrabCAD, and a ROBOTIS servo). Their redistribution terms are unconfirmed, so they are **not** in this repository; the visualization here uses original generic stand-ins for them. Sources and details are in [`THIRD_PARTY.md`](THIRD_PARTY.md).

**Licenses.**
- Original code: [Apache-2.0](LICENSE), Copyright 2026 Alaa Radhwan.
- My original CAD meshes (8 STL files): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), see [`assets/cad_visual/LICENSE-CAD.md`](assets/cad_visual/LICENSE-CAD.md).
- Third-party models: not included and not covered by either licence.
