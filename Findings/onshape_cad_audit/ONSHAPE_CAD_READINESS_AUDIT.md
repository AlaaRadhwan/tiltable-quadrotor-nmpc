# Onshape CAD Readiness Audit — Tiltable Quadrotor

> Public-snapshot note: the raw `assembly_observations.json` referenced below (Onshape API data) is not included in this repository.

Read-only audit via authenticated Onshape REST GET requests (API v9). Nothing in Onshape, the repo, or any environment was modified. No credentials appear in this report or in `assembly_observations.json`.

- Document: `Tiltable_Quadrotor` — `…/documents/5b54224a…/w/b4196110…/e/404d1446…` (element "Titlable_Quadrotor_Assembly")
- Data: `getAssemblyDefinition` (mate features, mate connectors, non-solids), `boundingboxes`, `massproperties`, per-part bounding boxes, element list. Raw JSON is kept only in the session scratchpad.
- Units: the API returns meters. Transforms are 4x4 row-major, translation in entries 3/7/11.
- Labels: **[M]** measured from API data, **[I]** inferred, **[U]** unverified (API did not provide it).

---

## 1. Executive summary

**The CAD is geometrically excellent for a visual-only integration, but it is not ready for a naive `onshape-to-robot` export.** Recommended path: **Option B (direct per-part mesh export, attached to the existing MJCF bodies).**

- **[M]** Rotor centers match the MJCF exactly: (±0.275, 0, ·) and (0, ±0.275, ·) m, to <5 µm.
- **[M]** The CAD frame is already +X forward / +Y left / +Z up, at 1:1 scale in meters. **No rotation and no scaling** is needed. Only a z translation is needed (≈ −0.0246 m, see §4).
- **[M]** All four tilt servos have a Revolute mate whose axis points radially outward along the arm. This matches MJCF `tilt_i` (hinge about local +X, which points outward after the body's Z rotation).
- **[M]** All four propellers are independent instances (`Body1 <1..4>`) with their own Revolute spin mate about the vertical axis. They are ready for visual spin.
- **[M]** There are no suppressed instances. 36 hidden occurrences are all screws and LED caps inside the servo sub-assembly.
- **Main problems:**
  - Four of the six motor parts per rotor have no mate at all, so they would not follow the tilt in a CAD kinematic export.
  - The tilt servos are nested in a 15-part sub-assembly that contains its own internal Revolute.
  - No mate follows the `dof_`/`fix_`/`frame_`/`link_` convention.
  - The CAD contains an `EndEffector` part that the MJCF doesn't have.
  - The propeller diameter is 0.2016 m, not the 0.228 m of the placeholder.
  - Tilt axis height (z = +0.0246) differs from the rotor/prop plane (z ≈ −0.0138), unlike the MJCF where both are at z=0.

## 2. Component hierarchy and part identification

Top-level assembly: 46 instances, 154 occurrences, 74 mates (8 Revolute, 66 Fastened), 3 sub-assembly definitions, 0 patterns. **[M]** It is a valid top-level assembly for extraction.

| Role | Instances (CAD) | Group | Notes |
|---|---|---|---|
| Chassis plate | `Base <1>` | **Rigid body** | 0.14×0.14×0.04 m box, spans z 0…0.04 in assembly **[M]** |
| Arms | `Arm_plusX/plusY/negativeX/negativeY <1>` | **Rigid body** | 30 mm square, run 0→0.2248 m; arm axis at z=0.02 **[M]** |
| End effector | `EndEffector <1>` | Rigid body (decision needed) | 0.06×0.06×0.13 m, z 0.02…0.17; fastened to Base. Not present in the MJCF **[M]** |
| Servo mount | `Servo_Mount <2/4/6/8>` | Rigid body (fixed to arm) | one per arm |
| Servo case/parts | `DC15_A01_DUMMY_ASSY_IDLE_ASM <1..4>` → `CASE_F/M/B`, caps, `B3B-EH` ×2, screws | Rigid body (case side) | 15 parts each; 36 hidden screws/caps in total |
| Servo horn | `…/DC15_A01_HORN_DUMMY` (+ `HORN_IDLE2`, screw) | **Tilting motor assembly** | Moving side of tilt Revolute |
| Motor mount | `BLDC_Mount <3/5/9/7>` | **Tilting motor assembly** | fastened to horn |
| Motor stator/housing | `PartBody`, `Corpo.2`, `Corpo.3`, `Corpo.6.2` (`<1>`, `<5>`, `<4>`, `<6>`) | Tilting motor assembly | **unmated** (see §3) |
| Motor plate | `Corpo.6.3` | Tilting motor assembly | fastened to BLDC mount |
| Motor shaft | `Corpo.6.1` | Spinning | Revolute partner of the propeller |
| Propeller | `Body1 <1..4>` (APC 8x8E) | **Spinning** | independent instance per rotor |

Rotor ↔ CAD mapping **[M]**:

| MJCF body | Arm | Servo | BLDC mount | Motor set | Propeller |
|---|---|---|---|---|---|
| rotor_1 (+X) | Arm_plusX | DC15 `<1>` | `<3>` | `<1>` | Body1 `<1>` |
| rotor_2 (+Y) | Arm_plusY | DC15 `<2>` | `<5>` | `<5>` | Body1 `<2>` |
| rotor_3 (−X) | Arm_negativeX | DC15 `<3>` | `<9>` | `<4>` | Body1 `<3>` |
| rotor_4 (−Y) | Arm_negativeY | DC15 `<4>` | `<7>` | `<6>` | Body1 `<4>` |

Also present in the document but not instanced in this assembly: `helice_8_3_aspas`, `Part1`, `MG995`, `XL,XC-330 parts` studios and several BLOB imports. They are clutter, not a blocker.

## 3. Mate and joint analysis

All 8 Revolutes are active and unsuppressed, with **empty `mateLimits`** **[M]**. All names are Onshape defaults (`Revolute 1`, `Revolute 1 (1)` …). No mate connectors with custom names exist.

| Mate | Role | Axis origin (m) | Axis dir |
|---|---|---|---|
| Revolute 1 | tilt rotor_1 (horn vs case, DC15 `<1>`) | (0.2497, 0, 0.0246) | +X |
| Revolute 1 (1) | tilt rotor_2 | (0, 0.2497, 0.0246) | +Y |
| Revolute 1 (2) | tilt rotor_3 | (−0.2497, 0, 0.0246) | −X |
| Revolute 1 (3) | tilt rotor_4 | (0, −0.2497, 0.0246) | −Y |
| Revolute 2 | spin prop 1 vs shaft | (0.275, 0, −0.0138) | −Z |
| Revolute 3 | spin prop 2 | (0, 0.275, −0.0138) | −Z |
| Revolute 4 | spin prop 3 | (−0.275, 0, −0.0138) | −Z |
| Revolute 5 | spin prop 4 | (0, −0.275, −0.0138) | −Z |

At CAD zero pose, both mate-connector sides of every Revolute coincide in position and axis **[M]**, so there are no misaligned axes.

- **Tilt sign:** all four tilt axes point outward. This equals the MJCF's local +X after the body Z-rotation (0, 90°, 180°, −90°), so positive rotation is the same right-hand sense **[M/I]**. CAD zero pose matches MJCF `alpha=0`: prop disc horizontal, spin axis vertical **[I]**.
- **Spin sign:** all four spin mates have the same direction (−Z). The MJCF alternates yaw reaction torque (b = +1, −1, +1, −1). Visual animation must negate the angle on two rotors **[I]**.
- The propeller meshes are all identical `Body1` (same orientation matrices). There is no CW/CCW mirrored pair, which is cosmetic only **[M]**.
- **Unmated parts:** `PartBody`, `Corpo.2`, `Corpo.3`, `Corpo.6.2` have no mate in any of the four motors (16 parts). Their poses are correct at the zero pose and are not `fixed`, but nothing constrains them to the tilting group. They'd stay put or be orphaned onto the base in a kinematic export **[M]**.

### Blocking vs unnecessary (for this visual-only workflow)

| Item | Blocks `onshape-to-robot`? | Blocks Option B? |
|---|---|---|
| No `dof_`/`fix_`/`frame_`/`link_` mate names (the tool ignores unprefixed mates) | **Yes** | No |
| Tilt joint nested in a servo sub-assembly (document says nothing about nested mates) | Probably **[U]** | No |
| Unmated motor parts | Yes (they'd be orphaned to base) | Only needs correct grouping, which I can do from names |
| No joint limits | No | No |
| Dummy servo internal Revolute (horn vs case) is the actual tilt | n/a | No |
| Axis misalignment | none found | none |

## 4. Rotor coordinates and comparison with MuJoCo

| Quantity | MJCF | CAD **[M]** | Δ |
|---|---|---|---|
| Rotor 1 center (x,y) | (0.275, 0) | (0.275000, −0.0) | <5 µm |
| Rotor 2 | (0, 0.275) | (0.000004, 0.275) | <5 µm |
| Rotor 3 | (−0.275, 0) | (−0.275, 0) | ~0 |
| Rotor 4 | (0, −0.275) | (0, −0.275) | ~0 |
| Tilt axis | local X through rotor origin | outward, through (r=0.2497, z=0.0246) | z-offset only |
| Prop disc height | z = 0 (rotor frame) | z ≈ −0.0138 (spin mate); blades roughly −0.0214…+0.0002 | −0.038 m below tilt axis |
| Prop diameter | 0.228 (cylinder) | 0.2016 (blade-length bbox) | −12% |
| Thrust direction | +Z (up) | spin axis −Z; thrust sense **[I]** up | consistent |

**Assembly bounding box [M]:** x ±0.3758, y ±0.2946, z −0.0939…+0.1700 m, i.e. 0.752 × 0.589 × 0.264 m. The z top is the end effector; the asymmetry in x vs y comes from all propeller blades pointing along X.

**Transform needed [M/I]:** rotation = identity, scale = 1. Translation = (0, 0, −0.0246) puts the tilt axes on MJCF z=0, so the visuals tilt about the correct line. Alternatives: the CAD center of mass is z=0.0261, but it's built on incomplete data (**64 parts have no mass**), so don't use it. The base plate center is z=0.02. The ~1.5–4.6 mm differences are irrelevant visually. Unverified: how the MJCF author intended the base visual's z origin; the MJCF base geom is a ±0.025 box.

Consequence of the CAD design vs MJCF: in the MJCF the tilt axis passes through the rotor center, whereas in the CAD the pivot is 38 mm above the prop plane. Sticking the mesh to the rotor body works for the tilt axis, but the mesh will swing in a small arc (r = 38 mm) around the thrust site. This is a physical CAD feature, not a bug. The MJCF physics are unaffected because the visuals are non-colliding.

## 5. Issues

| # | Severity | Issue | Required correction |
|---|---|---|---|
| 1 | **Medium** (blocks Option A) | No `dof_`/`link_`/`fix_`/`frame_` mate naming | Not needed for Option B |
| 2 | **Medium** | 4 unmated motor parts per rotor (16 total) | Optionally fasten them to `Corpo.6.3` (motor plate) so the CAD kinematics are right |
| 3 | **Medium** | `EndEffector` part is in the CAD, not in the MJCF | Decide: include as a separate chassis mesh, or exclude. It is fastened to Base |
| 4 | Low | Servo is a 15-part nested sub-assembly with 36 hidden screws/caps/etc. | Export only visible occurrences; hidden ones must be excluded |
| 5 | Low | Prop diameter 0.2016 vs 0.228 m placeholder | Informational. Is the MJCF disc oversize, or is the CAD prop the wrong size? |
| 6 | Low | Tilt pivot 38 mm above prop plane | Informational; see §4 |
| 7 | Low | All 4 props are the same chirality | Cosmetic only |
| 8 | Low | No tilt limits in mates | Not needed |
| 9 | Info | Mass props incomplete (64 parts w/o mass); CAD mass 1.315 kg vs MJCF 2.959 kg | Do not use CAD mass; MJCF physics stay unchanged |
| 10 | Info | Typo in element name "Titlable" | Cosmetic |

## 6. Visual export suitability and recommended workflow

**Recommendation: Option B** (export visual meshes directly, attach to existing MJCF bodies).

Why:
- `onshape-to-robot` needs prefixed mates (none exist), and would produce its own body tree, 8+ joints (including the dummy servo's internal Revolute), and its own inertials. It would then have to be stripped and re-merged with the validated XML. Option A needs substantial editing in Onshape and then in the generated MJCF.
- The CAD already matches the MJCF frame and rotor centers, so Option B only needs one fixed offset plus per-group meshes.
- The MJCF tilt joints can be reused as is, without adding physical joints.

Suggested grouping for Option B (all positions in CAD frame, then apply T = (0,0,−0.0246)):

1. `base_visual`: Base, 4 arms, 4 servo mounts, 4 servo cases (CASE_F/M/B, caps, connectors; exclude hidden), plus the EndEffector if wanted. Attach to `base`.
2. `rotor_i_visual` (×4): horn, BLDC mount, `PartBody`, `Corpo.2`, `Corpo.3`, `Corpo.6.2`, `Corpo.6.3`. Attach to `rotor_i`. Mesh vertices must be expressed in the rotor-local frame: subtract the rotor center, then rotate by −(0, 90°, 180°, −90°) about Z.
3. `prop_i_visual` (×4): `Corpo.6.1` shaft plus `Body1 <i>`. For spin, attach to a child body of `rotor_i` with a visual-only hinge; the user wants that later, and this report doesn't add it. Spin axis: local −Z through the rotor center (z ≈ −0.0138 CAD).

Mesh size: **[U]** the API did not give triangle counts. The T-Motor STEP parts and the servo are likely heavy. Plan to simplify (decimate to ~10–50k triangles total) and to export only visible parts. Because there are only two unique sub-meshes per rotor group (all four are identical up to a rigid transform), one motor mesh and one prop mesh can be reused four times with `<mesh>` assets and per-body poses.

Also **[U]**: the actual export format available (STL/GLTF per part via `…/partstudios/…/stl` or `…/parts/…/stl`) was not called, as this audit does no export.

## 7. Confirmed vs uncertain

**Confirmed [M]:** rotor centers; tilt and spin axis positions and directions; frame orientation and units; component grouping; hidden/suppressed status; absence of naming conventions; unmated motor parts; assembly and part bounding boxes; incomplete mass properties.

**Inferred [I]:** positive tilt direction equals MJCF positive alpha; zero pose equals alpha=0; thrust sense up; the z offset recommendation; the intended motor grouping.

**Unverified [U]:** triangle counts and file sizes; whether `onshape-to-robot` handles the nested mates (the docs say nothing); which side is physically "moving" in each Revolute was inferred from fastened relationships, not from an API flag; material/colors/appearance (not retrieved); the MJCF author's intended base z origin.

## 8. What to change in Onshape before exporting

**Required for Option B (recommended): nothing.** You can export as-is, with the grouping above.

**Recommended but optional:**
1. Fasten `PartBody`, `Corpo.2`, `Corpo.3`, `Corpo.6.2` to `Corpo.6.3` (or to the BLDC mount) in all four motors, so the assembly actually moves correctly when the tilt Revolute is driven.
2. Decide on the `EndEffector` (keep as a separate part, or suppress it/hide it for the export).
3. Confirm the propeller diameter (0.2016 m) is what you intend.

**Only if you choose Option A (`onshape-to-robot`):** rename the 4 tilt Revolutes to `dof_tilt_1` … `dof_tilt_4`; add `link_`/`fix_` mates to merge the base, arms, servo cases, and mounts; consider `dof_`-prefixing the 4 spin Revolutes (`dof_spin_i`); consider adding `frame_rotor_i` mate connectors at the rotor centers for MuJoCo sites; and the sub-assembly nesting may still cause problems **[U]**. I advise against this path.
