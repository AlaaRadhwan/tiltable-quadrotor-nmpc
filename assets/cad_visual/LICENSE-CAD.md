# Licence for the original CAD meshes

The eight STL meshes in `assets/cad_visual/meshes/` are original designs by Alaa Radhwan, modeled in Onshape, and are licensed under the **Creative Commons Attribution 4.0 International** licence (CC BY 4.0):

<https://creativecommons.org/licenses/by/4.0/> (legal code: <https://creativecommons.org/licenses/by/4.0/legalcode>)

Copyright 2026 Alaa Radhwan.

| File | Part |
|---|---|
| `base.stl` | Base plate |
| `arm_plusX.stl`, `arm_plusY.stl`, `arm_negativeX.stl`, `arm_negativeY.stl` | Arms |
| `bldc_mount.stl` | BLDC motor mount |
| `servo_mount.stl` | Servo mount |
| `end_effector.stl` | End-effector |

You may share and adapt them, including commercially, if you give appropriate credit, link to the licence and indicate any changes. Suggested credit: "Tiltable quadrotor CAD by Alaa Radhwan, CC BY 4.0, https://creativecommons.org/licenses/by/4.0/" (add the repository URL).

**Scope.** This licence applies only to those eight files. It does not apply to:
- the project's source code, which is Apache-2.0 (see the top-level `LICENSE`);
- the third-party propeller, motor and servo models used in the author's private detailed visualization. They are not part of this repository and are not covered by any licence granted here (see `THIRD_PARTY.md`);
- the generic stand-in geometry, which is defined in code (`tools/cad_visual/build_visual_mjcf.py`) and therefore Apache-2.0;
- generated data files (`tiltable_quadrotor_visual.xml`, `cad_placements.json`, `propeller_spin_axes.json`, `meshes/parts_manifest.json`), which are Apache-2.0.
