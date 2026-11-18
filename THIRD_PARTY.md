# Third-party material and licensing

## Summary

- **Original code** in this repository is licensed under Apache-2.0 (`LICENSE`, Copyright 2026 Alaa Radhwan).
- The author's original CAD (base, four arms, BLDC mount, servo mount, end-effector: the 8 STLs in `assets/cad_visual/meshes/`) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see [`assets/cad_visual/LICENSE-CAD.md`](assets/cad_visual/LICENSE-CAD.md). Placement and spin-axis data files are Apache-2.0.
- **Three component models in the author's private, detailed visualization came from third parties.** Their redistribution terms are unconfirmed, so **their geometry is not included in this repository**. The public visualization uses original generic stand-ins (MuJoCo cylinders, boxes and an ellipsoid blade, written in `tools/cad_visual/build_visual_mjcf.py`).

Neither Apache-2.0 nor CC BY 4.0 applies to the third-party models below, and downloading a model does not by itself grant any right to redistribute it.

## Components not distributed here

| Component | Source | Creator / owner | Licence status |
|---|---|---|---|
| Propeller, APC 8×8E | [GrabCAD: "APC 8 x 8E propeller, recreation from optical scan"](https://grabcad.com/library/apc-8-x-8e-8-propeller-recreation-from-optical-scan-1) | A GrabCAD member's recreation of an APC product (uploader's name not verified; the page could not be read automatically). | **Unconfirmed.** Governed by the uploader's terms on GrabCAD. |
| BLDC motor, T-Motor AT2814 | [GrabCAD: "T-Motor AT2814"](https://grabcad.com/library/t-motor-at2814-1) | A GrabCAD member's model of a T-Motor product (uploader's name not verified). | **Unconfirmed**, as above. |
| Servo, ROBOTIS (DYNAMIXEL XL330/XC330 family, inferred) | [ROBOTIS download page](https://en.robotis.com/service/downloadpage.php?ca_id=70) | ROBOTIS Co., Ltd. | **Unconfirmed for the CAD files.** The site footer states Creative Commons Attribution 4.0 for the site's work; whether that covers the downloadable STEP files is not established, so it is not relied on. |

APC, T-Motor, ROBOTIS and DYNAMIXEL are trademarks of their owners. This project is not affiliated with or endorsed by them.

## What the public visualization contains

| Part | Public repository |
|---|---|
| Base, arms, servo mount, BLDC mount, end-effector | Author's own meshes, CC BY 4.0 |
| Propeller | Original generic two-blade ellipsoid |
| BLDC motor | Original generic cylinders (stator, bell, shaft) |
| Servo body, horn, idle cap, connector | Original generic cylinders and boxes |

The stand-ins use round-number dimensions chosen for this project and the author's own assembly layout. They were not copied from, traced over or fitted to the third-party meshes. Fourteen meshes were removed from the public snapshot: `prop`, `motor_partbody`, `motor_corpo2`, `motor_corpo3`, `motor_corpo62`, `motor_plate`, `motor_shaft`, `servo_case_b`, `servo_case_f`, `servo_case_m`, `servo_horn`, `servo_horn_idle`, `servo_idle_cap`, `servo_connector`.

`assets/cad_visual/cad_placements.json` still lists those parts by occurrence name and pose as assembly data (no geometry). `meshes/parts_manifest.json` covers only the author's eight meshes.

## Optional: your own detailed visualization

The default visualization needs none of this. If you want the detailed look, obtain the models yourself from the sources above, under their providers' terms. This repository does not download them, bypass any login, or redistribute them.

1. Convert each part to STL (metres) and place the files, named as below, in `assets/cad_visual/meshes/third_party/`. Each part's origin and axes must match the part-local frame that `cad_placements.json` expects. Raw downloads usually do not, so you may have to re-export or re-align them; this alignment is not provided.
   - Propeller: `prop.stl`
   - Motor: `motor_partbody.stl`, `motor_corpo2.stl`, `motor_corpo3.stl`, `motor_corpo62.stl`, `motor_plate.stl`, `motor_shaft.stl`
   - Servo: `servo_case_b.stl`, `servo_case_f.stl`, `servo_case_m.stl`, `servo_horn.stl`, `servo_horn_idle.stl`, `servo_idle_cap.stl`, `servo_connector.stl`
2. Check what is present: `tools/python tools/cad_visual/install_detailed_cad.py`
3. Generate the detailed model: `tools/python tools/cad_visual/install_detailed_cad.py --build`. This writes `assets/cad_visual/tiltable_quadrotor_visual_detailed.xml`, which is git-ignored along with the `third_party/` folder.

The scripts use the public model unless you point them at the detailed XML.

## Research references

The effector-centric NMPC formulation, the paper-style maneuvers and the RL motivation follow the work cited in `README.md`. The papers are not included. This repository is an independent implementation, not affiliated with the authors.

## Software

MuJoCo, MjLab, acados and CasADi are used under their own licences and are not redistributed.

## Open items

- Confirm the licence terms of the three third-party models (and the GrabCAD uploaders' names). Until then they stay out of the repository.
