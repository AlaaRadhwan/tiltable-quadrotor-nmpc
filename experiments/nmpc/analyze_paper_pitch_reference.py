from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    virtual_to_rotor_commands,
    wrench_to_virtual,
)
from tiltable_quadrotor_rl.control.end_effector import (
    DEFAULT_END_EFFECTOR,
)
from tiltable_quadrotor_rl.control.nmpc_effector_paper_pitch_trajectory import (
    PaperStyleFixedEffectorPitchTrajectory,
)
from tiltable_quadrotor_rl.control.nmpc_effector_reference import (
    end_effector_reference_to_body_reference,
    make_effector_stage_reference,
)
from tiltable_quadrotor_rl.control.nmpc_reference import (
    P_ALPHA,
    P_THRUST,
    SINGULARITY_THRUST_THRESHOLD,
    feedforward_wrench_body,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "nmpc"
)

TOOL_POSITION_W = np.array(
    [
        0.0,
        0.0,
        1.5,
    ],
    dtype=float,
)

SAMPLE_DT = 0.01


def nearest_physical_angle_difference(
    current: np.ndarray,
    previous: np.ndarray,
) -> np.ndarray:
    """
    Principal signed difference in [-pi, pi].

    This is used only for diagnostics so we can distinguish:
      - representation jumps,
      - genuinely large physical reference changes.
    """

    return (
        current
        - previous
        + np.pi
    ) % (
        2.0
        * np.pi
    ) - np.pi


def main() -> None:

    trajectory = (
        PaperStyleFixedEffectorPitchTrajectory(
            position_W=TOOL_POSITION_W
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    csv_path = (
        OUTPUT_DIR
        / f"paper_pitch_reference_diagnostic_{timestamp}.csv"
    )

    initial_reference = trajectory.sample(
        0.0
    )

    initial_body_reference = (
        end_effector_reference_to_body_reference(
            initial_reference,
            transform=DEFAULT_END_EFFECTOR,
        )
    )

    previous_effector_quaternion = (
        initial_reference.quaternion_WE.copy()
    )

    previous_body_quaternion = (
        initial_body_reference.quaternion_WB.copy()
    )

    previous_alpha_ref = np.zeros(
        4,
        dtype=float,
    )

    previous_logged_alpha_ref = (
        previous_alpha_ref.copy()
    )

    max_raw_alpha_abs = np.zeros(
        4,
        dtype=float,
    )

    max_refined_alpha_abs = np.zeros(
        4,
        dtype=float,
    )

    max_refined_step = np.zeros(
        4,
        dtype=float,
    )

    max_refined_physical_step = np.zeros(
        4,
        dtype=float,
    )

    min_raw_thrust = np.full(
        4,
        np.inf,
        dtype=float,
    )

    low_thrust_samples = np.zeros(
        4,
        dtype=int,
    )

    near_limit_samples = np.zeros(
        4,
        dtype=int,
    )

    first_near_limit_time = [
        None,
        None,
        None,
        None,
    ]

    fieldnames = [
        "time_s",
        "pitch_ref_deg",
        "pitch_rate_ref_degps",

        "wrench_Fx_B_N",
        "wrench_Fy_B_N",
        "wrench_Fz_B_N",
        "wrench_Tx_B_Nm",
        "wrench_Ty_B_Nm",
        "wrench_Tz_B_Nm",
    ]

    for rotor in range(4):
        index = rotor + 1

        fieldnames.extend(
            [
                f"raw_thrust_{index}_N",
                f"raw_alpha_{index}_deg",
                f"refined_thrust_{index}_N",
                f"refined_alpha_{index}_deg",
                f"refined_alpha_step_{index}_deg",
                f"refined_alpha_physical_step_{index}_deg",
                f"raw_low_thrust_{index}",
            ]
        )

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        times = np.arange(
            0.0,
            trajectory.end_time_s
            + 0.5 * SAMPLE_DT,
            SAMPLE_DT,
        )

        for sample_index, time_s in enumerate(
            times
        ):

            effector_reference = (
                trajectory.sample(
                    float(
                        time_s
                    )
                )
            )

            pitch_rad, pitch_rate_radps, _ = (
                trajectory.pitch_state(
                    float(
                        time_s
                    )
                )
            )

            body_reference = (
                end_effector_reference_to_body_reference(
                    effector_reference,
                    transform=DEFAULT_END_EFFECTOR,
                )
            )

            wrench_reference_B = (
                feedforward_wrench_body(
                    body_reference
                )
            )

            raw_virtual = np.asarray(
                wrench_to_virtual(
                    wrench_reference_B
                ),
                dtype=float,
            ).reshape(8)

            (
                raw_thrust,
                raw_alpha,
            ) = virtual_to_rotor_commands(
                raw_virtual
            )

            raw_thrust = np.asarray(
                raw_thrust,
                dtype=float,
            ).reshape(4)

            raw_alpha = np.asarray(
                raw_alpha,
                dtype=float,
            ).reshape(4)

            stage_reference = (
                make_effector_stage_reference(
                    effector_reference,
                    previous_effector_quaternion=previous_effector_quaternion,
                    previous_body_quaternion=previous_body_quaternion,
                    previous_alpha=previous_alpha_ref,
                    transform=DEFAULT_END_EFFECTOR,
                )
            )

            refined_thrust = (
                stage_reference.parameters[
                    P_THRUST
                ].copy()
            )

            refined_alpha = (
                stage_reference.parameters[
                    P_ALPHA
                ].copy()
            )

            if sample_index == 0:
                refined_step = np.zeros(
                    4,
                    dtype=float,
                )

                refined_physical_step = np.zeros(
                    4,
                    dtype=float,
                )

            else:
                refined_step = (
                    refined_alpha
                    - previous_logged_alpha_ref
                )

                refined_physical_step = (
                    nearest_physical_angle_difference(
                        refined_alpha,
                        previous_logged_alpha_ref,
                    )
                )

            max_raw_alpha_abs = np.maximum(
                max_raw_alpha_abs,
                np.abs(
                    raw_alpha
                ),
            )

            max_refined_alpha_abs = np.maximum(
                max_refined_alpha_abs,
                np.abs(
                    refined_alpha
                ),
            )

            max_refined_step = np.maximum(
                max_refined_step,
                np.abs(
                    refined_step
                ),
            )

            max_refined_physical_step = np.maximum(
                max_refined_physical_step,
                np.abs(
                    refined_physical_step
                ),
            )

            min_raw_thrust = np.minimum(
                min_raw_thrust,
                raw_thrust,
            )

            low = (
                raw_thrust
                < SINGULARITY_THRUST_THRESHOLD
            )

            low_thrust_samples += low.astype(
                int
            )

            # Diagnostic only: flag references that are already within
            # 5 degrees of +/-180 degrees.
            near_limit = (
                np.abs(
                    np.degrees(
                        refined_alpha
                    )
                )
                > 175.0
            )

            near_limit_samples += near_limit.astype(
                int
            )

            for rotor in range(4):
                if (
                    near_limit[rotor]
                    and first_near_limit_time[rotor] is None
                ):
                    first_near_limit_time[rotor] = float(
                        time_s
                    )

            row = {
                "time_s": float(
                    time_s
                ),
                "pitch_ref_deg": float(
                    np.degrees(
                        pitch_rad
                    )
                ),
                "pitch_rate_ref_degps": float(
                    np.degrees(
                        pitch_rate_radps
                    )
                ),

                "wrench_Fx_B_N": wrench_reference_B[0],
                "wrench_Fy_B_N": wrench_reference_B[1],
                "wrench_Fz_B_N": wrench_reference_B[2],
                "wrench_Tx_B_Nm": wrench_reference_B[3],
                "wrench_Ty_B_Nm": wrench_reference_B[4],
                "wrench_Tz_B_Nm": wrench_reference_B[5],
            }

            for rotor in range(4):
                index = rotor + 1

                row[
                    f"raw_thrust_{index}_N"
                ] = raw_thrust[rotor]

                row[
                    f"raw_alpha_{index}_deg"
                ] = np.degrees(
                    raw_alpha[rotor]
                )

                row[
                    f"refined_thrust_{index}_N"
                ] = refined_thrust[rotor]

                row[
                    f"refined_alpha_{index}_deg"
                ] = np.degrees(
                    refined_alpha[rotor]
                )

                row[
                    f"refined_alpha_step_{index}_deg"
                ] = np.degrees(
                    refined_step[rotor]
                )

                row[
                    f"refined_alpha_physical_step_{index}_deg"
                ] = np.degrees(
                    refined_physical_step[rotor]
                )

                row[
                    f"raw_low_thrust_{index}"
                ] = int(
                    low[rotor]
                )

            writer.writerow(
                row
            )

            previous_effector_quaternion = (
                stage_reference.parameters[
                    6:10
                ].copy()
            )

            previous_body_quaternion = (
                stage_reference.x_reference[
                    6:10
                ].copy()
            )

            previous_alpha_ref = (
                refined_alpha.copy()
            )

            previous_logged_alpha_ref = (
                refined_alpha.copy()
            )

    print()
    print("=" * 78)
    print("PAPER-STYLE PITCH REFERENCE / ALLOCATION DIAGNOSTIC")
    print("=" * 78)

    print()
    print(
        "Trajectory:"
    )

    print(
        "  0-2 s   : hold 0 deg"
    )

    print(
        "  2-7 s   : 0 -> +180 deg"
    )

    print(
        "  7-10 s  : hold +180 deg"
    )

    print(
        " 10-20 s  : +180 -> -180 deg (full 360 deg cycle)"
    )

    print(
        " 20-23 s  : hold -180 deg"
    )

    print(
        " 23-28 s  : -180 -> 0 deg"
    )

    print(
        " 28-31 s  : hold 0 deg"
    )

    print()
    print(
        "Singularity threshold: "
        f"{SINGULARITY_THRUST_THRESHOLD:.3f} N"
    )

    print()

    for rotor in range(4):
        print(
            f"Rotor {rotor + 1}:"
        )

        print(
            "  minimum raw thrust         : "
            f"{min_raw_thrust[rotor]:.6f} N"
        )

        print(
            "  raw low-thrust samples     : "
            f"{low_thrust_samples[rotor]}"
        )

        print(
            "  max |raw alpha|            : "
            f"{np.degrees(max_raw_alpha_abs[rotor]):.3f} deg"
        )

        print(
            "  max |refined alpha|        : "
            f"{np.degrees(max_refined_alpha_abs[rotor]):.3f} deg"
        )

        print(
            "  max raw refined step       : "
            f"{np.degrees(max_refined_step[rotor]):.3f} deg / 10 ms"
        )

        print(
            "  max physical refined step  : "
            f"{np.degrees(max_refined_physical_step[rotor]):.3f} deg / 10 ms"
        )

        print(
            "  samples |alpha_ref|>175deg : "
            f"{near_limit_samples[rotor]}"
        )

        print(
            "  first >175deg time         : "
            f"{first_near_limit_time[rotor]}"
        )

        print()

    print(
        "CSV saved to:"
    )

    print(
        csv_path
    )

    print()
    print(
        "This script does NOT run the NMPC or MuJoCo plant. "
        "It isolates the reference-generation and singularity-handling "
        "layer first."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
