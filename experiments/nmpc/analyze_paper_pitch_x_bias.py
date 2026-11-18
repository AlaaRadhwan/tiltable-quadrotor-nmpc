from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TOOL_OFFSET_M = 0.15

Q_POSITION = np.array([600.0, 600.0, 800.0])
Q_VELOCITY = np.array([30.0, 30.0, 30.0])
Q_ATTITUDE = np.array([300.0, 300.0, 600.0])
Q_OMEGA = np.array([5.0, 5.0, 5.0])

Q_ALPHA = 2.0
R_THRUST = 2.0
R_DELTA_ALPHA = 250.0


def rotate_tool_offset(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=float)
    w, x, y, z = q.T

    return TOOL_OFFSET_M * np.column_stack(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        )
    )


def quaternion_error_vector(
    q_ref: np.ndarray,
    q_actual: np.ndarray,
) -> np.ndarray:
    q_ref = q_ref.copy()

    signs = np.sum(q_ref * q_actual, axis=1) < 0.0
    q_ref[signs] *= -1.0

    a = q_ref.copy()
    a[:, 1:4] *= -1.0

    aw, ax, ay, az = a.T
    bw, bx, by, bz = q_actual.T

    return np.column_stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def mean_weighted_square(
    residual: np.ndarray,
    weights: np.ndarray | float,
    mask: np.ndarray,
) -> float:
    value = residual[mask]

    if np.isscalar(weights):
        return float(
            np.mean(
                np.sum(
                    float(weights) * value * value,
                    axis=1,
                )
            )
        )

    return float(
        np.mean(
            np.sum(
                value * value * np.asarray(weights),
                axis=1,
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose X-direction fixed-pivot tracking bias in a "
            "paper-style pitch closed-loop CSV."
        )
    )
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    time_s = df["time_s"].to_numpy()
    dt = float(np.median(np.diff(time_s)))

    pitch_ref_rad = np.deg2rad(
        df["pitch_ref_deg"].to_numpy()
    )

    pitch_rate_ref_degps = (
        df["pitch_rate_ref_degps"].to_numpy()
    )

    pitch_rate_ref_radps = np.deg2rad(
        pitch_rate_ref_degps
    )

    q_actual = df[
        ["qw", "qx", "qy", "qz"]
    ].to_numpy()

    body_position = df[
        ["body_px_m", "body_py_m", "body_pz_m"]
    ].to_numpy()

    tool_position = (
        body_position
        + rotate_tool_offset(q_actual)
    )

    tool_target = np.array([0.0, 0.0, 1.5])
    tool_error = tool_position - tool_target

    tool_velocity = np.gradient(
        tool_position,
        dt,
        axis=0,
    )

    desired_body_x = (
        -TOOL_OFFSET_M
        * np.sin(pitch_ref_rad)
    )

    desired_body_z = (
        1.5
        - TOOL_OFFSET_M
        * np.cos(pitch_ref_rad)
    )

    body_x_error = (
        body_position[:, 0]
        - desired_body_x
    )

    body_z_error = (
        body_position[:, 2]
        - desired_body_z
    )

    # Use the middle portions of the three constant-rate segments to
    # avoid entry/exit transients when diagnosing the persistent bias.
    constant_rate_mask = (
        ((time_s >= 2.5) & (time_s <= 6.5))
        | ((time_s >= 11.0) & (time_s <= 19.0))
        | ((time_s >= 23.5) & (time_s <= 27.5))
    )

    design = np.column_stack(
        (
            np.ones(np.count_nonzero(constant_rate_mask)),
            pitch_rate_ref_degps[constant_rate_mask],
        )
    )

    coefficients = np.linalg.lstsq(
        design,
        tool_error[constant_rate_mask, 0],
        rcond=None,
    )[0]

    prediction = design @ coefficients
    actual = tool_error[constant_rate_mask, 0]

    r_squared = (
        1.0
        - np.sum((actual - prediction) ** 2)
        / np.sum((actual - np.mean(actual)) ** 2)
    )

    actual_alpha = np.deg2rad(
        df[
            [
                "actual_alpha_1_deg",
                "actual_alpha_2_deg",
                "actual_alpha_3_deg",
                "actual_alpha_4_deg",
            ]
        ].to_numpy()
    )

    alpha_cmd = np.deg2rad(
        df[
            [
                "alpha_cmd_1_deg",
                "alpha_cmd_2_deg",
                "alpha_cmd_3_deg",
                "alpha_cmd_4_deg",
            ]
        ].to_numpy()
    )

    alpha_ref = np.deg2rad(
        df[
            [
                "refined_alpha_ref_1_deg",
                "refined_alpha_ref_2_deg",
                "refined_alpha_ref_3_deg",
                "refined_alpha_ref_4_deg",
            ]
        ].to_numpy()
    )

    thrust_cmd = df[
        [
            "thrust_cmd_1_N",
            "thrust_cmd_2_N",
            "thrust_cmd_3_N",
            "thrust_cmd_4_N",
        ]
    ].to_numpy()

    thrust_ref = df[
        [
            "refined_thrust_ref_1_N",
            "refined_thrust_ref_2_N",
            "refined_thrust_ref_3_N",
            "refined_thrust_ref_4_N",
        ]
    ].to_numpy()

    omega = df[
        [
            "omega_x_radps",
            "omega_y_radps",
            "omega_z_radps",
        ]
    ].to_numpy()

    omega_ref = np.column_stack(
        (
            np.zeros(len(df)),
            pitch_rate_ref_radps,
            np.zeros(len(df)),
        )
    )

    q_ref = np.column_stack(
        (
            np.cos(0.5 * pitch_ref_rad),
            np.zeros(len(df)),
            np.sin(0.5 * pitch_ref_rad),
            np.zeros(len(df)),
        )
    )

    attitude_error_vector = (
        quaternion_error_vector(
            q_ref,
            q_actual,
        )
    )

    alpha_error = actual_alpha - alpha_ref
    thrust_error = thrust_cmd - thrust_ref
    delta_alpha = alpha_cmd - actual_alpha

    moving_mask = (
        ((time_s >= 2.0) & (time_s <= 7.0))
        | ((time_s >= 10.0) & (time_s <= 20.0))
        | ((time_s >= 23.0) & (time_s <= 28.0))
    )

    contributions = {
        "tool position": mean_weighted_square(
            tool_error,
            Q_POSITION,
            moving_mask,
        ),
        "tool velocity": mean_weighted_square(
            tool_velocity,
            Q_VELOCITY,
            moving_mask,
        ),
        "attitude": mean_weighted_square(
            attitude_error_vector,
            Q_ATTITUDE,
            moving_mask,
        ),
        "angular rate": mean_weighted_square(
            omega - omega_ref,
            Q_OMEGA,
            moving_mask,
        ),
        "alpha reference": mean_weighted_square(
            alpha_error,
            Q_ALPHA,
            moving_mask,
        ),
        "thrust reference": mean_weighted_square(
            thrust_error,
            R_THRUST,
            moving_mask,
        ),
        "alpha_cmd-alpha": mean_weighted_square(
            delta_alpha,
            R_DELTA_ALPHA,
            moving_mask,
        ),
    }

    print()
    print("FIXED-PIVOT X-BIAS DIAGNOSTIC")
    print("=" * 72)

    axis_rms_cm = (
        100.0
        * np.sqrt(
            np.mean(
                tool_error * tool_error,
                axis=0,
            )
        )
    )

    print()
    print("Tool error RMS by axis [cm]")
    print(
        f"  X: {axis_rms_cm[0]:.4f}"
    )
    print(
        f"  Y: {axis_rms_cm[1]:.4f}"
    )
    print(
        f"  Z: {axis_rms_cm[2]:.4f}"
    )

    print()
    print("Desired vs actual body path")
    print(
        "  body-X RMS error [cm]: "
        f"{100.0 * np.sqrt(np.mean(body_x_error**2)):.4f}"
    )
    print(
        "  body-Z RMS error [cm]: "
        f"{100.0 * np.sqrt(np.mean(body_z_error**2)):.4f}"
    )

    print()
    print("Constant-rate X-bias fit")
    print(
        "  tool_x_error [m] ~= "
        f"{coefficients[0]:+.6f} "
        f"{coefficients[1]:+.6f} * pitch_rate_ref [deg/s]"
    )
    print(
        f"  R^2: {r_squared:.6f}"
    )
    print(
        "  equivalent gain [cm/(deg/s)]: "
        f"{100.0 * coefficients[1]:+.6f}"
    )

    print()
    print("Representative constant-rate segments")

    for name, start, end in (
        ("entry +36 deg/s", 2.5, 6.5),
        ("full  -36 deg/s", 11.0, 19.0),
        ("exit  +36 deg/s", 23.5, 27.5),
    ):
        mask = (
            (time_s >= start)
            & (time_s <= end)
        )

        print(
            f"  {name}: "
            f"mean tool-X error = "
            f"{100.0 * np.mean(tool_error[mask, 0]):+.3f} cm, "
            f"std = "
            f"{100.0 * np.std(tool_error[mask, 0]):.3f} cm, "
            f"mean tool vx = "
            f"{100.0 * np.mean(tool_velocity[mask, 0]):+.3f} cm/s"
        )

    delta_alpha_deg = np.rad2deg(
        delta_alpha[moving_mask]
    )

    print()
    print("Servo command-state separation during moving segments")
    print(
        "  RMS [deg]: "
        f"{np.sqrt(np.mean(delta_alpha_deg**2)):.4f}"
    )
    print(
        "  max |alpha_cmd-alpha| [deg]: "
        f"{np.max(np.abs(delta_alpha_deg)):.4f}"
    )

    print()
    print("Approximate mean weighted stage-cost contributions")
    for name, value in contributions.items():
        print(
            f"  {name:20s}: {value:.6f}"
        )

    print()
    print("Interpretation")
    print(
        "  A high R^2 for tool-X error versus pitch rate, together "
        "with near-zero mean tool-X velocity during constant-rate "
        "motion, indicates a persistent rate-dependent offset rather "
        "than simple accumulated drift."
    )
    print(
        "  Because alpha_cmd-alpha = tau*alpha_dot in the first-order "
        "servo model, R_DELTA_ALPHA directly regularizes servo rate. "
        "If its weighted contribution is comparable to or larger than "
        "the tool-position contribution, reducing that term is the "
        "cleanest next single-variable experiment."
    )
    print()


if __name__ == "__main__":
    main()
