from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tiltable_quadrotor_rl.control.nmpc_effector_reference import (
    EndEffectorReferencePoint,
)


# =====================================================================
# General 6-DoF end-effector benchmark
# =====================================================================
#
# The purpose of this trajectory is to test the existing end-effector
# NMPC without changing the controller:
#
#   simultaneous X/Y/Z translation
#   +
#   simultaneous roll/pitch/yaw motion
#
# The trajectory starts and ends at a stationary identity-orientation
# hover. A quintic activity window ramps the harmonic motion in and out
# with zero first and second derivatives at the boundaries.
#
# ZYX convention:
#
#   q_WE = q_z(yaw) * q_y(pitch) * q_x(roll)
#
# omega_E and omega_dot_E are analytically generated in the local
# end-effector/body frame, consistent with the quaternion reference.


CENTER_POSITION_W = np.array(
    [
        0.0,
        0.0,
        1.5,
    ],
    dtype=float,
)


TRANSLATION_AMPLITUDE_M = np.array(
    [
        0.35,   # X
        0.30,   # Y
        0.20,   # Z
    ],
    dtype=float,
)

TRANSLATION_PERIOD_S = np.array(
    [
        8.0,
        10.0,
        6.0,
    ],
    dtype=float,
)

TRANSLATION_PHASE_RAD = np.array(
    [
        0.0,
        1.1,
        -0.7,
    ],
    dtype=float,
)


EULER_AMPLITUDE_DEG = np.array(
    [
        40.0,   # roll
        30.0,   # pitch
        60.0,   # yaw
    ],
    dtype=float,
)

EULER_AMPLITUDE_RAD = np.radians(
    EULER_AMPLITUDE_DEG
)

EULER_PERIOD_S = np.array(
    [
        7.0,
        9.0,
        11.0,
    ],
    dtype=float,
)

EULER_PHASE_RAD = np.array(
    [
        0.4,
        -0.9,
        0.8,
    ],
    dtype=float,
)


INITIAL_HOLD_DURATION_S = 2.0
RAMP_IN_DURATION_S = 3.0
FULL_ACTIVITY_DURATION_S = 24.0
RAMP_OUT_DURATION_S = 3.0
FINAL_HOLD_DURATION_S = 3.0

# Reuse the gain identified from the isolated pitch/roll experiments.
# It is intentionally NOT re-fitted to the 6-DoF benchmark.
RATE_BIAS_GAIN_M_PER_DEGPS = 0.000750


@dataclass(frozen=True)
class ScalarKinematics:
    value: float
    velocity: float
    acceleration: float
    jerk: float


@dataclass(frozen=True)
class EulerKinematics:
    angles_rad: np.ndarray
    rates_radps: np.ndarray
    accelerations_radps2: np.ndarray
    jerks_radps3: np.ndarray
    omega_E: np.ndarray
    omega_dot_E: np.ndarray


@dataclass(frozen=True)
class BiasCompensationState:
    position_offset_W: np.ndarray
    velocity_offset_W: np.ndarray
    acceleration_offset_W: np.ndarray


class General6DofEndEffectorTrajectory:
    """
    Smooth simultaneous translation + orientation benchmark.

    Translation:
        p_E^W = p_center + envelope(t) * harmonic_xyz(t)

    Orientation:
        roll/pitch/yaw are independent windowed harmonics.

    The Euler-angle derivatives are converted analytically to local
    end-effector/body angular velocity so the quaternion, omega_E and
    omega_dot_E references are kinematically consistent.
    """

    def __init__(
        self,
        *,
        center_position_W: np.ndarray = CENTER_POSITION_W,
        bias_compensation: bool = False,
    ) -> None:
        center = np.asarray(
            center_position_W,
            dtype=float,
        ).reshape(-1)

        if center.shape != (3,):
            raise ValueError(
                "center_position_W must have shape (3,), "
                f"got {center.shape}."
            )

        self._center_position_W = (
            center.copy()
        )

        self._bias_compensation = bool(
            bias_compensation
        )

        self._initial_hold_end = (
            INITIAL_HOLD_DURATION_S
        )

        self._ramp_in_end = (
            self._initial_hold_end
            + RAMP_IN_DURATION_S
        )

        self._full_activity_end = (
            self._ramp_in_end
            + FULL_ACTIVITY_DURATION_S
        )

        self._ramp_out_end = (
            self._full_activity_end
            + RAMP_OUT_DURATION_S
        )

        self._final_hold_end = (
            self._ramp_out_end
            + FINAL_HOLD_DURATION_S
        )

    @property
    def start_time_s(self) -> float:
        return 0.0

    @property
    def active_start_s(self) -> float:
        return float(
            self._initial_hold_end
        )

    @property
    def full_activity_start_s(self) -> float:
        return float(
            self._ramp_in_end
        )

    @property
    def full_activity_end_s(self) -> float:
        return float(
            self._full_activity_end
        )

    @property
    def active_end_s(self) -> float:
        return float(
            self._ramp_out_end
        )

    @property
    def end_time_s(self) -> float:
        return float(
            self._final_hold_end
        )

    @property
    def center_position_W(self) -> np.ndarray:
        return (
            self._center_position_W.copy()
        )

    @property
    def bias_compensation(self) -> bool:
        return self._bias_compensation

    @property
    def rate_bias_gain_m_per_degps(self) -> float:
        return RATE_BIAS_GAIN_M_PER_DEGPS

    def schedule_lines(self) -> list[str]:
        return [
            (
                f"{0.0:5.1f}-{self._initial_hold_end:5.1f} s : "
                "stationary center hover"
            ),
            (
                f"{self._initial_hold_end:5.1f}-{self._ramp_in_end:5.1f} s : "
                "smoothly ramp in all 6 trajectory channels"
            ),
            (
                f"{self._ramp_in_end:5.1f}-{self._full_activity_end:5.1f} s : "
                "full simultaneous XYZ + roll/pitch/yaw motion"
            ),
            (
                f"{self._full_activity_end:5.1f}-{self._ramp_out_end:5.1f} s : "
                "smoothly ramp all 6 channels back to zero"
            ),
            (
                f"{self._ramp_out_end:5.1f}-{self._final_hold_end:5.1f} s : "
                "final stationary center hover"
            ),
        ]

    @staticmethod
    def _smootherstep(
        r: float,
    ) -> tuple[float, float, float]:
        """
        Quintic smootherstep and first two derivatives with respect to r.

        s(0)=0, s(1)=1
        s'(0)=s'(1)=0
        s''(0)=s''(1)=0
        """
        r = float(
            np.clip(
                r,
                0.0,
                1.0,
            )
        )

        s = (
            10.0 * r**3
            - 15.0 * r**4
            + 6.0 * r**5
        )

        ds_dr = (
            30.0 * r**2
            - 60.0 * r**3
            + 30.0 * r**4
        )

        d2s_dr2 = (
            60.0 * r
            - 180.0 * r**2
            + 120.0 * r**3
        )

        d3s_dr3 = (
            60.0
            - 360.0 * r
            + 360.0 * r**2
        )

        return (
            float(s),
            float(ds_dr),
            float(d2s_dr2),
            float(d3s_dr3),
        )

    def activity_window(
        self,
        time_s: float,
    ) -> tuple[float, float, float, float]:
        """
        Return:
            w(t), w_dot(t), w_ddot(t), w_dddot(t)
        """
        t = float(
            np.clip(
                time_s,
                self.start_time_s,
                self.end_time_s,
            )
        )

        if t < self._initial_hold_end:
            return (
                0.0,
                0.0,
                0.0,
                0.0,
            )

        if t < self._ramp_in_end:
            r = (
                t
                - self._initial_hold_end
            ) / RAMP_IN_DURATION_S

            (
                s,
                ds_dr,
                d2s_dr2,
                d3s_dr3,
            ) = self._smootherstep(
                r
            )

            return (
                s,
                ds_dr
                / RAMP_IN_DURATION_S,
                d2s_dr2
                / RAMP_IN_DURATION_S**2,
                d3s_dr3
                / RAMP_IN_DURATION_S**3,
            )

        if t < self._full_activity_end:
            return (
                1.0,
                0.0,
                0.0,
                0.0,
            )

        if t < self._ramp_out_end:
            r = (
                t
                - self._full_activity_end
            ) / RAMP_OUT_DURATION_S

            (
                s,
                ds_dr,
                d2s_dr2,
                d3s_dr3,
            ) = self._smootherstep(
                r
            )

            return (
                1.0 - s,
                -ds_dr
                / RAMP_OUT_DURATION_S,
                -d2s_dr2
                / RAMP_OUT_DURATION_S**2,
                -d3s_dr3
                / RAMP_OUT_DURATION_S**3,
            )

        return (
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def _carrier_time_s(
        self,
        time_s: float,
    ) -> float:
        """
        Carrier time starts when the activity window starts.

        The carrier continues through ramp-in, full activity and
        ramp-out. The zero-valued window makes the initial/final holds
        stationary regardless of carrier phase.
        """
        return max(
            0.0,
            float(time_s)
            - self._initial_hold_end,
        )

    def _windowed_harmonic(
        self,
        time_s: float,
        *,
        amplitude: float,
        period_s: float,
        phase_rad: float,
    ) -> ScalarKinematics:
        if period_s <= 0.0:
            raise ValueError(
                "period_s must be positive."
            )

        (
            window,
            window_dot,
            window_ddot,
            window_dddot,
        ) = self.activity_window(
            time_s
        )

        carrier_time = (
            self._carrier_time_s(
                time_s
            )
        )

        omega = (
            2.0
            * np.pi
            / period_s
        )

        phase = (
            omega
            * carrier_time
            + phase_rad
        )

        sine = np.sin(
            phase
        )

        cosine = np.cos(
            phase
        )

        value = (
            amplitude
            * window
            * sine
        )

        velocity = (
            amplitude
            * (
                window_dot
                * sine
                + window
                * omega
                * cosine
            )
        )

        acceleration = (
            amplitude
            * (
                window_ddot
                * sine
                + 2.0
                * window_dot
                * omega
                * cosine
                - window
                * omega**2
                * sine
            )
        )

        jerk = (
            amplitude
            * (
                window_dddot
                * sine
                + 3.0
                * window_ddot
                * omega
                * cosine
                - 3.0
                * window_dot
                * omega**2
                * sine
                - window
                * omega**3
                * cosine
            )
        )

        return ScalarKinematics(
            value=float(
                value
            ),
            velocity=float(
                velocity
            ),
            acceleration=float(
                acceleration
            ),
            jerk=float(
                jerk
            ),
        )

    def translation_state(
        self,
        time_s: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        offsets = np.zeros(
            3,
            dtype=float,
        )

        velocities = np.zeros(
            3,
            dtype=float,
        )

        accelerations = np.zeros(
            3,
            dtype=float,
        )

        for axis in range(3):
            state = (
                self._windowed_harmonic(
                    time_s,
                    amplitude=float(
                        TRANSLATION_AMPLITUDE_M[
                            axis
                        ]
                    ),
                    period_s=float(
                        TRANSLATION_PERIOD_S[
                            axis
                        ]
                    ),
                    phase_rad=float(
                        TRANSLATION_PHASE_RAD[
                            axis
                        ]
                    ),
                )
            )

            offsets[
                axis
            ] = state.value

            velocities[
                axis
            ] = state.velocity

            accelerations[
                axis
            ] = state.acceleration

        return (
            self._center_position_W
            + offsets,
            velocities,
            accelerations,
        )

    @staticmethod
    def _quaternion_from_zyx(
        roll: float,
        pitch: float,
        yaw: float,
    ) -> np.ndarray:
        """
        q_WE = q_z(yaw) * q_y(pitch) * q_x(roll)

        Quaternion ordering:
            [w, x, y, z]
        """
        cr = np.cos(
            0.5 * roll
        )
        sr = np.sin(
            0.5 * roll
        )

        cp = np.cos(
            0.5 * pitch
        )
        sp = np.sin(
            0.5 * pitch
        )

        cy = np.cos(
            0.5 * yaw
        )
        sy = np.sin(
            0.5 * yaw
        )

        quaternion = np.array(
            [
                (
                    cy * cp * cr
                    + sy * sp * sr
                ),
                (
                    cy * cp * sr
                    - sy * sp * cr
                ),
                (
                    cy * sp * cr
                    + sy * cp * sr
                ),
                (
                    sy * cp * cr
                    - cy * sp * sr
                ),
            ],
            dtype=float,
        )

        quaternion /= np.linalg.norm(
            quaternion
        )

        return quaternion

    def euler_state(
        self,
        time_s: float,
    ) -> EulerKinematics:
        angles = np.zeros(
            3,
            dtype=float,
        )

        rates = np.zeros(
            3,
            dtype=float,
        )

        accelerations = np.zeros(
            3,
            dtype=float,
        )

        jerks = np.zeros(
            3,
            dtype=float,
        )

        for axis in range(3):
            state = (
                self._windowed_harmonic(
                    time_s,
                    amplitude=float(
                        EULER_AMPLITUDE_RAD[
                            axis
                        ]
                    ),
                    period_s=float(
                        EULER_PERIOD_S[
                            axis
                        ]
                    ),
                    phase_rad=float(
                        EULER_PHASE_RAD[
                            axis
                        ]
                    ),
                )
            )

            angles[
                axis
            ] = state.value

            rates[
                axis
            ] = state.velocity

            accelerations[
                axis
            ] = state.acceleration

            jerks[
                axis
            ] = state.jerk

        (
            roll,
            pitch,
            _yaw,
        ) = angles

        (
            roll_dot,
            pitch_dot,
            yaw_dot,
        ) = rates

        (
            roll_ddot,
            pitch_ddot,
            yaw_ddot,
        ) = accelerations

        sin_roll = np.sin(
            roll
        )
        cos_roll = np.cos(
            roll
        )

        sin_pitch = np.sin(
            pitch
        )
        cos_pitch = np.cos(
            pitch
        )

        # Body/local angular velocity for ZYX Euler angles.
        omega_E = np.array(
            [
                (
                    roll_dot
                    - yaw_dot
                    * sin_pitch
                ),
                (
                    pitch_dot
                    * cos_roll
                    + yaw_dot
                    * sin_roll
                    * cos_pitch
                ),
                (
                    -pitch_dot
                    * sin_roll
                    + yaw_dot
                    * cos_roll
                    * cos_pitch
                ),
            ],
            dtype=float,
        )

        # Time derivative of the body/local angular-velocity
        # components. This is the omega_dot quantity used by the rigid
        # body dynamics and end-effector inverse kinematics.
        omega_dot_E = np.array(
            [
                (
                    roll_ddot
                    - yaw_ddot
                    * sin_pitch
                    - yaw_dot
                    * cos_pitch
                    * pitch_dot
                ),
                (
                    pitch_ddot
                    * cos_roll
                    - pitch_dot
                    * sin_roll
                    * roll_dot
                    + yaw_ddot
                    * sin_roll
                    * cos_pitch
                    + yaw_dot
                    * cos_roll
                    * roll_dot
                    * cos_pitch
                    - yaw_dot
                    * sin_roll
                    * sin_pitch
                    * pitch_dot
                ),
                (
                    -pitch_ddot
                    * sin_roll
                    - pitch_dot
                    * cos_roll
                    * roll_dot
                    + yaw_ddot
                    * cos_roll
                    * cos_pitch
                    - yaw_dot
                    * sin_roll
                    * roll_dot
                    * cos_pitch
                    - yaw_dot
                    * cos_roll
                    * sin_pitch
                    * pitch_dot
                ),
            ],
            dtype=float,
        )

        return EulerKinematics(
            angles_rad=angles,
            rates_radps=rates,
            accelerations_radps2=(
                accelerations
            ),
            jerks_radps3=(
                jerks
            ),
            omega_E=omega_E,
            omega_dot_E=omega_dot_E,
        )

    def _bias_compensation_from_euler(
        self,
        euler: EulerKinematics,
    ) -> BiasCompensationState:
        """
        Evaluate the existing empirical bias law from already-computed
        Euler kinematics.

        This helper avoids recomputing the complete harmonic/Euler
        trajectory inside every NMPC horizon sample.
        """
        zero = np.zeros(
            3,
            dtype=float,
        )

        if not self._bias_compensation:
            return BiasCompensationState(
                position_offset_W=zero.copy(),
                velocity_offset_W=zero.copy(),
                acceleration_offset_W=zero.copy(),
            )

        rates_degps = np.degrees(
            euler.rates_radps
        )

        accelerations_degps2 = np.degrees(
            euler.accelerations_radps2
        )

        jerks_degps3 = np.degrees(
            euler.jerks_radps3
        )

        position_offset = np.array(
            [
                RATE_BIAS_GAIN_M_PER_DEGPS
                * rates_degps[1],
                -RATE_BIAS_GAIN_M_PER_DEGPS
                * rates_degps[0],
                0.0,
            ],
            dtype=float,
        )

        velocity_offset = np.array(
            [
                RATE_BIAS_GAIN_M_PER_DEGPS
                * accelerations_degps2[1],
                -RATE_BIAS_GAIN_M_PER_DEGPS
                * accelerations_degps2[0],
                0.0,
            ],
            dtype=float,
        )

        acceleration_offset = np.array(
            [
                RATE_BIAS_GAIN_M_PER_DEGPS
                * jerks_degps3[1],
                -RATE_BIAS_GAIN_M_PER_DEGPS
                * jerks_degps3[0],
                0.0,
            ],
            dtype=float,
        )

        return BiasCompensationState(
            position_offset_W=position_offset,
            velocity_offset_W=velocity_offset,
            acceleration_offset_W=acceleration_offset,
        )

    def bias_compensation_state(
        self,
        time_s: float,
    ) -> BiasCompensationState:
        """
        Reuse the previously identified rate-bias law:

            Delta x = +k_b * pitch_rate_degps
            Delta y = -k_b * roll_rate_degps
            Delta z = 0

        This public method is retained for logging/inspection.
        """
        euler = self.euler_state(
            time_s
        )

        return self._bias_compensation_from_euler(
            euler
        )

    def true_sample(
        self,
        time_s: float,
    ) -> EndEffectorReferencePoint:
        """
        Original physical benchmark target, before compensation.
        """
        (
            position_W,
            velocity_W,
            acceleration_W,
        ) = self.translation_state(
            time_s
        )

        euler = self.euler_state(
            time_s
        )

        (
            roll,
            pitch,
            yaw,
        ) = euler.angles_rad

        quaternion_WE = (
            self._quaternion_from_zyx(
                float(
                    roll
                ),
                float(
                    pitch
                ),
                float(
                    yaw
                ),
            )
        )

        return EndEffectorReferencePoint(
            position_W=position_W,
            velocity_W=velocity_W,
            acceleration_W=acceleration_W,
            quaternion_WE=quaternion_WE,
            omega_E=euler.omega_E,
            omega_dot_E=euler.omega_dot_E,
        )

    def sample(
        self,
        time_s: float,
    ) -> EndEffectorReferencePoint:
        """
        Reference supplied to the NMPC.

        With compensation disabled this is exactly the physical target.
        With compensation enabled, only the translational reference is
        shifted. The requested tool orientation remains unchanged.

        Translation and Euler kinematics are evaluated only once per
        call so the 21-node NMPC reference horizon does not duplicate
        trajectory work.
        """
        (
            position_W,
            velocity_W,
            acceleration_W,
        ) = self.translation_state(
            time_s
        )

        euler = self.euler_state(
            time_s
        )

        (
            roll,
            pitch,
            yaw,
        ) = euler.angles_rad

        quaternion_WE = (
            self._quaternion_from_zyx(
                float(
                    roll
                ),
                float(
                    pitch
                ),
                float(
                    yaw
                ),
            )
        )

        compensation = (
            self._bias_compensation_from_euler(
                euler
            )
        )

        return EndEffectorReferencePoint(
            position_W=(
                position_W
                + compensation.position_offset_W
            ),
            velocity_W=(
                velocity_W
                + compensation.velocity_offset_W
            ),
            acceleration_W=(
                acceleration_W
                + compensation.acceleration_offset_W
            ),
            quaternion_WE=quaternion_WE,
            omega_E=euler.omega_E,
            omega_dot_E=euler.omega_dot_E,
        )
