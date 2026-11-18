from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from tiltable_quadrotor_rl.control.end_effector import (
    quaternion_product_np,
)
from tiltable_quadrotor_rl.control.nmpc_effector_reference import (
    EndEffectorReferencePoint,
)


PaperManeuver = Literal["pitch", "roll", "cartwheel", "yaw"]


# ---------------------------------------------------------------------
# Current identified fixed-pivot bias model
# ---------------------------------------------------------------------

# From the R_delta_alpha = 50 pitch/roll experiments:
# approximately 27 mm of steady lateral tool bias at 36 deg/s.
RATE_BIAS_GAIN_M_PER_DEGPS = 7.5e-4

# Validated pitch/roll baseline.
MAX_ANGULAR_RATE_DEGPS = 36.0
ANGULAR_RATE_RAMP_DURATION_S = 1.5

# Paper-style continuous cartwheel baseline:
# initialize at +90 deg roll and rotate about body/tool Z.
CARTWHEEL_MAX_ANGULAR_RATE_DEGPS = 12.0
CARTWHEEL_ANGULAR_RATE_RAMP_DURATION_S = 1.5

INITIAL_HOLD_DURATION_S = 2.0
INTERMEDIATE_HOLD_DURATION_S = 3.0
FINAL_HOLD_DURATION_S = 3.0

# Independent static cartwheel-pose diagnostic.
STATIC_CARTWHEEL_HOLD_DURATION_S = 5.0
STATIC_CARTWHEEL_STEADY_STATE_START_S = 2.0


@dataclass(frozen=True)
class BiasCompensationState:
    position_offset_W: np.ndarray
    velocity_offset_W: np.ndarray
    acceleration_offset_W: np.ndarray


@dataclass(frozen=True)
class RotationKinematics:
    angle_rad: float
    rate_radps: float
    acceleration_radps2: float
    jerk_radps3: float


def cartwheel_quaternion_WE(
    cartwheel_angle_rad: float,
) -> np.ndarray:
    """
    End-effector orientation for the vertical cartwheel.

    The vehicle is first placed at +90 deg roll and then rotated
    about its LOCAL/BODY Z axis:

        q_WE = q_x(+90 deg) * q_z(gamma)
    """
    quarter_pi = 0.25 * np.pi
    half = 0.5 * float(
        cartwheel_angle_rad
    )

    q_roll_90 = np.array(
        [
            np.cos(quarter_pi),
            np.sin(quarter_pi),
            0.0,
            0.0,
        ],
        dtype=float,
    )

    q_body_z = np.array(
        [
            np.cos(half),
            0.0,
            0.0,
            np.sin(half),
        ],
        dtype=float,
    )

    return quaternion_product_np(
        q_roll_90,
        q_body_z,
    )


class PaperStyleFixedEffectorTrajectory:
    """
    Reusable fixed-end-effector paper-style trajectory.

    Supported maneuvers
    -------------------
    pitch / roll:
        Existing validated benchmark:
          0 -> +180 deg
          hold
          +180 -> -180 deg (full physical revolution)
          hold
          -180 -> 0 deg

        Peak rate: 36 deg/s.
        Optional empirical rate-bias compensation is available.

    cartwheel:
        The vehicle is initialized at +90 deg roll and then performs
        one continuous 360 deg rotation about its BODY/TOOL Z axis
        while the end-effector origin remains fixed in world.

        The cartwheel intentionally uses the slower paper-style
        12 deg/s peak rate. Bias compensation is disabled because the
        pitch/roll empirical bias law has not been identified for this
        maneuver.

    All moving segments use a smooth angular-rate profile:
        smootherstep acceleration
        constant-rate cruise
        smootherstep deceleration
    """

    def __init__(
        self,
        *,
        position_W: np.ndarray,
        maneuver: PaperManeuver,
        bias_compensation: bool = False,
    ) -> None:
        position = np.asarray(
            position_W,
            dtype=float,
        ).reshape(-1)

        if position.shape != (3,):
            raise ValueError(
                "position_W must have shape (3,), "
                f"got {position.shape}."
            )

        if maneuver not in (
            "pitch",
            "roll",
            "cartwheel",
            "yaw",
        ):
            raise ValueError(
                "maneuver must be 'pitch', 'roll', 'cartwheel', or "
                f"'yaw', got {maneuver!r}."
            )

        if (
            maneuver in ("cartwheel", "yaw")
            and bias_compensation
        ):
            raise ValueError(
                "The empirical pitch/roll rate-bias "
                "compensation is not calibrated for the "
                f"{maneuver} maneuver."
            )

        self._position_W = position.copy()
        self._maneuver = maneuver
        self._bias_compensation = bool(
            bias_compensation
        )

        self._initial_hold_end = (
            INITIAL_HOLD_DURATION_S
        )

        if self._maneuver in ("cartwheel", "yaw"):
            self._cartwheel_start = (
                self._initial_hold_end
            )
            self._cartwheel_end = (
                self._cartwheel_start
                + self._move_duration_s(
                    0.0,
                    2.0 * np.pi,
                )
            )
            self._final_hold_end = (
                self._cartwheel_end
                + FINAL_HOLD_DURATION_S
            )
            return

        # Existing pitch / roll schedule.
        self._entry_start = (
            self._initial_hold_end
        )
        self._entry_end = (
            self._entry_start
            + self._move_duration_s(
                0.0,
                np.pi,
            )
        )

        self._positive_hold_end = (
            self._entry_end
            + INTERMEDIATE_HOLD_DURATION_S
        )

        self._full_cycle_start = (
            self._positive_hold_end
        )
        self._full_cycle_end = (
            self._full_cycle_start
            + self._move_duration_s(
                np.pi,
                -np.pi,
            )
        )

        self._negative_hold_end = (
            self._full_cycle_end
            + INTERMEDIATE_HOLD_DURATION_S
        )

        self._exit_start = (
            self._negative_hold_end
        )
        self._exit_end = (
            self._exit_start
            + self._move_duration_s(
                -np.pi,
                0.0,
            )
        )

        self._final_hold_end = (
            self._exit_end
            + FINAL_HOLD_DURATION_S
        )

    @property
    def maneuver(self) -> PaperManeuver:
        return self._maneuver

    @property
    def bias_compensation(self) -> bool:
        return self._bias_compensation

    @property
    def start_time_s(self) -> float:
        return 0.0

    @property
    def end_time_s(self) -> float:
        return float(
            self._final_hold_end
        )

    @property
    def position_W(self) -> np.ndarray:
        """
        TRUE physical fixed tool point.

        Any empirical controller-reference compensation is added only
        inside sample().
        """
        return self._position_W.copy()

    @property
    def max_angular_rate_degps(self) -> float:
        if self._maneuver == "cartwheel":
            return (
                CARTWHEEL_MAX_ANGULAR_RATE_DEGPS
            )

        return MAX_ANGULAR_RATE_DEGPS

    @property
    def angular_rate_ramp_duration_s(self) -> float:
        if self._maneuver == "cartwheel":
            return (
                CARTWHEEL_ANGULAR_RATE_RAMP_DURATION_S
            )

        return ANGULAR_RATE_RAMP_DURATION_S

    @property
    def maximum_compensation_m(self) -> float:
        if not self._bias_compensation:
            return 0.0

        return (
            RATE_BIAS_GAIN_M_PER_DEGPS
            * self.max_angular_rate_degps
        )

    def schedule_lines(self) -> list[str]:
        if self._maneuver == "yaw":
            return [
                (
                    f"{0.0:5.1f}-{self._cartwheel_start:5.1f} s : "
                    "hold level, yaw angle 0 deg"
                ),
                (
                    f"{self._cartwheel_start:5.1f}-"
                    f"{self._cartwheel_end:5.1f} s : "
                    "body-Z yaw 0 -> 360 deg, smooth rate, "
                    f"max +{self.max_angular_rate_degps:.0f} deg/s"
                ),
                (
                    f"{self._cartwheel_end:5.1f}-"
                    f"{self._final_hold_end:5.1f} s : "
                    "final hold at the completed 360 deg orientation"
                ),
            ]

        if self._maneuver == "cartwheel":
            return [
                (
                    f"{0.0:5.1f}-{self._cartwheel_start:5.1f} s : "
                    "hold +90 deg roll, cartwheel angle 0 deg"
                ),
                (
                    f"{self._cartwheel_start:5.1f}-"
                    f"{self._cartwheel_end:5.1f} s : "
                    "body-Z cartwheel 0 -> 360 deg, smooth rate, "
                    f"max +{self.max_angular_rate_degps:.0f} deg/s"
                ),
                (
                    f"{self._cartwheel_end:5.1f}-"
                    f"{self._final_hold_end:5.1f} s : "
                    "final hold at the completed 360 deg orientation"
                ),
            ]

        return [
            (
                f"{0.0:5.1f}-{self._entry_start:5.1f} s : "
                "hold 0 deg"
            ),
            (
                f"{self._entry_start:5.1f}-{self._entry_end:5.1f} s : "
                "0 -> +180 deg, smooth rate, "
                f"max +{self.max_angular_rate_degps:.0f} deg/s"
            ),
            (
                f"{self._entry_end:5.1f}-"
                f"{self._full_cycle_start:5.1f} s : "
                "hold +180 deg"
            ),
            (
                f"{self._full_cycle_start:5.1f}-"
                f"{self._full_cycle_end:5.1f} s : "
                "+180 -> -180 deg, full 360 deg, smooth rate, "
                f"max -{self.max_angular_rate_degps:.0f} deg/s"
            ),
            (
                f"{self._full_cycle_end:5.1f}-"
                f"{self._exit_start:5.1f} s : "
                "hold -180 deg"
            ),
            (
                f"{self._exit_start:5.1f}-{self._exit_end:5.1f} s : "
                "-180 -> 0 deg, smooth rate, "
                f"max +{self.max_angular_rate_degps:.0f} deg/s"
            ),
            (
                f"{self._exit_end:5.1f}-"
                f"{self._final_hold_end:5.1f} s : "
                "hold 0 deg"
            ),
        ]

    @staticmethod
    def _smootherstep(
        r: float,
    ) -> tuple[float, float, float]:
        """
        Quintic smootherstep and its first two derivatives with
        respect to the normalized coordinate r.
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

        return (
            float(s),
            float(ds_dr),
            float(d2s_dr2),
        )

    @staticmethod
    def _integral_smootherstep(
        r: float,
    ) -> float:
        """
        Integral of smootherstep from 0 to r.

        Integral over [0, 1] = 0.5.
        """
        r = float(
            np.clip(
                r,
                0.0,
                1.0,
            )
        )

        return float(
            2.5 * r**4
            - 3.0 * r**5
            + r**6
        )

    def _max_rate_radps(
        self,
    ) -> float:
        return float(
            np.radians(
                self.max_angular_rate_degps
            )
        )

    def _move_duration_s(
        self,
        start_angle_rad: float,
        end_angle_rad: float,
    ) -> float:
        delta = abs(
            end_angle_rad
            - start_angle_rad
        )

        max_rate = self._max_rate_radps()
        ramp = (
            self.angular_rate_ramp_duration_s
        )

        # Each smooth ramp contributes 0.5 * Vmax * ramp.
        # Both ramps together contribute Vmax * ramp.
        ramp_angle_total = (
            max_rate
            * ramp
        )

        if delta + 1e-12 < ramp_angle_total:
            raise ValueError(
                "Move is too short for the configured "
                "rate limit and ramp duration."
            )

        cruise_duration = (
            delta
            - ramp_angle_total
        ) / max_rate

        return float(
            2.0 * ramp
            + cruise_duration
        )

    def _smooth_move_kinematics(
        self,
        time_s: float,
        *,
        start_time_s: float,
        start_angle_rad: float,
        end_angle_rad: float,
    ) -> RotationKinematics:
        """
        Smooth-rate point-to-point rotation.

        Velocity profile:
            smootherstep 0 -> signed Vmax
            constant signed Vmax
            smootherstep signed Vmax -> 0

        Returns:
            angle
            angular rate
            angular acceleration
            angular jerk
        """
        delta = (
            end_angle_rad
            - start_angle_rad
        )

        if abs(delta) < 1e-15:
            return RotationKinematics(
                angle_rad=float(
                    start_angle_rad
                ),
                rate_radps=0.0,
                acceleration_radps2=0.0,
                jerk_radps3=0.0,
            )

        direction = (
            1.0
            if delta > 0.0
            else -1.0
        )

        vmax = (
            direction
            * self._max_rate_radps()
        )

        ramp = float(
            self.angular_rate_ramp_duration_s
        )

        total_duration = self._move_duration_s(
            start_angle_rad,
            end_angle_rad,
        )

        cruise_duration = (
            total_duration
            - 2.0 * ramp
        )

        local_t = float(
            np.clip(
                time_s
                - start_time_s,
                0.0,
                total_duration,
            )
        )

        # Acceleration ramp.
        if local_t < ramp:
            r = local_t / ramp

            (
                s,
                ds_dr,
                d2s_dr2,
            ) = self._smootherstep(
                r
            )

            integral_s = (
                self._integral_smootherstep(
                    r
                )
            )

            rate = vmax * s

            acceleration = (
                vmax
                * ds_dr
                / ramp
            )

            jerk = (
                vmax
                * d2s_dr2
                / (
                    ramp * ramp
                )
            )

            angle = (
                start_angle_rad
                + vmax
                * ramp
                * integral_s
            )

            return RotationKinematics(
                angle_rad=float(angle),
                rate_radps=float(rate),
                acceleration_radps2=float(
                    acceleration
                ),
                jerk_radps3=float(jerk),
            )

        angle_after_accel = (
            start_angle_rad
            + 0.5
            * vmax
            * ramp
        )

        # Constant-rate cruise.
        if local_t < (
            ramp
            + cruise_duration
        ):
            cruise_t = (
                local_t
                - ramp
            )

            angle = (
                angle_after_accel
                + vmax
                * cruise_t
            )

            return RotationKinematics(
                angle_rad=float(angle),
                rate_radps=float(vmax),
                acceleration_radps2=0.0,
                jerk_radps3=0.0,
            )

        angle_before_decel = (
            angle_after_accel
            + vmax
            * cruise_duration
        )

        # Deceleration ramp.
        decel_t = (
            local_t
            - ramp
            - cruise_duration
        )

        r = float(
            np.clip(
                decel_t / ramp,
                0.0,
                1.0,
            )
        )

        (
            s,
            ds_dr,
            d2s_dr2,
        ) = self._smootherstep(
            r
        )

        integral_s = (
            self._integral_smootherstep(
                r
            )
        )

        rate = (
            vmax
            * (1.0 - s)
        )

        acceleration = (
            -vmax
            * ds_dr
            / ramp
        )

        jerk = (
            -vmax
            * d2s_dr2
            / (
                ramp * ramp
            )
        )

        angle = (
            angle_before_decel
            + vmax
            * ramp
            * (
                r
                - integral_s
            )
        )

        if r >= 1.0 - 1e-12:
            angle = end_angle_rad
            rate = 0.0
            acceleration = 0.0
            jerk = 0.0

        return RotationKinematics(
            angle_rad=float(angle),
            rate_radps=float(rate),
            acceleration_radps2=float(
                acceleration
            ),
            jerk_radps3=float(jerk),
        )

    def rotation_kinematics(
        self,
        time_s: float,
    ) -> RotationKinematics:
        t = float(
            np.clip(
                time_s,
                self.start_time_s,
                self.end_time_s,
            )
        )

        if self._maneuver in ("cartwheel", "yaw"):
            if t < self._cartwheel_start:
                return RotationKinematics(
                    angle_rad=0.0,
                    rate_radps=0.0,
                    acceleration_radps2=0.0,
                    jerk_radps3=0.0,
                )

            if t < self._cartwheel_end:
                return self._smooth_move_kinematics(
                    t,
                    start_time_s=self._cartwheel_start,
                    start_angle_rad=0.0,
                    end_angle_rad=2.0 * np.pi,
                )

            return RotationKinematics(
                angle_rad=float(
                    2.0 * np.pi
                ),
                rate_radps=0.0,
                acceleration_radps2=0.0,
                jerk_radps3=0.0,
            )

        if t < self._entry_start:
            return RotationKinematics(
                angle_rad=0.0,
                rate_radps=0.0,
                acceleration_radps2=0.0,
                jerk_radps3=0.0,
            )

        if t < self._entry_end:
            return self._smooth_move_kinematics(
                t,
                start_time_s=self._entry_start,
                start_angle_rad=0.0,
                end_angle_rad=np.pi,
            )

        if t < self._full_cycle_start:
            return RotationKinematics(
                angle_rad=float(np.pi),
                rate_radps=0.0,
                acceleration_radps2=0.0,
                jerk_radps3=0.0,
            )

        if t < self._full_cycle_end:
            return self._smooth_move_kinematics(
                t,
                start_time_s=self._full_cycle_start,
                start_angle_rad=np.pi,
                end_angle_rad=-np.pi,
            )

        if t < self._exit_start:
            return RotationKinematics(
                angle_rad=float(-np.pi),
                rate_radps=0.0,
                acceleration_radps2=0.0,
                jerk_radps3=0.0,
            )

        if t < self._exit_end:
            return self._smooth_move_kinematics(
                t,
                start_time_s=self._exit_start,
                start_angle_rad=-np.pi,
                end_angle_rad=0.0,
            )

        return RotationKinematics(
            angle_rad=0.0,
            rate_radps=0.0,
            acceleration_radps2=0.0,
            jerk_radps3=0.0,
        )

    def rotation_state(
        self,
        time_s: float,
    ) -> tuple[float, float, float]:
        """
        Backward-compatible 3-value interface.
        """
        kinematics = (
            self.rotation_kinematics(
                time_s
            )
        )

        return (
            kinematics.angle_rad,
            kinematics.rate_radps,
            kinematics.acceleration_radps2,
        )

    def _compensation_direction_W(
        self,
    ) -> np.ndarray:
        """
        Empirical direction from the R_delta_alpha=50 experiments.

        pitch:
            + pitch rate produced negative X tool bias
            -> command positive X compensation.

        roll:
            + roll rate produced positive Y tool bias
            -> command negative Y compensation.
        """
        if self._maneuver == "pitch":
            return np.array(
                [1.0, 0.0, 0.0],
                dtype=float,
            )

        if self._maneuver == "roll":
            return np.array(
                [0.0, -1.0, 0.0],
                dtype=float,
            )

        raise RuntimeError(
            "Bias compensation direction is not "
            "defined for cartwheel."
        )

    def bias_compensation_state(
        self,
        time_s: float,
    ) -> BiasCompensationState:
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

        kinematics = (
            self.rotation_kinematics(
                time_s
            )
        )

        rate_degps = float(
            np.degrees(
                kinematics.rate_radps
            )
        )

        acceleration_degps2 = float(
            np.degrees(
                kinematics.acceleration_radps2
            )
        )

        jerk_degps3 = float(
            np.degrees(
                kinematics.jerk_radps3
            )
        )

        direction_W = (
            self._compensation_direction_W()
        )

        return BiasCompensationState(
            position_offset_W=(
                RATE_BIAS_GAIN_M_PER_DEGPS
                * rate_degps
                * direction_W
            ),
            velocity_offset_W=(
                RATE_BIAS_GAIN_M_PER_DEGPS
                * acceleration_degps2
                * direction_W
            ),
            acceleration_offset_W=(
                RATE_BIAS_GAIN_M_PER_DEGPS
                * jerk_degps3
                * direction_W
            ),
        )

    def sample(
        self,
        time_s: float,
    ) -> EndEffectorReferencePoint:
        kinematics = (
            self.rotation_kinematics(
                time_s
            )
        )

        compensation = (
            self.bias_compensation_state(
                time_s
            )
        )

        angle = kinematics.angle_rad
        angle_rate = kinematics.rate_radps
        angle_acceleration = (
            kinematics.acceleration_radps2
        )

        half = 0.5 * angle

        if self._maneuver == "pitch":
            quaternion_WE = np.array(
                [
                    np.cos(half),
                    0.0,
                    np.sin(half),
                    0.0,
                ],
                dtype=float,
            )

            omega_E = np.array(
                [
                    0.0,
                    angle_rate,
                    0.0,
                ],
                dtype=float,
            )

            omega_dot_E = np.array(
                [
                    0.0,
                    angle_acceleration,
                    0.0,
                ],
                dtype=float,
            )

        elif self._maneuver == "roll":
            quaternion_WE = np.array(
                [
                    np.cos(half),
                    np.sin(half),
                    0.0,
                    0.0,
                ],
                dtype=float,
            )

            omega_E = np.array(
                [
                    angle_rate,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )

            omega_dot_E = np.array(
                [
                    angle_acceleration,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )

        elif self._maneuver == "yaw":
            # Level full revolution about the body/tool Z axis.
            quaternion_WE = np.array(
                [
                    np.cos(half),
                    0.0,
                    0.0,
                    np.sin(half),
                ],
                dtype=float,
            )

            omega_E = np.array(
                [
                    0.0,
                    0.0,
                    angle_rate,
                ],
                dtype=float,
            )

            omega_dot_E = np.array(
                [
                    0.0,
                    0.0,
                    angle_acceleration,
                ],
                dtype=float,
            )

        else:
            # Vertical cartwheel:
            # start at +90 deg roll, then rotate about local/body Z.
            quaternion_WE = cartwheel_quaternion_WE(
                angle
            )

            omega_E = np.array(
                [
                    0.0,
                    0.0,
                    angle_rate,
                ],
                dtype=float,
            )

            omega_dot_E = np.array(
                [
                    0.0,
                    0.0,
                    angle_acceleration,
                ],
                dtype=float,
            )

        return EndEffectorReferencePoint(
            position_W=(
                self._position_W
                + compensation.position_offset_W
            ),
            velocity_W=(
                compensation.velocity_offset_W
            ),
            acceleration_W=(
                compensation.acceleration_offset_W
            ),
            quaternion_WE=quaternion_WE,
            omega_E=omega_E,
            omega_dot_E=omega_dot_E,
        )


class PaperStyleFixedEffectorPitchTrajectory(
    PaperStyleFixedEffectorTrajectory
):
    def __init__(
        self,
        *,
        position_W: np.ndarray,
        bias_compensation: bool = False,
    ) -> None:
        super().__init__(
            position_W=position_W,
            maneuver="pitch",
            bias_compensation=bias_compensation,
        )


class PaperStyleFixedEffectorRollTrajectory(
    PaperStyleFixedEffectorTrajectory
):
    def __init__(
        self,
        *,
        position_W: np.ndarray,
        bias_compensation: bool = False,
    ) -> None:
        super().__init__(
            position_W=position_W,
            maneuver="roll",
            bias_compensation=bias_compensation,
        )


class PaperStyleFixedEffectorCartwheelTrajectory(
    PaperStyleFixedEffectorTrajectory
):
    def __init__(
        self,
        *,
        position_W: np.ndarray,
    ) -> None:
        super().__init__(
            position_W=position_W,
            maneuver="cartwheel",
            bias_compensation=False,
        )



class PaperStyleStaticCartwheelHoldTrajectory:
    """
    Independent fixed-pose cartwheel diagnostic.

    The vehicle is initialized directly at one selected cartwheel
    angle and then holds that exact pose for 5 s. There is no angular
    motion and no empirical bias compensation.

    Running each angle in a separate process gives:
      - a fresh acados solver / warm start,
      - a fresh MuJoCo state,
      - a fresh stage-0 actuator reference.

    This isolates static holding capability from the actuator-reference
    branch transitions encountered during the continuous cartwheel.
    """

    VALID_ANGLES_DEG = (
        90.0,
        180.0,
        270.0,
    )

    def __init__(
        self,
        *,
        position_W: np.ndarray,
        cartwheel_angle_deg: float,
    ) -> None:
        position = np.asarray(
            position_W,
            dtype=float,
        ).reshape(-1)

        if position.shape != (3,):
            raise ValueError(
                "position_W must have shape (3,), "
                f"got {position.shape}."
            )

        angle_deg = float(
            cartwheel_angle_deg
        )

        if not any(
            np.isclose(
                angle_deg,
                valid,
                atol=1e-12,
            )
            for valid in self.VALID_ANGLES_DEG
        ):
            raise ValueError(
                "cartwheel_angle_deg must be one of "
                f"{self.VALID_ANGLES_DEG}, got {angle_deg}."
            )

        self._position_W = position.copy()
        self._cartwheel_angle_deg = angle_deg
        self._cartwheel_angle_rad = float(
            np.radians(
                angle_deg
            )
        )

    @property
    def maneuver(self) -> str:
        return "cartwheel-static"

    @property
    def bias_compensation(self) -> bool:
        return False

    @property
    def start_time_s(self) -> float:
        return 0.0

    @property
    def end_time_s(self) -> float:
        return (
            STATIC_CARTWHEEL_HOLD_DURATION_S
        )

    @property
    def steady_state_start_s(self) -> float:
        return (
            STATIC_CARTWHEEL_STEADY_STATE_START_S
        )

    @property
    def position_W(self) -> np.ndarray:
        return self._position_W.copy()

    @property
    def cartwheel_angle_deg(self) -> float:
        return self._cartwheel_angle_deg

    @property
    def max_angular_rate_degps(self) -> float:
        return 0.0

    @property
    def angular_rate_ramp_duration_s(self) -> float:
        return 0.0

    @property
    def maximum_compensation_m(self) -> float:
        return 0.0

    def schedule_lines(self) -> list[str]:
        return [
            (
                f"{0.0:5.1f}-{self.end_time_s:5.1f} s : "
                f"hold cartwheel angle "
                f"{self._cartwheel_angle_deg:.0f} deg "
                "with zero angular rate"
            ),
        ]

    def rotation_kinematics(
        self,
        time_s: float,
    ) -> RotationKinematics:
        del time_s

        return RotationKinematics(
            angle_rad=self._cartwheel_angle_rad,
            rate_radps=0.0,
            acceleration_radps2=0.0,
            jerk_radps3=0.0,
        )

    def rotation_state(
        self,
        time_s: float,
    ) -> tuple[float, float, float]:
        kinematics = (
            self.rotation_kinematics(
                time_s
            )
        )

        return (
            kinematics.angle_rad,
            kinematics.rate_radps,
            kinematics.acceleration_radps2,
        )

    def bias_compensation_state(
        self,
        time_s: float,
    ) -> BiasCompensationState:
        del time_s

        zero = np.zeros(
            3,
            dtype=float,
        )

        return BiasCompensationState(
            position_offset_W=zero.copy(),
            velocity_offset_W=zero.copy(),
            acceleration_offset_W=zero.copy(),
        )

    def sample(
        self,
        time_s: float,
    ) -> EndEffectorReferencePoint:
        del time_s

        quaternion_WE = (
            cartwheel_quaternion_WE(
                self._cartwheel_angle_rad
            )
        )

        return EndEffectorReferencePoint(
            position_W=self._position_W,
            velocity_W=np.zeros(
                3,
                dtype=float,
            ),
            acceleration_W=np.zeros(
                3,
                dtype=float,
            ),
            quaternion_WE=quaternion_WE,
            omega_E=np.zeros(
                3,
                dtype=float,
            ),
            omega_dot_E=np.zeros(
                3,
                dtype=float,
            ),
        )


# ---------------------------------------------------------------------
# Continuous combined maneuver
# ---------------------------------------------------------------------

COMBINED_HOLD_START_S = 3.0
COMBINED_SETTLE_S = 2.0
COMBINED_FINAL_HOLD_S = 3.0

_AXIS_UNIT = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _axis_quaternion(
    axis: str,
    angle_rad: float,
) -> np.ndarray:
    half = 0.5 * float(angle_rad)
    q = np.zeros(4, dtype=float)
    q[0] = np.cos(half)
    q["xyz".index(axis) + 1] = np.sin(half)
    return q


@dataclass(frozen=True)
class _CombinedSegment:
    name: str
    angle_variable: str          # "roll" | "pitch" | "tilt" | "cartwheel"
    axis: str                    # rotation axis in the tool/body frame
    start_angle_rad: float
    end_angle_rad: float
    start_time_s: float
    end_time_s: float
    profile: object              # PaperStyleFixedEffectorTrajectory providing the rate profile
    bias_direction_W: np.ndarray | None


class PaperStyleCombinedManeuverTrajectory:
    """
    One continuous fixed-end-effector reference:

        level hover
        -> roll  (validated schedule: 0 -> +180, hold, +180 -> -180 = full 360, hold, -180 -> 0)
        -> pitch (same schedule about Y)
        -> pitch 0 -> +90 deg   (tool axis body +Z -> world +X)
        -> cartwheel: spin 0 -> 360 deg about the TOOL axis (body Z)
        -> pitch +90 -> 0 deg   (back to level)
        -> level hover

    The tool point stays at ``position_W`` throughout.

    Orientation (identity initial attitude, quaternion product, body-frame right-multiplication):

        R(t) = Rx(roll) Ry(pitch) Ry(tilt) Rz(cartwheel)

    with exactly one angle moving at a time.  Every angle is a continuous scalar (the cartwheel
    ends at 2*pi: the identity *rotation* but the continuous, negated quaternion), so the reference
    never flips sign or takes a shortest-path shortcut.  With ``tilt = pi/2``:  R = Ry(pi/2) Rz(psi), whose tool axis
    R e_z = (sin(pi/2), 0, cos(pi/2)) = +X for every psi.

    Angular rates/accelerations come from the existing smooth-rate move profile (smootherstep ramps,
    constant-rate cruise): 36 deg/s for roll/pitch (validated pitch/roll rate), 12 deg/s for the
    cartwheel (validated cartwheel rate).  Optional empirical bias compensation (validated for
    pitch/roll only) is applied just during the 360 deg roll and 360 deg pitch segments.
    """

    def __init__(
        self,
        *,
        position_W: np.ndarray,
        bias_compensation: bool = False,
    ) -> None:
        position = np.asarray(position_W, dtype=float).reshape(-1)
        if position.shape != (3,):
            raise ValueError(
                f"position_W must have shape (3,), got {position.shape}."
            )

        self._position_W = position.copy()
        self._bias_compensation = bool(bias_compensation)

        fast = PaperStyleFixedEffectorTrajectory(
            position_W=position, maneuver="pitch"
        )
        slow = PaperStyleFixedEffectorTrajectory(
            position_W=position, maneuver="cartwheel"
        )

        pi = np.pi
        quarter = 0.5 * np.pi
        roll_dir = np.array([0.0, -1.0, 0.0])
        pitch_dir = np.array([1.0, 0.0, 0.0])

        # (name, angle variable, axis, start angle, end angle, rate profile,
        #  bias direction, hold after [s])
        #
        # Roll/pitch use the validated paper schedule: 0 -> +180, hold,
        # +180 -> -180 (the full 360 deg revolution), hold, -180 -> 0.
        # A single continuous 0 -> 360 sweep through the inverted attitude
        # was tried and does NOT track (attitude error > 100 deg), so the
        # validated schedule is used unchanged.
        hold = INTERMEDIATE_HOLD_DURATION_S
        plan = [
            ("roll_to_+180", "roll", "x", 0.0, pi, fast, roll_dir, hold),
            ("roll_full_360", "roll", "x", pi, -pi, fast, roll_dir, hold),
            ("roll_return_to_level", "roll", "x", -pi, 0.0, fast, roll_dir, COMBINED_SETTLE_S),
            ("pitch_to_+180", "pitch", "y", 0.0, pi, fast, pitch_dir, hold),
            ("pitch_full_360", "pitch", "y", pi, -pi, fast, pitch_dir, hold),
            ("pitch_return_to_level", "pitch", "y", -pi, 0.0, fast, pitch_dir, COMBINED_SETTLE_S),
            ("pitch_to_horizontal", "tilt", "y", 0.0, quarter, fast, None, COMBINED_SETTLE_S),
            ("cartwheel_360", "cartwheel", "z", 0.0, 2.0 * pi, slow, None, COMBINED_SETTLE_S),
            ("pitch_to_level", "tilt", "y", quarter, 0.0, fast, None, 0.0),
        ]

        segments: list[_CombinedSegment] = []
        self._phases: list[tuple[str, float, float]] = []
        t = 0.0

        def add_hold(name: str, duration: float) -> None:
            nonlocal t
            if duration <= 0.0:
                return
            self._phases.append((name, t, t + duration))
            t += duration

        add_hold("initial_hover", COMBINED_HOLD_START_S)
        for name, var, axis, a0, a1, profile, bias_dir, hold_after in plan:
            duration = profile._move_duration_s(a0, a1)
            segments.append(
                _CombinedSegment(
                    name=name,
                    angle_variable=var,
                    axis=axis,
                    start_angle_rad=a0,
                    end_angle_rad=a1,
                    start_time_s=t,
                    end_time_s=t + duration,
                    profile=profile,
                    bias_direction_W=bias_dir,
                )
            )
            self._phases.append((name, t, t + duration))
            t += duration
            add_hold(f"hold_after_{name}", hold_after)
        add_hold("final_hover", COMBINED_FINAL_HOLD_S)

        self._segments = segments
        self._final_hold_end = t

    # ----------------------------------------------------------- properties
    @property
    def maneuver(self) -> str:
        return "combined"

    @property
    def bias_compensation(self) -> bool:
        return self._bias_compensation

    @property
    def start_time_s(self) -> float:
        return 0.0

    @property
    def end_time_s(self) -> float:
        return float(self._final_hold_end)

    @property
    def position_W(self) -> np.ndarray:
        return self._position_W.copy()

    @property
    def max_angular_rate_degps(self) -> float:
        return MAX_ANGULAR_RATE_DEGPS

    @property
    def angular_rate_ramp_duration_s(self) -> float:
        return ANGULAR_RATE_RAMP_DURATION_S

    @property
    def maximum_compensation_m(self) -> float:
        if not self._bias_compensation:
            return 0.0
        return RATE_BIAS_GAIN_M_PER_DEGPS * MAX_ANGULAR_RATE_DEGPS

    @property
    def phases(self) -> list[tuple[str, float, float]]:
        """[(name, start_s, end_s)] for every hold and moving phase, in order."""
        return list(self._phases)

    def phase_name(self, time_s: float) -> str:
        for name, t0, t1 in self._phases:
            if t0 <= time_s < t1:
                return name
        return self._phases[-1][0]

    def schedule_lines(self) -> list[str]:
        lines = []
        for name, t0, t1 in self._phases:
            lines.append(f"{t0:6.1f}-{t1:6.1f} s : {name}")
        return lines

    # ----------------------------------------------------------- kinematics
    def _segment_state(
        self,
        segment: _CombinedSegment,
        t: float,
    ) -> RotationKinematics:
        if t <= segment.start_time_s:
            return RotationKinematics(segment.start_angle_rad, 0.0, 0.0, 0.0)
        if t >= segment.end_time_s:
            return RotationKinematics(segment.end_angle_rad, 0.0, 0.0, 0.0)
        return segment.profile._smooth_move_kinematics(
            t,
            start_time_s=segment.start_time_s,
            start_angle_rad=segment.start_angle_rad,
            end_angle_rad=segment.end_angle_rad,
        )

    def _angles_and_active(
        self,
        time_s: float,
    ) -> tuple[dict[str, float], _CombinedSegment | None, RotationKinematics | None]:
        t = float(np.clip(time_s, self.start_time_s, self.end_time_s))
        angles = {"roll": 0.0, "pitch": 0.0, "tilt": 0.0, "cartwheel": 0.0}
        active = None
        active_kin = None
        for seg in self._segments:
            if t < seg.start_time_s:
                # Not started yet: do not let a later segment overwrite
                # the value left by an earlier one on the same variable.
                continue

            kin = self._segment_state(seg, t)
            angles[seg.angle_variable] = kin.angle_rad
            if seg.start_time_s < t < seg.end_time_s:
                active, active_kin = seg, kin
        return angles, active, active_kin

    def rotation_kinematics(
        self,
        time_s: float,
    ) -> RotationKinematics:
        """Angle/rate of the active segment's variable (zero rate during holds)."""
        angles, active, kin = self._angles_and_active(time_s)
        if active is None:
            last = 0.0
            for seg in self._segments:
                if time_s >= seg.end_time_s:
                    last = seg.end_angle_rad
            return RotationKinematics(last, 0.0, 0.0, 0.0)
        return kin

    def bias_compensation_state(
        self,
        time_s: float,
    ) -> BiasCompensationState:
        zero = np.zeros(3, dtype=float)
        _, active, kin = self._angles_and_active(time_s)

        if (
            not self._bias_compensation
            or active is None
            or active.bias_direction_W is None
        ):
            return BiasCompensationState(zero.copy(), zero.copy(), zero.copy())

        d = active.bias_direction_W
        g = RATE_BIAS_GAIN_M_PER_DEGPS
        return BiasCompensationState(
            position_offset_W=g * np.degrees(kin.rate_radps) * d,
            velocity_offset_W=g * np.degrees(kin.acceleration_radps2) * d,
            acceleration_offset_W=g * np.degrees(kin.jerk_radps3) * d,
        )

    def orientation_quaternion_WE(
        self,
        time_s: float,
    ) -> np.ndarray:
        angles, _, _ = self._angles_and_active(time_s)
        q = _axis_quaternion("x", angles["roll"])
        q = quaternion_product_np(q, _axis_quaternion("y", angles["pitch"]))
        q = quaternion_product_np(q, _axis_quaternion("y", angles["tilt"]))
        q = quaternion_product_np(q, _axis_quaternion("z", angles["cartwheel"]))
        return q

    def sample(
        self,
        time_s: float,
    ) -> EndEffectorReferencePoint:
        _, active, kin = self._angles_and_active(time_s)
        compensation = self.bias_compensation_state(time_s)

        omega_E = np.zeros(3, dtype=float)
        omega_dot_E = np.zeros(3, dtype=float)
        if active is not None:
            axis = _AXIS_UNIT[active.axis]
            omega_E = kin.rate_radps * axis
            omega_dot_E = kin.acceleration_radps2 * axis

        return EndEffectorReferencePoint(
            position_W=self._position_W + compensation.position_offset_W,
            velocity_W=compensation.velocity_offset_W,
            acceleration_W=compensation.acceleration_offset_W,
            quaternion_WE=self.orientation_quaternion_WE(time_s),
            omega_E=omega_E,
            omega_dot_E=omega_dot_E,
        )
