from __future__ import annotations

import numpy as np

from tiltable_quadrotor_rl.control.nmpc_effector_reference import (
    EndEffectorReferencePoint,
)


class PaperStyleFixedEffectorPitchTrajectory:
    """
    Original paper-style LINEAR pitch baseline.

    This is the trajectory we use for the clean A/B comparison with the
    stronger end-effector tracking weights.

    Timeline:
        0-2 s    : hold 0 deg
        2-7 s    : 0 -> +180 deg at +36 deg/s
        7-10 s   : hold +180 deg
        10-20 s  : +180 -> -180 deg at -36 deg/s
                    (one complete 360 deg physical rotation)
        20-23 s  : hold -180 deg
        23-28 s  : -180 -> 0 deg at +36 deg/s
        28-31 s  : hold 0 deg

    The end-effector origin remains exactly fixed in world:

        p_E^W = constant
        v_E^W = 0
        a_E^W = 0

    The orientation is generated directly from the unwrapped pitch angle
    so the +180 -> -180 segment remains a physical full revolution.
    """

    INITIAL_HOLD_END = 2.0

    ENTRY_END = 7.0
    POSITIVE_HOLD_END = 10.0

    FULL_CYCLE_END = 20.0
    NEGATIVE_HOLD_END = 23.0

    EXIT_END = 28.0
    FINAL_HOLD_END = 31.0

    def __init__(
        self,
        *,
        position_W: np.ndarray,
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

        self._position_W = position.copy()

    @property
    def start_time_s(self) -> float:
        return 0.0

    @property
    def end_time_s(self) -> float:
        return self.FINAL_HOLD_END

    @property
    def position_W(self) -> np.ndarray:
        return self._position_W.copy()

    @staticmethod
    def _linear_segment(
        time_s: float,
        *,
        start_time_s: float,
        end_time_s: float,
        start_angle_rad: float,
        end_angle_rad: float,
    ) -> tuple[float, float, float]:

        duration = (
            end_time_s
            - start_time_s
        )

        fraction = (
            time_s
            - start_time_s
        ) / duration

        angle = (
            start_angle_rad
            + fraction
            * (
                end_angle_rad
                - start_angle_rad
            )
        )

        angle_rate = (
            end_angle_rad
            - start_angle_rad
        ) / duration

        # Linear segment:
        # constant angular velocity inside the segment.
        angle_acceleration = 0.0

        return (
            float(angle),
            float(angle_rate),
            float(angle_acceleration),
        )

    def pitch_state(
        self,
        time_s: float,
    ) -> tuple[float, float, float]:

        t = float(
            np.clip(
                time_s,
                self.start_time_s,
                self.end_time_s,
            )
        )

        if t < self.INITIAL_HOLD_END:
            return (
                0.0,
                0.0,
                0.0,
            )

        if t < self.ENTRY_END:
            return self._linear_segment(
                t,
                start_time_s=self.INITIAL_HOLD_END,
                end_time_s=self.ENTRY_END,
                start_angle_rad=0.0,
                end_angle_rad=np.pi,
            )

        if t < self.POSITIVE_HOLD_END:
            return (
                np.pi,
                0.0,
                0.0,
            )

        if t < self.FULL_CYCLE_END:
            return self._linear_segment(
                t,
                start_time_s=self.POSITIVE_HOLD_END,
                end_time_s=self.FULL_CYCLE_END,
                start_angle_rad=np.pi,
                end_angle_rad=-np.pi,
            )

        if t < self.NEGATIVE_HOLD_END:
            return (
                -np.pi,
                0.0,
                0.0,
            )

        if t < self.EXIT_END:
            return self._linear_segment(
                t,
                start_time_s=self.NEGATIVE_HOLD_END,
                end_time_s=self.EXIT_END,
                start_angle_rad=-np.pi,
                end_angle_rad=0.0,
            )

        return (
            0.0,
            0.0,
            0.0,
        )

    def sample(
        self,
        time_s: float,
    ) -> EndEffectorReferencePoint:

        (
            pitch,
            pitch_rate,
            pitch_acceleration,
        ) = self.pitch_state(
            time_s
        )

        half = (
            0.5
            * pitch
        )

        quaternion_WE = np.array(
            [
                np.cos(
                    half
                ),
                0.0,
                np.sin(
                    half
                ),
                0.0,
            ],
            dtype=float,
        )

        omega_E = np.array(
            [
                0.0,
                pitch_rate,
                0.0,
            ],
            dtype=float,
        )

        omega_dot_E = np.array(
            [
                0.0,
                pitch_acceleration,
                0.0,
            ],
            dtype=float,
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
            omega_E=omega_E,
            omega_dot_E=omega_dot_E,
        )
