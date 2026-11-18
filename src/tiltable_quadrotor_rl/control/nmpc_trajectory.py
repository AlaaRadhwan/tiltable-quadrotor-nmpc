from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tiltable_quadrotor_rl.control.nmpc_reference import (
    ReferencePoint,
)


@dataclass(frozen=True)
class PositionYawWaypoint:
    """Timed position + yaw waypoint."""

    time_s: float
    position_W: np.ndarray
    yaw_rad: float

    def __post_init__(self) -> None:
        position = np.asarray(
            self.position_W,
            dtype=float,
        ).reshape(-1)

        if position.shape != (3,):
            raise ValueError(
                "position_W must have shape (3,), "
                f"got {position.shape}."
            )

        object.__setattr__(self, "position_W", position)
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(self, "yaw_rad", float(self.yaw_rad))


class QuinticPositionYawTrajectory:
    """
    Smooth trajectory through timed position/yaw waypoints.

    Each segment uses:
        s(r) = 10 r^3 - 15 r^4 + 6 r^5

    This gives zero velocity and acceleration at every waypoint.
    """

    def __init__(
        self,
        waypoints: list[PositionYawWaypoint],
    ) -> None:

        if len(waypoints) < 2:
            raise ValueError(
                "At least two waypoints are required."
            )

        ordered = sorted(
            waypoints,
            key=lambda waypoint: waypoint.time_s,
        )

        times = np.array(
            [waypoint.time_s for waypoint in ordered],
            dtype=float,
        )

        if np.any(np.diff(times) <= 0.0):
            raise ValueError(
                "Waypoint times must be strictly increasing."
            )

        positions = np.stack(
            [waypoint.position_W for waypoint in ordered],
            axis=0,
        )

        # Prevent yaw interpolation from jumping across +/-pi.
        yaws = np.unwrap(
            np.array(
                [waypoint.yaw_rad for waypoint in ordered],
                dtype=float,
            )
        )

        self._times = times
        self._positions = positions
        self._yaws = yaws

    @property
    def start_time_s(self) -> float:
        return float(self._times[0])

    @property
    def end_time_s(self) -> float:
        return float(self._times[-1])

    @staticmethod
    def _quintic(
        r: float,
        duration: float,
    ) -> tuple[float, float, float]:
        """Return s, ds/dt, and d2s/dt2."""

        r2 = r * r
        r3 = r2 * r
        r4 = r3 * r
        r5 = r4 * r

        s = (
            10.0 * r3
            - 15.0 * r4
            + 6.0 * r5
        )

        ds_dr = (
            30.0 * r2
            - 60.0 * r3
            + 30.0 * r4
        )

        d2s_dr2 = (
            60.0 * r
            - 180.0 * r2
            + 120.0 * r3
        )

        s_dot = ds_dr / duration
        s_ddot = d2s_dr2 / (duration * duration)

        return s, s_dot, s_ddot

    @staticmethod
    def _yaw_quaternion(
        yaw_rad: float,
    ) -> np.ndarray:
        half_yaw = 0.5 * yaw_rad

        return np.array(
            [
                np.cos(half_yaw),
                0.0,
                0.0,
                np.sin(half_yaw),
            ],
            dtype=float,
        )

    def sample(
        self,
        time_s: float,
    ) -> ReferencePoint:
        """
        Return desired motion at time_s.

        Before/after the waypoint range, hold the endpoint.
        """

        time_s = float(time_s)

        if time_s <= self._times[0]:
            position = self._positions[0].copy()
            yaw = float(self._yaws[0])

            velocity = np.zeros(3)
            acceleration = np.zeros(3)
            yaw_rate = 0.0
            yaw_acceleration = 0.0

        elif time_s >= self._times[-1]:
            position = self._positions[-1].copy()
            yaw = float(self._yaws[-1])

            velocity = np.zeros(3)
            acceleration = np.zeros(3)
            yaw_rate = 0.0
            yaw_acceleration = 0.0

        else:
            right = int(
                np.searchsorted(
                    self._times,
                    time_s,
                    side="right",
                )
            )
            left = right - 1

            t0 = float(self._times[left])
            t1 = float(self._times[right])
            duration = t1 - t0

            r = (time_s - t0) / duration

            s, s_dot, s_ddot = self._quintic(
                r,
                duration,
            )

            delta_position = (
                self._positions[right]
                - self._positions[left]
            )

            position = (
                self._positions[left]
                + s * delta_position
            )

            velocity = (
                s_dot * delta_position
            )

            acceleration = (
                s_ddot * delta_position
            )

            delta_yaw = (
                self._yaws[right]
                - self._yaws[left]
            )

            yaw = (
                self._yaws[left]
                + s * delta_yaw
            )

            yaw_rate = (
                s_dot * delta_yaw
            )

            yaw_acceleration = (
                s_ddot * delta_yaw
            )

        quaternion = self._yaw_quaternion(
            yaw
        )

        # This trajectory keeps roll=pitch=0, so yaw rotation is
        # simply body angular velocity around body Z.
        omega_B = np.array(
            [0.0, 0.0, yaw_rate],
            dtype=float,
        )

        omega_dot_B = np.array(
            [0.0, 0.0, yaw_acceleration],
            dtype=float,
        )

        return ReferencePoint(
            position_W=position,
            velocity_W=velocity,
            acceleration_W=acceleration,
            quaternion_WB=quaternion,
            omega_B=omega_B,
            omega_dot_B=omega_dot_B,
        )
