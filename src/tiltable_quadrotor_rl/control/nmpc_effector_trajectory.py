from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tiltable_quadrotor_rl.control.end_effector import (
    normalize_quaternion_np,
    quaternion_conjugate_np,
    quaternion_product_np,
)
from tiltable_quadrotor_rl.control.nmpc_effector_reference import (
    EndEffectorReferencePoint,
)


def quaternion_from_rpy(
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
) -> np.ndarray:
    """
    Quaternion q_WE from roll/pitch/yaw.

    Convention:
        q = [qw, qx, qy, qz]
    """

    cr = np.cos(
        0.5 * roll_rad
    )
    sr = np.sin(
        0.5 * roll_rad
    )

    cp = np.cos(
        0.5 * pitch_rad
    )
    sp = np.sin(
        0.5 * pitch_rad
    )

    cy = np.cos(
        0.5 * yaw_rad
    )
    sy = np.sin(
        0.5 * yaw_rad
    )

    return normalize_quaternion_np(
        np.array(
            [
                cr*cp*cy + sr*sp*sy,
                sr*cp*cy - cr*sp*sy,
                cr*sp*cy + sr*cp*sy,
                cr*cp*sy - sr*sp*cy,
            ],
            dtype=float,
        )
    )


@dataclass(frozen=True)
class EffectorPoseWaypoint:
    time_s: float
    position_W: np.ndarray
    quaternion_WE: np.ndarray

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

        object.__setattr__(
            self,
            "time_s",
            float(
                self.time_s
            ),
        )

        object.__setattr__(
            self,
            "position_W",
            position,
        )

        object.__setattr__(
            self,
            "quaternion_WE",
            normalize_quaternion_np(
                self.quaternion_WE
            ),
        )


class QuinticEffectorPoseTrajectory:
    """
    Smooth free-space 6-DoF end-effector trajectory.

    Translation:
        quintic smooth-step interpolation.

    Orientation:
        shortest-path quaternion interpolation over a fixed relative
        rotation axis for each segment using the same quintic progress.

    Therefore velocity, acceleration, angular velocity and angular
    acceleration all go to zero at each waypoint.
    """

    def __init__(
        self,
        waypoints: list[EffectorPoseWaypoint],
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
            [
                waypoint.time_s
                for waypoint in ordered
            ],
            dtype=float,
        )

        if np.any(
            np.diff(
                times
            ) <= 0.0
        ):
            raise ValueError(
                "Waypoint times must be strictly increasing."
            )

        quaternions = []
        previous = None

        for waypoint in ordered:
            quaternion = (
                waypoint.quaternion_WE.copy()
            )

            # q and -q are physically identical. Choose signs once so
            # every segment follows the shorter quaternion arc.
            if (
                previous is not None
                and np.dot(
                    previous,
                    quaternion,
                ) < 0.0
            ):
                quaternion = -quaternion

            quaternions.append(
                quaternion
            )

            previous = quaternion

        self._times = times

        self._positions = np.stack(
            [
                waypoint.position_W
                for waypoint in ordered
            ],
            axis=0,
        )

        self._quaternions = np.stack(
            quaternions,
            axis=0,
        )

    @property
    def start_time_s(self) -> float:
        return float(
            self._times[0]
        )

    @property
    def end_time_s(self) -> float:
        return float(
            self._times[-1]
        )

    @staticmethod
    def _quintic(
        r: float,
        duration: float,
    ) -> tuple[float, float, float]:

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

        s_dot = (
            ds_dr
            / duration
        )

        s_ddot = (
            d2s_dr2
            / (
                duration
                * duration
            )
        )

        return (
            s,
            s_dot,
            s_ddot,
        )

    @staticmethod
    def _relative_axis_angle(
        q0: np.ndarray,
        q1: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        q_rel = q0^-1 * q1

        With:
            q(s) = q0 * exp(0.5 * axis * angle * s)

        and the project convention:
            qdot = 0.5 q * [0, omega_E]

        we obtain:
            omega_E     = axis * angle * s_dot
            omega_dot_E = axis * angle * s_ddot
        """

        q_rel = quaternion_product_np(
            quaternion_conjugate_np(
                q0
            ),
            q1,
        )

        if q_rel[0] < 0.0:
            q_rel = -q_rel

        vector = q_rel[1:4]

        vector_norm = float(
            np.linalg.norm(
                vector
            )
        )

        if vector_norm <= 1e-12:
            return (
                np.array(
                    [
                        1.0,
                        0.0,
                        0.0,
                    ],
                    dtype=float,
                ),
                0.0,
            )

        angle = (
            2.0
            * np.arctan2(
                vector_norm,
                q_rel[0],
            )
        )

        axis = (
            vector
            / vector_norm
        )

        return (
            axis,
            float(
                angle
            ),
        )

    @staticmethod
    def _rotation_increment(
        axis: np.ndarray,
        angle: float,
    ) -> np.ndarray:

        half_angle = (
            0.5
            * angle
        )

        return np.array(
            [
                np.cos(
                    half_angle
                ),
                axis[0]
                * np.sin(
                    half_angle
                ),
                axis[1]
                * np.sin(
                    half_angle
                ),
                axis[2]
                * np.sin(
                    half_angle
                ),
            ],
            dtype=float,
        )

    def sample(
        self,
        time_s: float,
    ) -> EndEffectorReferencePoint:

        time_s = float(
            time_s
        )

        if time_s <= self._times[0]:
            position = (
                self._positions[0]
                .copy()
            )

            quaternion = (
                self._quaternions[0]
                .copy()
            )

            velocity = np.zeros(3)
            acceleration = np.zeros(3)
            omega_E = np.zeros(3)
            omega_dot_E = np.zeros(3)

        elif time_s >= self._times[-1]:
            position = (
                self._positions[-1]
                .copy()
            )

            quaternion = (
                self._quaternions[-1]
                .copy()
            )

            velocity = np.zeros(3)
            acceleration = np.zeros(3)
            omega_E = np.zeros(3)
            omega_dot_E = np.zeros(3)

        else:
            right = int(
                np.searchsorted(
                    self._times,
                    time_s,
                    side="right",
                )
            )

            left = (
                right - 1
            )

            t0 = float(
                self._times[left]
            )

            t1 = float(
                self._times[right]
            )

            duration = (
                t1 - t0
            )

            r = (
                (time_s - t0)
                / duration
            )

            (
                s,
                s_dot,
                s_ddot,
            ) = self._quintic(
                r,
                duration,
            )

            delta_position = (
                self._positions[right]
                - self._positions[left]
            )

            position = (
                self._positions[left]
                + s
                * delta_position
            )

            velocity = (
                s_dot
                * delta_position
            )

            acceleration = (
                s_ddot
                * delta_position
            )

            q0 = self._quaternions[left]
            q1 = self._quaternions[right]

            axis, angle = (
                self._relative_axis_angle(
                    q0,
                    q1,
                )
            )

            increment = (
                self._rotation_increment(
                    axis,
                    angle * s,
                )
            )

            quaternion = (
                quaternion_product_np(
                    q0,
                    increment,
                )
            )

            omega_E = (
                axis
                * angle
                * s_dot
            )

            omega_dot_E = (
                axis
                * angle
                * s_ddot
            )

        return EndEffectorReferencePoint(
            position_W=position,
            velocity_W=velocity,
            acceleration_W=acceleration,
            quaternion_WE=quaternion,
            omega_E=omega_E,
            omega_dot_E=omega_dot_E,
        )
