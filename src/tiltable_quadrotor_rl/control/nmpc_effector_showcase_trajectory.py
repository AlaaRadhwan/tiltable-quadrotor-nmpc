from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tiltable_quadrotor_rl.control.end_effector import (
    normalize_quaternion_np,
    quaternion_product_np,
)
from tiltable_quadrotor_rl.control.nmpc_effector_reference import (
    EndEffectorReferencePoint,
)


@dataclass(frozen=True)
class RotationManeuver:
    """
    One full unwrapped rotation of the end-effector orientation.

    axis_E:
        Constant rotation axis expressed in the end-effector frame.

    start_time_s / end_time_s:
        Rotation interval. The maneuver starts and ends at zero
        angular velocity and angular acceleration.

    turns:
        Number of complete revolutions. Positive values follow the
        right-hand rule about axis_E.
    """

    start_time_s: float
    end_time_s: float
    axis_E: np.ndarray
    turns: float = 1.0

    def __post_init__(self) -> None:
        start = float(
            self.start_time_s
        )

        end = float(
            self.end_time_s
        )

        if end <= start:
            raise ValueError(
                "Rotation maneuver end_time_s must be greater "
                "than start_time_s."
            )

        axis = np.asarray(
            self.axis_E,
            dtype=float,
        ).reshape(-1)

        if axis.shape != (3,):
            raise ValueError(
                "axis_E must have shape (3,), "
                f"got {axis.shape}."
            )

        norm = float(
            np.linalg.norm(
                axis
            )
        )

        if norm <= 1e-12:
            raise ValueError(
                "Rotation axis norm is too small."
            )

        object.__setattr__(
            self,
            "start_time_s",
            start,
        )

        object.__setattr__(
            self,
            "end_time_s",
            end,
        )

        object.__setattr__(
            self,
            "axis_E",
            axis / norm,
        )

        object.__setattr__(
            self,
            "turns",
            float(
                self.turns
            ),
        )


class FixedEffectorRotationShowcaseTrajectory:
    """
    Keep the end-effector ORIGIN fixed in world while rotating the
    rigid vehicle through large attitudes around that fixed point.

    The end-effector ORIENTATION is not fixed because the tool is
    rigidly attached to the vehicle. Instead, we explicitly command
    full unwrapped rotations.

    Why this class exists instead of using ordinary waypoint SLERP:
        0 deg and 360 deg are the same physical orientation.
        A shortest-path quaternion interpolator therefore sees
        "0 -> 360" as zero motion.

    Here we preserve the intended unwrapped rotation angle:

        theta(t) = theta_total * s(t)

    with quintic smooth-step:

        s(r) = 10 r^3 - 15 r^4 + 6 r^5

    giving:

        omega_E     = axis_E * theta_dot
        omega_dot_E = axis_E * theta_ddot

    The quaternion itself can end at -q_start after one complete turn.
    That is intentional: q and -q are the same physical orientation,
    and preserving that sign makes the quaternion path continuous.
    """

    def __init__(
        self,
        *,
        position_W: np.ndarray,
        start_quaternion_WE: np.ndarray | None = None,
        maneuvers: list[RotationManeuver] | None = None,
        end_time_s: float | None = None,
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

        if start_quaternion_WE is None:
            start_quaternion_WE = np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )

        start_quaternion_WE = (
            normalize_quaternion_np(
                start_quaternion_WE
            )
        )

        if maneuvers is None:
            maneuvers = [
                # Front/back vertical loop: rotation about +Y_E.
                RotationManeuver(
                    start_time_s=2.0,
                    end_time_s=10.0,
                    axis_E=np.array(
                        [
                            0.0,
                            1.0,
                            0.0,
                        ]
                    ),
                ),

                # Lateral loop: rotation about +X_E.
                RotationManeuver(
                    start_time_s=12.0,
                    end_time_s=20.0,
                    axis_E=np.array(
                        [
                            1.0,
                            0.0,
                            0.0,
                        ]
                    ),
                ),

                # Spin about tool/body Z.
                RotationManeuver(
                    start_time_s=22.0,
                    end_time_s=30.0,
                    axis_E=np.array(
                        [
                            0.0,
                            0.0,
                            1.0,
                        ]
                    ),
                ),

                # 3-D diagonal tumble.
                RotationManeuver(
                    start_time_s=32.0,
                    end_time_s=40.0,
                    axis_E=np.array(
                        [
                            1.0,
                            1.0,
                            0.5,
                        ]
                    ),
                ),
            ]

        maneuvers = sorted(
            maneuvers,
            key=lambda maneuver: maneuver.start_time_s,
        )

        previous_end = -np.inf

        for maneuver in maneuvers:
            if maneuver.start_time_s < previous_end:
                raise ValueError(
                    "Rotation maneuvers must not overlap."
                )

            previous_end = (
                maneuver.end_time_s
            )

        if end_time_s is None:
            if maneuvers:
                end_time_s = (
                    maneuvers[-1].end_time_s
                    + 3.0
                )
            else:
                end_time_s = 3.0

        if maneuvers:
            if end_time_s < maneuvers[-1].end_time_s:
                raise ValueError(
                    "end_time_s cannot be earlier than the "
                    "last maneuver end time."
                )

        self._position_W = (
            position.copy()
        )

        self._start_quaternion_WE = (
            start_quaternion_WE.copy()
        )

        self._maneuvers = list(
            maneuvers
        )

        self._end_time_s = float(
            end_time_s
        )

        # Precompute the continuous quaternion that begins every
        # maneuver. A full turn multiplies q by +/-1 depending on the
        # number of turns, so we carry that sign forward rather than
        # snapping back to an equivalent quaternion representation.
        self._maneuver_start_quaternions: list[np.ndarray] = []

        current_quaternion = (
            self._start_quaternion_WE.copy()
        )

        for maneuver in self._maneuvers:
            self._maneuver_start_quaternions.append(
                current_quaternion.copy()
            )

            total_angle = (
                2.0
                * np.pi
                * maneuver.turns
            )

            full_increment = self._axis_angle_quaternion(
                maneuver.axis_E,
                total_angle,
            )

            current_quaternion = (
                quaternion_product_np(
                    current_quaternion,
                    full_increment,
                )
            )

        self._final_quaternion_WE = (
            current_quaternion.copy()
        )

    @property
    def start_time_s(self) -> float:
        return 0.0

    @property
    def end_time_s(self) -> float:
        return self._end_time_s

    @property
    def position_W(self) -> np.ndarray:
        return self._position_W.copy()

    @property
    def maneuvers(self) -> tuple[RotationManeuver, ...]:
        return tuple(
            self._maneuvers
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
    def _axis_angle_quaternion(
        axis: np.ndarray,
        angle_rad: float,
    ) -> np.ndarray:

        half_angle = (
            0.5
            * angle_rad
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

    def _completed_quaternion_before(
        self,
        time_s: float,
    ) -> np.ndarray:
        """
        Return the continuous quaternion representation corresponding
        to all maneuvers that have already finished before time_s.
        """

        current = (
            self._start_quaternion_WE
        )

        for index, maneuver in enumerate(
            self._maneuvers
        ):
            if time_s < maneuver.end_time_s:
                break

            start_q = (
                self._maneuver_start_quaternions[
                    index
                ]
            )

            total_angle = (
                2.0
                * np.pi
                * maneuver.turns
            )

            increment = (
                self._axis_angle_quaternion(
                    maneuver.axis_E,
                    total_angle,
                )
            )

            current = (
                quaternion_product_np(
                    start_q,
                    increment,
                )
            )

        return current.copy()

    def sample(
        self,
        time_s: float,
    ) -> EndEffectorReferencePoint:

        time_s = float(
            time_s
        )

        quaternion_WE = (
            self._completed_quaternion_before(
                time_s
            )
        )

        omega_E = np.zeros(
            3,
            dtype=float,
        )

        omega_dot_E = np.zeros(
            3,
            dtype=float,
        )

        for index, maneuver in enumerate(
            self._maneuvers
        ):
            if (
                maneuver.start_time_s
                <= time_s
                < maneuver.end_time_s
            ):

                duration = (
                    maneuver.end_time_s
                    - maneuver.start_time_s
                )

                r = (
                    (
                        time_s
                        - maneuver.start_time_s
                    )
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

                total_angle = (
                    2.0
                    * np.pi
                    * maneuver.turns
                )

                angle = (
                    total_angle
                    * s
                )

                angle_dot = (
                    total_angle
                    * s_dot
                )

                angle_ddot = (
                    total_angle
                    * s_ddot
                )

                increment = (
                    self._axis_angle_quaternion(
                        maneuver.axis_E,
                        angle,
                    )
                )

                quaternion_WE = (
                    quaternion_product_np(
                        self._maneuver_start_quaternions[
                            index
                        ],
                        increment,
                    )
                )

                omega_E = (
                    maneuver.axis_E
                    * angle_dot
                )

                omega_dot_E = (
                    maneuver.axis_E
                    * angle_ddot
                )

                break

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
