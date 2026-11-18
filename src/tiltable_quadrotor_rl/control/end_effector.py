from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from tiltable_quadrotor_rl.robot.parameters import (
    END_EFFECTOR_POSITION_B,
    END_EFFECTOR_QUATERNION_BE,
)


def normalize_quaternion_np(
    quaternion: np.ndarray,
) -> np.ndarray:
    quaternion = np.asarray(
        quaternion,
        dtype=float,
    ).reshape(-1)

    if quaternion.shape != (4,):
        raise ValueError(
            "Quaternion must have shape (4,), "
            f"got {quaternion.shape}."
        )

    norm = float(
        np.linalg.norm(
            quaternion
        )
    )

    if norm <= 1e-12:
        raise ValueError(
            "Quaternion norm is too small."
        )

    return quaternion / norm


@dataclass(frozen=True)
class RigidEndEffectorTransform:
    """
    Fixed rigid transform from vehicle body frame B to tool frame E.

    position_B:
        Position of the end-effector origin relative to the body/CoG,
        expressed in B.

    quaternion_BE:
        Orientation of E relative to B.
        Quaternion convention: [qw, qx, qy, qz].
    """

    position_B: np.ndarray
    quaternion_BE: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(
            self.position_B,
            dtype=float,
        ).reshape(-1)

        quaternion = np.asarray(
            self.quaternion_BE,
            dtype=float,
        ).reshape(-1)

        if position.shape != (3,):
            raise ValueError(
                "position_B must have shape (3,), "
                f"got {position.shape}."
            )

        if quaternion.shape != (4,):
            raise ValueError(
                "quaternion_BE must have shape (4,), "
                f"got {quaternion.shape}."
            )

        quaternion = normalize_quaternion_np(
            quaternion
        )

        object.__setattr__(
            self,
            "position_B",
            position.copy(),
        )

        object.__setattr__(
            self,
            "quaternion_BE",
            quaternion,
        )


DEFAULT_END_EFFECTOR = RigidEndEffectorTransform(
    position_B=END_EFFECTOR_POSITION_B,
    quaternion_BE=END_EFFECTOR_QUATERNION_BE,
)


@dataclass(frozen=True)
class EndEffectorKinematics:
    """
    End-effector kinematics derived from the rigid body state.

    position_W / velocity_W:
        Expressed in world coordinates.

    quaternion_WE:
        Orientation of tool frame E in world W.

    omega_E:
        Angular velocity expressed in E.
    """

    position_W: np.ndarray
    velocity_W: np.ndarray
    quaternion_WE: np.ndarray
    omega_E: np.ndarray


@dataclass(frozen=True)
class BodyMotion:
    """
    Body/CoG motion corresponding to a desired rigid tool motion.
    """

    position_W: np.ndarray
    velocity_W: np.ndarray
    acceleration_W: np.ndarray
    quaternion_WB: np.ndarray
    omega_B: np.ndarray
    omega_dot_B: np.ndarray


def quaternion_conjugate_np(
    quaternion: np.ndarray,
) -> np.ndarray:
    quaternion = normalize_quaternion_np(
        quaternion
    )

    return np.array(
        [
            quaternion[0],
            -quaternion[1],
            -quaternion[2],
            -quaternion[3],
        ],
        dtype=float,
    )


def quaternion_product_np(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """
    Hamilton product for two unit quaternions [w, x, y, z].
    """

    left = normalize_quaternion_np(
        left
    )

    right = normalize_quaternion_np(
        right
    )

    lw = left[0]
    lx, ly, lz = left[1:4]

    rw = right[0]
    rx, ry, rz = right[1:4]

    result = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )

    return normalize_quaternion_np(
        result
    )


def quaternion_to_rotation_np(
    quaternion: np.ndarray,
) -> np.ndarray:
    """
    Body/local-to-world rotation for quaternion [w, x, y, z].
    """

    qw, qx, qy, qz = (
        normalize_quaternion_np(
            quaternion
        )
    )

    return np.array(
        [
            [
                1.0 - 2.0 * (qy*qy + qz*qz),
                2.0 * (qx*qy - qz*qw),
                2.0 * (qx*qz + qy*qw),
            ],
            [
                2.0 * (qx*qy + qz*qw),
                1.0 - 2.0 * (qx*qx + qz*qz),
                2.0 * (qy*qz - qx*qw),
            ],
            [
                2.0 * (qx*qz - qy*qw),
                2.0 * (qy*qz + qx*qw),
                1.0 - 2.0 * (qx*qx + qy*qy),
            ],
        ],
        dtype=float,
    )


def body_to_end_effector_kinematics(
    position_W: np.ndarray,
    velocity_W: np.ndarray,
    quaternion_WB: np.ndarray,
    omega_B: np.ndarray,
    *,
    transform: RigidEndEffectorTransform = DEFAULT_END_EFFECTOR,
) -> EndEffectorKinematics:
    """
    Compute tool pose and velocity from the body state.

        p_E^W = p_B^W + R_WB r_BE^B

        v_E^W = v_B^W
                + R_WB (omega_B x r_BE^B)

        q_WE = q_WB * q_BE

        omega_E = R_BE^T omega_B
    """

    position_W = np.asarray(
        position_W,
        dtype=float,
    ).reshape(3)

    velocity_W = np.asarray(
        velocity_W,
        dtype=float,
    ).reshape(3)

    omega_B = np.asarray(
        omega_B,
        dtype=float,
    ).reshape(3)

    quaternion_WB = normalize_quaternion_np(
        quaternion_WB
    )

    R_WB = quaternion_to_rotation_np(
        quaternion_WB
    )

    R_BE = quaternion_to_rotation_np(
        transform.quaternion_BE
    )

    position_E_W = (
        position_W
        + R_WB
        @ transform.position_B
    )

    velocity_E_W = (
        velocity_W
        + R_WB
        @ np.cross(
            omega_B,
            transform.position_B,
        )
    )

    quaternion_WE = quaternion_product_np(
        quaternion_WB,
        transform.quaternion_BE,
    )

    omega_E = (
        R_BE.T
        @ omega_B
    )

    return EndEffectorKinematics(
        position_W=position_E_W,
        velocity_W=velocity_E_W,
        quaternion_WE=quaternion_WE,
        omega_E=omega_E,
    )


def end_effector_motion_to_body_motion(
    *,
    position_E_W: np.ndarray,
    velocity_E_W: np.ndarray,
    acceleration_E_W: np.ndarray,
    quaternion_WE: np.ndarray,
    omega_E: np.ndarray,
    omega_dot_E: np.ndarray,
    transform: RigidEndEffectorTransform = DEFAULT_END_EFFECTOR,
) -> BodyMotion:
    """
    Convert a desired rigid tool motion into the required CoG/body motion.

    For r = r_BE^B:

        p_B = p_E - R_WB r

        v_B = v_E - R_WB (omega_B x r)

        a_B = a_E - R_WB[
                    omega_dot_B x r
                    + omega_B x (omega_B x r)
                ]

    The acceleration correction contains both tangential and
    centripetal acceleration of the offset tool point.
    """

    position_E_W = np.asarray(
        position_E_W,
        dtype=float,
    ).reshape(3)

    velocity_E_W = np.asarray(
        velocity_E_W,
        dtype=float,
    ).reshape(3)

    acceleration_E_W = np.asarray(
        acceleration_E_W,
        dtype=float,
    ).reshape(3)

    omega_E = np.asarray(
        omega_E,
        dtype=float,
    ).reshape(3)

    omega_dot_E = np.asarray(
        omega_dot_E,
        dtype=float,
    ).reshape(3)

    quaternion_WE = normalize_quaternion_np(
        quaternion_WE
    )

    quaternion_EB = quaternion_conjugate_np(
        transform.quaternion_BE
    )

    quaternion_WB = quaternion_product_np(
        quaternion_WE,
        quaternion_EB,
    )

    R_WB = quaternion_to_rotation_np(
        quaternion_WB
    )

    R_BE = quaternion_to_rotation_np(
        transform.quaternion_BE
    )

    omega_B = (
        R_BE
        @ omega_E
    )

    omega_dot_B = (
        R_BE
        @ omega_dot_E
    )

    offset_velocity_B = np.cross(
        omega_B,
        transform.position_B,
    )

    offset_acceleration_B = (
        np.cross(
            omega_dot_B,
            transform.position_B,
        )
        + np.cross(
            omega_B,
            np.cross(
                omega_B,
                transform.position_B,
            ),
        )
    )

    position_B_W = (
        position_E_W
        - R_WB
        @ transform.position_B
    )

    velocity_B_W = (
        velocity_E_W
        - R_WB
        @ offset_velocity_B
    )

    acceleration_B_W = (
        acceleration_E_W
        - R_WB
        @ offset_acceleration_B
    )

    return BodyMotion(
        position_W=position_B_W,
        velocity_W=velocity_B_W,
        acceleration_W=acceleration_B_W,
        quaternion_WB=quaternion_WB,
        omega_B=omega_B,
        omega_dot_B=omega_dot_B,
    )


def quaternion_product_ca(
    left: ca.SX,
    right: ca.SX | ca.DM,
) -> ca.SX:
    """
    Hamilton product for CasADi symbolic quaternions.
    """

    lw = left[0]
    lv = left[1:4]

    rw = right[0]
    rv = right[1:4]

    scalar = (
        lw * rw
        - ca.dot(
            lv,
            rv,
        )
    )

    vector = (
        lw * rv
        + rw * lv
        + ca.cross(
            lv,
            rv,
        )
    )

    return ca.vertcat(
        scalar,
        vector,
    )


def quaternion_to_rotation_ca(
    quaternion: ca.SX,
) -> ca.SX:
    """
    Symbolic local-to-world rotation for q=[w,x,y,z].

    The NMPC quaternion state is already governed by the validated
    quaternion dynamics. As in the existing model, we do not insert
    a symbolic normalization operation inside every expression.
    """

    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    return ca.vertcat(
        ca.horzcat(
            1.0 - 2.0 * (qy*qy + qz*qz),
            2.0 * (qx*qy - qz*qw),
            2.0 * (qx*qz + qy*qw),
        ),
        ca.horzcat(
            2.0 * (qx*qy + qz*qw),
            1.0 - 2.0 * (qx*qx + qz*qz),
            2.0 * (qy*qz - qx*qw),
        ),
        ca.horzcat(
            2.0 * (qx*qz - qy*qw),
            2.0 * (qy*qz + qx*qw),
            1.0 - 2.0 * (qx*qx + qy*qy),
        ),
    )


def body_to_end_effector_kinematics_ca(
    position_W: ca.SX,
    velocity_W: ca.SX,
    quaternion_WB: ca.SX,
    omega_B: ca.SX,
    *,
    transform: RigidEndEffectorTransform = DEFAULT_END_EFFECTOR,
) -> tuple[ca.SX, ca.SX, ca.SX, ca.SX]:
    """
    CasADi version used directly inside the end-effector NMPC cost.
    """

    R_WB = quaternion_to_rotation_ca(
        quaternion_WB
    )

    r_BE_B = ca.DM(
        transform.position_B
    )

    q_BE = ca.DM(
        transform.quaternion_BE
    )

    R_BE = ca.DM(
        quaternion_to_rotation_np(
            transform.quaternion_BE
        )
    )

    position_E_W = (
        position_W
        + R_WB
        @ r_BE_B
    )

    velocity_E_W = (
        velocity_W
        + R_WB
        @ ca.cross(
            omega_B,
            r_BE_B,
        )
    )

    quaternion_WE = quaternion_product_ca(
        quaternion_WB,
        q_BE,
    )

    omega_E = (
        R_BE.T
        @ omega_B
    )

    return (
        position_E_W,
        velocity_E_W,
        quaternion_WE,
        omega_E,
    )
