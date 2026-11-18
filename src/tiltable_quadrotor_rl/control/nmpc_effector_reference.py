from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tiltable_quadrotor_rl.control.end_effector import (
    DEFAULT_END_EFFECTOR,
    RigidEndEffectorTransform,
    body_to_end_effector_kinematics,
    end_effector_motion_to_body_motion,
)
from tiltable_quadrotor_rl.control.nmpc_reference import (
    P_OMEGA,
    P_POSITION,
    P_QUATERNION,
    P_VELOCITY,
    ReferencePoint,
    StageReference,
    align_quaternion_sign,
    make_stage_reference,
    normalize_quaternion,
)


def _vec(
    value: np.ndarray,
    size: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=float,
    ).reshape(-1)

    if array.shape != (size,):
        raise ValueError(
            f"{name} must have shape ({size},), "
            f"got {array.shape}."
        )

    return array


@dataclass(frozen=True)
class EndEffectorReferencePoint:
    """
    Desired tool motion at one instant.

    Translation is expressed in world coordinates.

    quaternion_WE:
        Orientation of E in world W.

    omega_E / omega_dot_E:
        Angular velocity / acceleration expressed in E.
    """

    position_W: np.ndarray
    velocity_W: np.ndarray
    acceleration_W: np.ndarray

    quaternion_WE: np.ndarray

    omega_E: np.ndarray
    omega_dot_E: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_W",
            _vec(
                self.position_W,
                3,
                "position_W",
            ),
        )

        object.__setattr__(
            self,
            "velocity_W",
            _vec(
                self.velocity_W,
                3,
                "velocity_W",
            ),
        )

        object.__setattr__(
            self,
            "acceleration_W",
            _vec(
                self.acceleration_W,
                3,
                "acceleration_W",
            ),
        )

        object.__setattr__(
            self,
            "quaternion_WE",
            normalize_quaternion(
                self.quaternion_WE
            ),
        )

        object.__setattr__(
            self,
            "omega_E",
            _vec(
                self.omega_E,
                3,
                "omega_E",
            ),
        )

        object.__setattr__(
            self,
            "omega_dot_E",
            _vec(
                self.omega_dot_E,
                3,
                "omega_dot_E",
            ),
        )


class EndEffectorReferenceTrajectory(
    Protocol
):
    def sample(
        self,
        time_s: float,
    ) -> EndEffectorReferencePoint:
        ...


@dataclass(frozen=True)
class ConstantEndEffectorReference:
    point: EndEffectorReferencePoint

    def sample(
        self,
        time_s: float,
    ) -> EndEffectorReferencePoint:
        del time_s
        return self.point


def end_effector_reference_to_body_reference(
    reference: EndEffectorReferencePoint,
    *,
    transform: RigidEndEffectorTransform = DEFAULT_END_EFFECTOR,
) -> ReferencePoint:
    """
    Convert a desired tool trajectory point to the equivalent CoG/body
    motion.

    This body reference is used for:
      - nominal feed-forward rigid-body wrench,
      - actuator reference generation,
      - acados state/input initial guesses.

    The actual NMPC tracking cost remains end-effector-centric.
    """

    body = end_effector_motion_to_body_motion(
        position_E_W=reference.position_W,
        velocity_E_W=reference.velocity_W,
        acceleration_E_W=reference.acceleration_W,
        quaternion_WE=reference.quaternion_WE,
        omega_E=reference.omega_E,
        omega_dot_E=reference.omega_dot_E,
        transform=transform,
    )

    return ReferencePoint(
        position_W=body.position_W,
        velocity_W=body.velocity_W,
        acceleration_W=body.acceleration_W,
        quaternion_WB=body.quaternion_WB,
        omega_B=body.omega_B,
        omega_dot_B=body.omega_dot_B,
    )


def make_effector_stage_reference(
    reference: EndEffectorReferencePoint,
    *,
    previous_effector_quaternion: np.ndarray,
    previous_body_quaternion: np.ndarray,
    previous_alpha: np.ndarray,
    transform: RigidEndEffectorTransform = DEFAULT_END_EFFECTOR,
) -> StageReference:
    """
    Build one end-effector NMPC stage reference.

    The existing CoG reference generator is reused for:
      - body feed-forward wrench,
      - pseudoinverse actuator reference,
      - singularity refinement,
      - actuator-angle continuity,
      - x/u solver initialization.

    The first 13 parameter entries are then replaced by the desired
    END-EFFECTOR position, velocity, orientation and angular rate,
    because that is what the effector-centric OCP tracks.
    """

    quaternion_WE = align_quaternion_sign(
        reference.quaternion_WE,
        previous_effector_quaternion,
    )

    aligned_effector_reference = (
        EndEffectorReferencePoint(
            position_W=reference.position_W,
            velocity_W=reference.velocity_W,
            acceleration_W=reference.acceleration_W,
            quaternion_WE=quaternion_WE,
            omega_E=reference.omega_E,
            omega_dot_E=reference.omega_dot_E,
        )
    )

    body_reference = (
        end_effector_reference_to_body_reference(
            aligned_effector_reference,
            transform=transform,
        )
    )

    body_stage = make_stage_reference(
        body_reference,
        previous_quaternion=previous_body_quaternion,
        previous_alpha=previous_alpha,
    )

    parameters = (
        body_stage.parameters.copy()
    )

    parameters[P_POSITION] = (
        aligned_effector_reference.position_W
    )

    parameters[P_VELOCITY] = (
        aligned_effector_reference.velocity_W
    )

    parameters[P_QUATERNION] = (
        aligned_effector_reference.quaternion_WE
    )

    parameters[P_OMEGA] = (
        aligned_effector_reference.omega_E
    )

    return StageReference(
        parameters=parameters,
        x_reference=body_stage.x_reference,
        u_reference=body_stage.u_reference,
        wrench_reference_B=body_stage.wrench_reference_B,
    )


def build_effector_reference_horizon(
    trajectory: EndEffectorReferenceTrajectory,
    *,
    start_time_s: float,
    stage_step_s: float,
    n_horizon: int,
    initial_body_quaternion: np.ndarray,
    initial_alpha: np.ndarray,
    transform: RigidEndEffectorTransform = DEFAULT_END_EFFECTOR,
) -> list[StageReference]:
    """
    Build all N+1 end-effector references while preserving quaternion
    and actuator-reference continuity across the horizon.
    """

    initial_body_quaternion = (
        normalize_quaternion(
            initial_body_quaternion
        )
    )

    current_effector = (
        body_to_end_effector_kinematics(
            position_W=np.zeros(3),
            velocity_W=np.zeros(3),
            quaternion_WB=initial_body_quaternion,
            omega_B=np.zeros(3),
            transform=transform,
        )
    )

    previous_effector_quaternion = (
        current_effector.quaternion_WE
    )

    previous_body_quaternion = (
        initial_body_quaternion
    )

    previous_alpha = _vec(
        initial_alpha,
        4,
        "initial_alpha",
    ).copy()

    stages: list[StageReference] = []

    for stage in range(
        n_horizon + 1
    ):
        time_s = (
            start_time_s
            + stage
            * stage_step_s
        )

        reference = trajectory.sample(
            time_s
        )

        stage_reference = (
            make_effector_stage_reference(
                reference,
                previous_effector_quaternion=previous_effector_quaternion,
                previous_body_quaternion=previous_body_quaternion,
                previous_alpha=previous_alpha,
                transform=transform,
            )
        )

        stages.append(
            stage_reference
        )

        previous_effector_quaternion = (
            stage_reference.parameters[
                P_QUATERNION
            ]
        )

        previous_body_quaternion = (
            stage_reference.x_reference[
                6:10
            ]
        )

        previous_alpha = (
            stage_reference.x_reference[
                13:17
            ]
        )

    return stages
