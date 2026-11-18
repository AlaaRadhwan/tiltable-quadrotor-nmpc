from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    allocation_matrix,
    virtual_to_rotor_commands,
    wrench_to_virtual,
)
from tiltable_quadrotor_rl.robot.parameters import (
    GRAVITY,
    INERTIA_DIAG,
    MASS,
    MAX_THRUST,
    SERVO_COMMAND_LIMIT,
)


# ============================================================
# Stage-parameter layout used by acados
# ============================================================
#
# p = [
#   position_ref_W      3
#   velocity_ref_W      3
#   quaternion_ref_WB   4
#   omega_ref_B         3
#   alpha_ref           4
#   thrust_ref          4
# ]
#
# Total: 21 parameters.
#
# These are references supplied to the cost function. They are
# NOT additional dynamic states.

NP = 21

P_POSITION = slice(0, 3)
P_VELOCITY = slice(3, 6)
P_QUATERNION = slice(6, 10)
P_OMEGA = slice(10, 13)
P_ALPHA = slice(13, 17)
P_THRUST = slice(17, 21)


# For our current geometry every rotor lies in the body XY plane,
# so p_Ei,z = 0. The paper's Eq. (17) therefore reduces to its
# robustness bias f_t,epsilon = 1.5 N.
#
# The refinement is only a reference-generation heuristic. The
# NMPC remains responsible for the real actuator commands.
SINGULARITY_THRUST_THRESHOLD = 1.5

# When a reference rotor thrust is essentially zero, atan2(h, v)
# has no useful physical meaning. Keep the preceding tilt reference
# instead of allowing numerical noise to pick an arbitrary angle.
ZERO_THRUST_EPS = 1e-8


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


def normalize_quaternion(
    quaternion: np.ndarray,
) -> np.ndarray:
    quaternion = _vec(
        quaternion,
        4,
        "quaternion",
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


def align_quaternion_sign(
    quaternion: np.ndarray,
    previous_quaternion: np.ndarray,
) -> np.ndarray:
    """
    q and -q represent the same physical orientation.

    Choose the sign nearest to the previous quaternion so the
    reference sequence is continuous for initialization and logging.
    """

    quaternion = normalize_quaternion(
        quaternion
    )

    previous_quaternion = normalize_quaternion(
        previous_quaternion
    )

    if np.dot(
        quaternion,
        previous_quaternion,
    ) < 0.0:
        quaternion = -quaternion

    return quaternion


def quaternion_to_rotation(
    quaternion: np.ndarray,
) -> np.ndarray:
    """
    Body-to-world rotation matrix R_WB.

    Quaternion convention:
        q = [qw, qx, qy, qz]
    """

    qw, qx, qy, qz = (
        normalize_quaternion(
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


@dataclass(frozen=True)
class ReferencePoint:
    """
    Desired motion at one instant.

    Translation:
        position_W, velocity_W, acceleration_W are world-frame.

    Rotation:
        quaternion_WB is body orientation in world.
        omega_B and omega_dot_B are body-frame angular quantities.

    A trajectory provider can represent a hover, waypoint path,
    spline, teleoperation command, or arbitrary 6-DoF motion as
    long as it can return this structure for sample(t).
    """

    position_W: np.ndarray
    velocity_W: np.ndarray
    acceleration_W: np.ndarray

    quaternion_WB: np.ndarray

    omega_B: np.ndarray
    omega_dot_B: np.ndarray

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
            "quaternion_WB",
            normalize_quaternion(
                self.quaternion_WB
            ),
        )

        object.__setattr__(
            self,
            "omega_B",
            _vec(
                self.omega_B,
                3,
                "omega_B",
            ),
        )

        object.__setattr__(
            self,
            "omega_dot_B",
            _vec(
                self.omega_dot_B,
                3,
                "omega_dot_B",
            ),
        )


class ReferenceTrajectory(Protocol):
    """
    Minimal interface required by the NMPC.

    Different flight tasks are different implementations of
    sample(t), not different controllers.
    """

    def sample(
        self,
        time_s: float,
    ) -> ReferencePoint:
        ...


@dataclass(frozen=True)
class ConstantReference:
    """Convenience trajectory for a fixed 6-DoF setpoint."""

    point: ReferencePoint

    def sample(
        self,
        time_s: float,
    ) -> ReferencePoint:
        del time_s
        return self.point


@dataclass(frozen=True)
class StageReference:
    """
    Everything needed for one NMPC shooting node.

    parameters:
        21 stagewise parameters used by the nonlinear cost.

    x_reference:
        17-state reference, used to initialize the solver.

    u_reference:
        8-input reference, also used to initialize the solver.

    wrench_reference_B:
        Feed-forward body wrench, useful for logging/debugging.
    """

    parameters: np.ndarray
    x_reference: np.ndarray
    u_reference: np.ndarray
    wrench_reference_B: np.ndarray


def feedforward_wrench_body(
    reference: ReferencePoint,
) -> np.ndarray:
    """
    Compute the nominal body wrench required by the desired motion.

    Translation:
        m*a_W = R_WB*F_B + m*g_W

    therefore:
        F_B = R_WB.T * m * (a_W - g_W)

    Rotation:
        I*omega_dot = tau - omega x (I*omega)

    therefore:
        tau = I*omega_dot + omega x (I*omega)
    """

    R_WB = quaternion_to_rotation(
        reference.quaternion_WB
    )

    gravity_W = np.array(
        [
            0.0,
            0.0,
            -GRAVITY,
        ],
        dtype=float,
    )

    force_W = (
        MASS
        * (
            reference.acceleration_W
            - gravity_W
        )
    )

    force_B = (
        R_WB.T
        @ force_W
    )

    inertia = np.asarray(
        INERTIA_DIAG,
        dtype=float,
    )

    angular_momentum_B = (
        inertia
        * reference.omega_B
    )

    torque_B = (
        inertia
        * reference.omega_dot_B
        + np.cross(
            reference.omega_B,
            angular_momentum_B,
        )
    )

    return np.concatenate(
        [
            force_B,
            torque_B,
        ]
    )


def _paper_reference_refinement(
    wrench_body: np.ndarray,
    virtual: np.ndarray,
) -> np.ndarray:
    """
    Apply the paper's inexpensive low-thrust singularity refinement.

    First obtain the pseudoinverse virtual force z. If one or more
    rotor thrusts lie below the threshold, choose the largest one
    among those low-thrust rotors, modify its angle according to
    Eq. (16), keep its thrust magnitude fixed, then solve exactly
    for the remaining three rotors.

    This refinement only changes the REFERENCE supplied to NMPC.
    It is not a dynamic allocator and it does not bypass the NMPC.
    """

    wrench_body = _vec(
        wrench_body,
        6,
        "wrench_body",
    )

    virtual = _vec(
        virtual,
        8,
        "virtual",
    )

    thrust, _ = (
        virtual_to_rotor_commands(
            virtual
        )
    )

    thrust = np.asarray(
        thrust,
        dtype=float,
    ).reshape(4)

    low = np.flatnonzero(
        thrust
        < SINGULARITY_THRUST_THRESHOLD
    )

    if low.size == 0:
        return virtual.copy()

    # Paper: if two thrusts are below the threshold, select the
    # larger-thrust rotor. Extending the same rule naturally to
    # more than two candidates means choosing max(thrust) in low.
    rotor_index = int(
        low[
            np.argmax(
                thrust[low]
            )
        ]
    )

    rotor_thrust = float(
        thrust[rotor_index]
    )

    h_index = 2 * rotor_index
    v_index = h_index + 1

    h_original = float(
        virtual[h_index]
    )

    if rotor_thrust <= ZERO_THRUST_EPS:
        alpha_fixed = 0.0
    else:
        ratio = np.clip(
            h_original
            / SINGULARITY_THRUST_THRESHOLD,
            -1.0,
            1.0,
        )

        alpha_fixed = (
            0.5 * np.pi
            - np.arccos(
                ratio
            )
        )

    fixed_virtual = np.array(
        [
            rotor_thrust
            * np.sin(
                alpha_fixed
            ),
            rotor_thrust
            * np.cos(
                alpha_fixed
            ),
        ],
        dtype=float,
    )

    A = np.asarray(
        allocation_matrix(),
        dtype=float,
    )

    fixed_columns = np.array(
        [
            h_index,
            v_index,
        ],
        dtype=int,
    )

    free_columns = np.array(
        [
            column
            for column in range(8)
            if column not in fixed_columns
        ],
        dtype=int,
    )

    A_free = A[:, free_columns]

    residual_wrench = (
        wrench_body
        - A[:, fixed_columns]
        @ fixed_virtual
    )

    # A_free is 6x6. If the geometry makes this particular reduced
    # system poorly conditioned, keep the original pseudoinverse
    # reference rather than injecting a numerically bad reference.
    condition_number = float(
        np.linalg.cond(
            A_free
        )
    )

    if (
        not np.isfinite(
            condition_number
        )
        or condition_number > 1e10
    ):
        return virtual.copy()

    try:
        free_virtual = np.linalg.solve(
            A_free,
            residual_wrench,
        )
    except np.linalg.LinAlgError:
        return virtual.copy()

    refined = virtual.copy()

    refined[fixed_columns] = (
        fixed_virtual
    )

    refined[free_columns] = (
        free_virtual
    )

    # Numerical consistency check. This should be near machine
    # precision for a well-conditioned reduced solve.
    residual = (
        A @ refined
        - wrench_body
    )

    if np.linalg.norm(
        residual
    ) > 1e-7:
        return virtual.copy()

    return refined


def actuator_reference_from_wrench(
    wrench_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate coarse physical actuator references.

    Pipeline:
        desired wrench
            -> Moore-Penrose virtual allocation
            -> low-thrust singularity refinement
            -> physical thrust/tilt references

    These are SOFT NMPC references, not commands.
    """

    wrench_body = _vec(
        wrench_body,
        6,
        "wrench_body",
    )

    virtual = np.asarray(
        wrench_to_virtual(
            wrench_body
        ),
        dtype=float,
    ).reshape(8)

    virtual = (
        _paper_reference_refinement(
            wrench_body,
            virtual,
        )
    )

    thrust_ref, alpha_ref = (
        virtual_to_rotor_commands(
            virtual
        )
    )

    thrust_ref = np.asarray(
        thrust_ref,
        dtype=float,
    ).reshape(4)

    alpha_ref = np.asarray(
        alpha_ref,
        dtype=float,
    ).reshape(4)

    # The controller has hard physical bounds. Keeping its soft
    # references inside the same admissible set avoids asking the
    # cost function to chase an impossible actuator reference.
    thrust_ref = np.clip(
        thrust_ref,
        0.0,
        MAX_THRUST,
    )

    alpha_ref = np.clip(
        alpha_ref,
        -SERVO_COMMAND_LIMIT,
        SERVO_COMMAND_LIMIT,
    )

    return (
        thrust_ref,
        alpha_ref,
    )


def _nearest_equivalent_angle(
    raw_angle: float,
    previous_angle: float,
) -> float:
    """
    atan2 returns a principal angle near [-pi, pi].

    Near the +/-pi branch cut, choose the equivalent 2*pi-shifted
    angle closest to the previous reference, provided it remains
    within the NMPC servo-angle bounds.
    """

    candidates = []

    for turns in (-1, 0, 1):
        candidate = (
            raw_angle
            + turns
            * 2.0
            * np.pi
        )

        if (
            -SERVO_COMMAND_LIMIT
            <= candidate
            <= SERVO_COMMAND_LIMIT
        ):
            candidates.append(
                candidate
            )

    if not candidates:
        return float(
            np.clip(
                raw_angle,
                -SERVO_COMMAND_LIMIT,
                SERVO_COMMAND_LIMIT,
            )
        )

    return float(
        min(
            candidates,
            key=lambda value: abs(
                value
                - previous_angle
            ),
        )
    )


def make_stage_reference(
    reference: ReferencePoint,
    *,
    previous_quaternion: np.ndarray,
    previous_alpha: np.ndarray,
) -> StageReference:
    """
    Convert one trajectory point into one NMPC stage reference.

    Continuity is enforced for quaternion sign and tilt-angle branch.
    """

    previous_alpha = _vec(
        previous_alpha,
        4,
        "previous_alpha",
    )

    quaternion_ref = (
        align_quaternion_sign(
            reference.quaternion_WB,
            previous_quaternion,
        )
    )

    aligned_reference = ReferencePoint(
        position_W=reference.position_W,
        velocity_W=reference.velocity_W,
        acceleration_W=reference.acceleration_W,
        quaternion_WB=quaternion_ref,
        omega_B=reference.omega_B,
        omega_dot_B=reference.omega_dot_B,
    )

    wrench_ref = (
        feedforward_wrench_body(
            aligned_reference
        )
    )

    thrust_ref, raw_alpha_ref = (
        actuator_reference_from_wrench(
            wrench_ref
        )
    )

    alpha_ref = np.empty(
        4,
        dtype=float,
    )

    for rotor in range(4):
        if thrust_ref[rotor] <= ZERO_THRUST_EPS:
            alpha_ref[rotor] = (
                previous_alpha[rotor]
            )
        else:
            alpha_ref[rotor] = (
                _nearest_equivalent_angle(
                    float(
                        raw_alpha_ref[rotor]
                    ),
                    float(
                        previous_alpha[rotor]
                    ),
                )
            )

    parameters = np.zeros(
        NP,
        dtype=float,
    )

    parameters[P_POSITION] = (
        aligned_reference.position_W
    )

    parameters[P_VELOCITY] = (
        aligned_reference.velocity_W
    )

    parameters[P_QUATERNION] = (
        quaternion_ref
    )

    parameters[P_OMEGA] = (
        aligned_reference.omega_B
    )

    parameters[P_ALPHA] = (
        alpha_ref
    )

    parameters[P_THRUST] = (
        thrust_ref
    )

    x_reference = np.concatenate(
        [
            aligned_reference.position_W,
            aligned_reference.velocity_W,
            quaternion_ref,
            aligned_reference.omega_B,
            alpha_ref,
        ]
    )

    # At the reference, the desired servo command is simply the
    # desired servo angle. The NMPC is free to command something
    # else when transient behavior makes that beneficial.
    u_reference = np.concatenate(
        [
            thrust_ref,
            alpha_ref,
        ]
    )

    return StageReference(
        parameters=parameters,
        x_reference=x_reference,
        u_reference=u_reference,
        wrench_reference_B=wrench_ref,
    )


def build_reference_horizon(
    trajectory: ReferenceTrajectory,
    *,
    start_time_s: float,
    stage_step_s: float,
    n_horizon: int,
    initial_quaternion: np.ndarray,
    initial_alpha: np.ndarray,
) -> list[StageReference]:
    """
    Build all N+1 references together.

    Doing this sequentially lets us preserve quaternion-sign and
    servo-angle continuity across the complete prediction horizon.
    """

    previous_quaternion = (
        normalize_quaternion(
            initial_quaternion
        )
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

        reference = (
            trajectory.sample(
                time_s
            )
        )

        stage_reference = (
            make_stage_reference(
                reference,
                previous_quaternion=previous_quaternion,
                previous_alpha=previous_alpha,
            )
        )

        stages.append(
            stage_reference
        )

        previous_quaternion = (
            stage_reference.x_reference[6:10]
        )

        previous_alpha = (
            stage_reference.x_reference[13:17]
        )

    return stages
