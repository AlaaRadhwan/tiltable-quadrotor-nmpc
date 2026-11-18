from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tiltable_quadrotor_rl.control.allocation import (
    allocation_matrix,
    virtual_to_rotor_commands,
)
from tiltable_quadrotor_rl.control.nmpc_effector_reference import (
    EndEffectorReferenceTrajectory,
    build_effector_reference_horizon,
)
from tiltable_quadrotor_rl.control.nmpc_reference import (
    P_ALPHA,
    P_THRUST,
    StageReference,
)
from tiltable_quadrotor_rl.control.end_effector import (
    DEFAULT_END_EFFECTOR,
    RigidEndEffectorTransform,
)
from tiltable_quadrotor_rl.robot.parameters import (
    MAX_THRUST,
    SERVO_COMMAND_LIMIT,
)


# ---------------------------------------------------------------------
# Conservative predictive-preposition settings
# ---------------------------------------------------------------------
#
# A rotor may only be pre-positioned while its BASELINE reference
# thrust is at or below this value.
PREPOSITION_UNLOADED_THRUST_N = 1.5

# A future rotor state is considered reliably re-engaged only when its
# baseline reference thrust is at least this value for several
# consecutive shooting nodes.
PREPOSITION_REENGAGE_THRUST_N = 2.5

# At the current 0.1 s shooting interval this means 0.3 s of confirmed
# future loading before we trust the future tilt angle.
PREPOSITION_CONFIRM_STAGES = 3

# Do not create preview motion for tiny angle changes.
PREPOSITION_MIN_ANGLE_MOVE_DEG = 5.0

# After exact wrench-preserving reallocation, any rotor whose angle was
# deliberately fixed for pre-positioning must remain lightly loaded.
# Otherwise the "move it while unloaded" assumption is no longer true.
PREPOSITION_MAX_FIXED_THRUST_N = 2.5

# Numerical safety limits for the exact reduced solves.
PREPOSITION_CONDITION_LIMIT = 1e8
PREPOSITION_WRENCH_RESIDUAL_TOL = 1e-7

# Current allocation has a two-dimensional null space. We support up to
# two simultaneously pre-positioned low-thrust rotors:
#
#   one fixed rotor:
#       keep its baseline thrust magnitude + choose its angle,
#       solve the remaining 3 rotors (6 virtual components)
#
#   two fixed rotors:
#       choose both angles,
#       solve their two thrust magnitudes + the remaining 2 rotors
#       (2 + 4 = 6 unknowns)
MAX_SIMULTANEOUS_PREPOSITION_ROTORS = 2


@dataclass(frozen=True)
class PrepositionEvent:
    rotor: int
    low_start_stage: int
    reengage_stage: int
    start_alpha_rad: float
    target_alpha_rad: float

    @property
    def angle_move_rad(self) -> float:
        return (
            self.target_alpha_rad
            - self.start_alpha_rad
        )


@dataclass(frozen=True)
class PredictivePrepositionDiagnostics:
    events: tuple[PrepositionEvent, ...]

    attempted_stage_count: int
    applied_stage_count: int
    fallback_stage_count: int

    applied_stage_count_by_rotor: tuple[int, int, int, int]
    fallback_stage_count_by_rotor: tuple[int, int, int, int]

    first_modified_stage_by_rotor: tuple[
        int | None,
        int | None,
        int | None,
        int | None,
    ]

    first_modified_alpha_rad_by_rotor: tuple[
        float | None,
        float | None,
        float | None,
        float | None,
    ]

    maximum_wrench_residual: float

    def first_event_for_rotor(
        self,
        rotor: int,
    ) -> PrepositionEvent | None:
        candidates = [
            event
            for event in self.events
            if event.rotor == rotor
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda event: (
                event.low_start_stage,
                event.reengage_stage,
            ),
        )


def _nearest_equivalent_angle(
    raw_angle: float,
    previous_angle: float,
) -> float:
    """
    Pick the mechanically admissible 2*pi-equivalent angle closest to
    previous_angle.
    """
    candidates: list[float] = []

    for turns in (-1, 0, 1):
        candidate = (
            float(raw_angle)
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


def _smootherstep(
    r: float,
) -> float:
    r = float(
        np.clip(
            r,
            0.0,
            1.0,
        )
    )

    return float(
        10.0 * r**3
        - 15.0 * r**4
        + 6.0 * r**5
    )


def _stage_with_actuator_reference(
    stage: StageReference,
    *,
    thrust_ref: np.ndarray,
    alpha_ref: np.ndarray,
) -> StageReference:
    thrust_ref = np.asarray(
        thrust_ref,
        dtype=float,
    ).reshape(4)

    alpha_ref = np.asarray(
        alpha_ref,
        dtype=float,
    ).reshape(4)

    parameters = (
        stage.parameters.copy()
    )
    parameters[P_THRUST] = (
        thrust_ref
    )
    parameters[P_ALPHA] = (
        alpha_ref
    )

    x_reference = (
        stage.x_reference.copy()
    )
    x_reference[13:17] = (
        alpha_ref
    )

    u_reference = (
        stage.u_reference.copy()
    )
    u_reference[0:4] = (
        thrust_ref
    )
    u_reference[4:8] = (
        alpha_ref
    )

    return StageReference(
        parameters=parameters,
        x_reference=x_reference,
        u_reference=u_reference,
        wrench_reference_B=(
            stage.wrench_reference_B.copy()
        ),
    )


def _confirmed_reengagement_stage(
    thrust_series: np.ndarray,
    *,
    search_start: int,
) -> int | None:
    """
    Find the first future stage where the rotor remains above the
    re-engagement threshold for PREPOSITION_CONFIRM_STAGES nodes.
    """
    n_stage = int(
        thrust_series.shape[0]
    )

    last_start = (
        n_stage
        - PREPOSITION_CONFIRM_STAGES
    )

    for stage in range(
        search_start,
        last_start + 1,
    ):
        window = thrust_series[
            stage:
            stage + PREPOSITION_CONFIRM_STAGES
        ]

        if np.all(
            window
            >= PREPOSITION_REENGAGE_THRUST_N
        ):
            return stage

    return None


def _find_events(
    stages: list[StageReference],
    *,
    initial_alpha: np.ndarray,
) -> list[PrepositionEvent]:
    """
    Find low-thrust intervals that have a trustworthy re-engagement
    angle inside the current prediction horizon.
    """
    initial_alpha = np.asarray(
        initial_alpha,
        dtype=float,
    ).reshape(4)

    thrust = np.stack(
        [
            stage.parameters[
                P_THRUST
            ]
            for stage in stages
        ],
        axis=0,
    )

    alpha = np.stack(
        [
            stage.parameters[
                P_ALPHA
            ]
            for stage in stages
        ],
        axis=0,
    )

    events: list[
        PrepositionEvent
    ] = []

    n_stage = len(
        stages
    )

    for rotor in range(4):
        stage = 0

        while stage < (
            n_stage - 1
        ):
            if (
                thrust[
                    stage,
                    rotor,
                ]
                > PREPOSITION_UNLOADED_THRUST_N
            ):
                stage += 1
                continue

            low_start = stage

            reengage = (
                _confirmed_reengagement_stage(
                    thrust[:, rotor],
                    search_start=(
                        low_start + 1
                    ),
                )
            )

            if reengage is None:
                break

            # We need at least one interior prediction node to create a
            # meaningful smooth preview profile.
            if (
                reengage
                - low_start
                < 2
            ):
                stage = (
                    reengage + 1
                )
                continue

            if low_start == 0:
                start_alpha = float(
                    initial_alpha[
                        rotor
                    ]
                )
            else:
                start_alpha = float(
                    alpha[
                        low_start - 1,
                        rotor,
                    ]
                )

            target_alpha = (
                _nearest_equivalent_angle(
                    float(
                        alpha[
                            reengage,
                            rotor,
                        ]
                    ),
                    start_alpha,
                )
            )

            if abs(
                np.degrees(
                    target_alpha
                    - start_alpha
                )
            ) >= (
                PREPOSITION_MIN_ANGLE_MOVE_DEG
            ):
                events.append(
                    PrepositionEvent(
                        rotor=rotor,
                        low_start_stage=(
                            low_start
                        ),
                        reengage_stage=(
                            reengage
                        ),
                        start_alpha_rad=(
                            start_alpha
                        ),
                        target_alpha_rad=(
                            target_alpha
                        ),
                    )
                )

            stage = (
                reengage + 1
            )

    return events


def _desired_fixed_angles_by_stage(
    events: list[PrepositionEvent],
    *,
    n_stage: int,
) -> list[
    dict[int, float]
]:
    desired: list[
        dict[int, float]
    ] = [
        {}
        for _ in range(
            n_stage
        )
    ]

    for event in events:
        duration = (
            event.reengage_stage
            - event.low_start_stage
        )

        # Leave the first low-thrust node unchanged. The preview
        # starts in future shooting nodes, so stage 0 never receives an
        # artificial instantaneous angle-reference jump.
        for stage in range(
            event.low_start_stage + 1,
            event.reengage_stage,
        ):
            progress = (
                stage
                - event.low_start_stage
            ) / duration

            blend = (
                _smootherstep(
                    progress
                )
            )

            angle = (
                event.start_alpha_rad
                + blend
                * event.angle_move_rad
            )

            desired[
                stage
            ][
                event.rotor
            ] = float(
                angle
            )

    return desired


def _virtual_from_physical(
    thrust: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    thrust = np.asarray(
        thrust,
        dtype=float,
    ).reshape(4)

    alpha = np.asarray(
        alpha,
        dtype=float,
    ).reshape(4)

    virtual = np.empty(
        8,
        dtype=float,
    )

    virtual[0::2] = (
        thrust
        * np.sin(
            alpha
        )
    )

    virtual[1::2] = (
        thrust
        * np.cos(
            alpha
        )
    )

    return virtual


def _solve_exact_with_fixed_angles(
    stage: StageReference,
    *,
    fixed_angles: dict[int, float],
    previous_alpha: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
] | None:
    """
    Preserve stage.wrench_reference_B exactly while pre-positioning one
    or two lightly loaded rotors.

    Returns
    -------
    (thrust_ref, alpha_ref, wrench_residual_norm)
        on success.

    None
        if the exact reduced allocation is ill-conditioned, violates
        actuator limits, requires a pre-positioned rotor to carry too
        much thrust, or otherwise fails a safety check.
    """
    if not fixed_angles:
        return None

    fixed_rotors = sorted(
        int(rotor)
        for rotor in fixed_angles
    )

    if len(
        fixed_rotors
    ) > (
        MAX_SIMULTANEOUS_PREPOSITION_ROTORS
    ):
        return None

    previous_alpha = np.asarray(
        previous_alpha,
        dtype=float,
    ).reshape(4)

    baseline_thrust = np.asarray(
        stage.parameters[
            P_THRUST
        ],
        dtype=float,
    ).reshape(4)

    baseline_alpha = np.asarray(
        stage.parameters[
            P_ALPHA
        ],
        dtype=float,
    ).reshape(4)

    wrench = np.asarray(
        stage.wrench_reference_B,
        dtype=float,
    ).reshape(6)

    A = np.asarray(
        allocation_matrix(),
        dtype=float,
    )

    # -------------------------------------------------------------
    # One fixed rotor:
    #   angle chosen by preview
    #   thrust magnitude kept at its baseline low-thrust value
    #   remaining 3 rotors -> 6 free virtual components
    # -------------------------------------------------------------
    if len(
        fixed_rotors
    ) == 1:
        rotor = (
            fixed_rotors[0]
        )

        fixed_thrust = float(
            baseline_thrust[
                rotor
            ]
        )

        if (
            fixed_thrust
            > PREPOSITION_MAX_FIXED_THRUST_N
        ):
            return None

        fixed_angle = float(
            fixed_angles[
                rotor
            ]
        )

        h_index = (
            2 * rotor
        )
        v_index = (
            h_index + 1
        )

        fixed_columns = np.array(
            [
                h_index,
                v_index,
            ],
            dtype=int,
        )

        fixed_virtual = np.array(
            [
                fixed_thrust
                * np.sin(
                    fixed_angle
                ),
                fixed_thrust
                * np.cos(
                    fixed_angle
                ),
            ],
            dtype=float,
        )

        free_columns = np.array(
            [
                column
                for column in range(8)
                if column
                not in fixed_columns
            ],
            dtype=int,
        )

        A_free = (
            A[
                :,
                free_columns,
            ]
        )

        condition = float(
            np.linalg.cond(
                A_free
            )
        )

        if (
            not np.isfinite(
                condition
            )
            or condition
            > PREPOSITION_CONDITION_LIMIT
        ):
            return None

        residual_wrench = (
            wrench
            - A[
                :,
                fixed_columns,
            ]
            @ fixed_virtual
        )

        try:
            free_virtual = (
                np.linalg.solve(
                    A_free,
                    residual_wrench,
                )
            )
        except np.linalg.LinAlgError:
            return None

        virtual = np.empty(
            8,
            dtype=float,
        )

        virtual[
            fixed_columns
        ] = fixed_virtual

        virtual[
            free_columns
        ] = free_virtual

    # -------------------------------------------------------------
    # Two fixed rotors:
    #
    # Their ANGLES are fixed, but their thrust magnitudes remain
    # unknown. Together with both virtual components of the two free
    # rotors:
    #
    #   2 fixed thrust magnitudes
    #   + 4 free virtual components
    #   = 6 unknowns
    #
    # so the 6-D wrench can still be solved exactly.
    # -------------------------------------------------------------
    else:
        free_rotors = [
            rotor
            for rotor in range(4)
            if rotor
            not in fixed_rotors
        ]

        columns: list[
            np.ndarray
        ] = []

        # First unknowns are the thrust magnitudes of the fixed-angle
        # rotors.
        for rotor in fixed_rotors:
            angle = float(
                fixed_angles[
                    rotor
                ]
            )

            direction_virtual = (
                np.zeros(
                    8,
                    dtype=float,
                )
            )

            direction_virtual[
                2 * rotor
            ] = np.sin(
                angle
            )

            direction_virtual[
                2 * rotor + 1
            ] = np.cos(
                angle
            )

            columns.append(
                A
                @ direction_virtual
            )

        # Remaining unknowns are the h/v pairs of the two free rotors.
        for rotor in free_rotors:
            columns.append(
                A[
                    :,
                    2 * rotor,
                ]
            )
            columns.append(
                A[
                    :,
                    2 * rotor + 1,
                ]
            )

        reduced = np.column_stack(
            columns
        )

        condition = float(
            np.linalg.cond(
                reduced
            )
        )

        if (
            not np.isfinite(
                condition
            )
            or condition
            > PREPOSITION_CONDITION_LIMIT
        ):
            return None

        try:
            solution = (
                np.linalg.solve(
                    reduced,
                    wrench,
                )
            )
        except np.linalg.LinAlgError:
            return None

        fixed_thrust = (
            solution[
                0:2
            ]
        )

        if np.any(
            fixed_thrust
            < -1e-9
        ):
            return None

        if np.any(
            fixed_thrust
            > PREPOSITION_MAX_FIXED_THRUST_N
        ):
            return None

        fixed_thrust = np.maximum(
            fixed_thrust,
            0.0,
        )

        virtual = np.zeros(
            8,
            dtype=float,
        )

        for local_index, rotor in enumerate(
            fixed_rotors
        ):
            angle = float(
                fixed_angles[
                    rotor
                ]
            )

            thrust = float(
                fixed_thrust[
                    local_index
                ]
            )

            virtual[
                2 * rotor
            ] = (
                thrust
                * np.sin(
                    angle
                )
            )

            virtual[
                2 * rotor + 1
            ] = (
                thrust
                * np.cos(
                    angle
                )
            )

        offset = 2

        for rotor in free_rotors:
            virtual[
                2 * rotor
            ] = solution[
                offset
            ]
            virtual[
                2 * rotor + 1
            ] = solution[
                offset + 1
            ]

            offset += 2

    wrench_residual = float(
        np.linalg.norm(
            A
            @ virtual
            - wrench
        )
    )

    if (
        not np.isfinite(
            wrench_residual
        )
        or wrench_residual
        > PREPOSITION_WRENCH_RESIDUAL_TOL
    ):
        return None

    thrust_ref, raw_alpha_ref = (
        virtual_to_rotor_commands(
            virtual
        )
    )

    thrust_ref = np.asarray(
        thrust_ref,
        dtype=float,
    ).reshape(4)

    raw_alpha_ref = np.asarray(
        raw_alpha_ref,
        dtype=float,
    ).reshape(4)

    if not np.all(
        np.isfinite(
            thrust_ref
        )
    ):
        return None

    if np.any(
        thrust_ref
        > MAX_THRUST + 1e-9
    ):
        return None

    alpha_ref = np.empty(
        4,
        dtype=float,
    )

    for rotor in range(4):
        if rotor in fixed_angles:
            alpha_ref[
                rotor
            ] = float(
                fixed_angles[
                    rotor
                ]
            )

        elif thrust_ref[
            rotor
        ] <= 1e-8:
            # An unloaded free rotor's direction is meaningless.
            alpha_ref[
                rotor
            ] = float(
                previous_alpha[
                    rotor
                ]
            )

        else:
            alpha_ref[
                rotor
            ] = (
                _nearest_equivalent_angle(
                    float(
                        raw_alpha_ref[
                            rotor
                        ]
                    ),
                    float(
                        previous_alpha[
                            rotor
                        ]
                    ),
                )
            )

    if np.any(
        np.abs(
            alpha_ref
        )
        > SERVO_COMMAND_LIMIT + 1e-12
    ):
        return None

    # Recheck exactness from the PHYSICAL commands after angle branch
    # selection. 2*pi-equivalent branch changes must not alter wrench.
    physical_virtual = (
        _virtual_from_physical(
            thrust_ref,
            alpha_ref,
        )
    )

    physical_residual = float(
        np.linalg.norm(
            A
            @ physical_virtual
            - wrench
        )
    )

    if (
        not np.isfinite(
            physical_residual
        )
        or physical_residual
        > PREPOSITION_WRENCH_RESIDUAL_TOL
    ):
        return None

    return (
        thrust_ref,
        alpha_ref,
        physical_residual,
    )


def apply_predictive_low_thrust_preposition(
    stages: list[StageReference],
    *,
    initial_alpha: np.ndarray,
) -> tuple[
    list[StageReference],
    PredictivePrepositionDiagnostics,
]:
    """
    Apply trajectory-previewed low-thrust rotor pre-positioning.

    Important safety behavior
    -------------------------
    * The existing stage references are built first.
    * Preview only acts inside a low-thrust interval that has a
      confirmed future re-engagement inside the horizon.
    * At most two low-thrust rotors are pre-positioned at once.
    * The desired 6-D wrench must remain exact.
    * Rotor thrust and servo-angle limits are checked.
    * Ill-conditioned or infeasible stages fall back to the baseline
      reference with no exception and no partial command.
    """
    if len(
        stages
    ) < 2:
        diagnostics = (
            PredictivePrepositionDiagnostics(
                events=(),
                attempted_stage_count=0,
                applied_stage_count=0,
                fallback_stage_count=0,
                applied_stage_count_by_rotor=(
                    0,
                    0,
                    0,
                    0,
                ),
                fallback_stage_count_by_rotor=(
                    0,
                    0,
                    0,
                    0,
                ),
                first_modified_stage_by_rotor=(
                    None,
                    None,
                    None,
                    None,
                ),
                first_modified_alpha_rad_by_rotor=(
                    None,
                    None,
                    None,
                    None,
                ),
                maximum_wrench_residual=0.0,
            )
        )

        return (
            list(stages),
            diagnostics,
        )

    initial_alpha = np.asarray(
        initial_alpha,
        dtype=float,
    ).reshape(4)

    events = _find_events(
        stages,
        initial_alpha=initial_alpha,
    )

    desired_by_stage = (
        _desired_fixed_angles_by_stage(
            events,
            n_stage=len(
                stages
            ),
        )
    )

    output: list[
        StageReference
    ] = []

    previous_alpha = (
        initial_alpha.copy()
    )

    attempted = 0
    applied = 0
    fallback = 0

    applied_by_rotor = np.zeros(
        4,
        dtype=int,
    )

    fallback_by_rotor = np.zeros(
        4,
        dtype=int,
    )

    first_modified_stage: list[
        int | None
    ] = [
        None,
        None,
        None,
        None,
    ]

    first_modified_alpha: list[
        float | None
    ] = [
        None,
        None,
        None,
        None,
    ]

    max_residual = 0.0

    for stage_index, baseline in enumerate(
        stages
    ):
        desired = (
            desired_by_stage[
                stage_index
            ]
        )

        # Keep only rotors that are actually low-thrust at this stage.
        desired = {
            rotor: angle
            for rotor, angle in desired.items()
            if baseline.parameters[
                P_THRUST
            ][
                rotor
            ] <= (
                PREPOSITION_UNLOADED_THRUST_N
            )
        }

        if not desired:
            output.append(
                baseline
            )

            previous_alpha = np.asarray(
                baseline.parameters[
                    P_ALPHA
                ],
                dtype=float,
            ).copy()

            continue

        attempted += 1

        # If more than two events overlap, prioritize rotors with the
        # largest requested angle departure from the current reference.
        ordered = sorted(
            desired,
            key=lambda rotor: abs(
                desired[
                    rotor
                ]
                - previous_alpha[
                    rotor
                ]
            ),
            reverse=True,
        )

        ordered = ordered[
            :
            MAX_SIMULTANEOUS_PREPOSITION_ROTORS
        ]

        candidate_sets: list[
            dict[int, float]
        ] = []

        if len(
            ordered
        ) == 2:
            candidate_sets.append(
                {
                    rotor: desired[
                        rotor
                    ]
                    for rotor in ordered
                }
            )

        # Safe degradation: if the two-rotor exact solve is infeasible,
        # try each rotor individually before falling back completely.
        for rotor in ordered:
            candidate_sets.append(
                {
                    rotor: desired[
                        rotor
                    ]
                }
            )

        solved = None
        solved_candidate: dict[
            int,
            float,
        ] | None = None

        for candidate in candidate_sets:
            solved = (
                _solve_exact_with_fixed_angles(
                    baseline,
                    fixed_angles=candidate,
                    previous_alpha=previous_alpha,
                )
            )

            if solved is not None:
                solved_candidate = (
                    candidate
                )
                break

        if (
            solved is None
            or solved_candidate is None
        ):
            output.append(
                baseline
            )

            previous_alpha = np.asarray(
                baseline.parameters[
                    P_ALPHA
                ],
                dtype=float,
            ).copy()

            fallback += 1

            for rotor in desired:
                fallback_by_rotor[
                    rotor
                ] += 1

            continue

        (
            thrust_ref,
            alpha_ref,
            residual,
        ) = solved

        modified = (
            _stage_with_actuator_reference(
                baseline,
                thrust_ref=thrust_ref,
                alpha_ref=alpha_ref,
            )
        )

        output.append(
            modified
        )

        previous_alpha = (
            alpha_ref.copy()
        )

        applied += 1

        for rotor in solved_candidate:
            applied_by_rotor[
                rotor
            ] += 1

            if (
                first_modified_stage[
                    rotor
                ]
                is None
            ):
                first_modified_stage[
                    rotor
                ] = stage_index

                first_modified_alpha[
                    rotor
                ] = float(
                    alpha_ref[
                        rotor
                    ]
                )

        # A requested preview rotor that could not be included in the
        # successful reduced solve is explicitly counted as a fallback
        # for that rotor at this stage.
        for rotor in desired:
            if rotor not in solved_candidate:
                fallback_by_rotor[
                    rotor
                ] += 1

        max_residual = max(
            max_residual,
            float(
                residual
            ),
        )

    diagnostics = (
        PredictivePrepositionDiagnostics(
            events=tuple(
                events
            ),
            attempted_stage_count=(
                attempted
            ),
            applied_stage_count=(
                applied
            ),
            fallback_stage_count=(
                fallback
            ),
            applied_stage_count_by_rotor=tuple(
                int(value)
                for value in applied_by_rotor
            ),
            fallback_stage_count_by_rotor=tuple(
                int(value)
                for value in fallback_by_rotor
            ),
            first_modified_stage_by_rotor=tuple(
                first_modified_stage
            ),
            first_modified_alpha_rad_by_rotor=tuple(
                first_modified_alpha
            ),
            maximum_wrench_residual=(
                max_residual
            ),
        )
    )

    return (
        output,
        diagnostics,
    )


def build_predictive_effector_reference_horizon(
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
    Build the ordinary validated end-effector reference horizon first,
    then apply the optional predictive low-thrust pre-positioning layer.
    """
    baseline = (
        build_effector_reference_horizon(
            trajectory,
            start_time_s=start_time_s,
            stage_step_s=stage_step_s,
            n_horizon=n_horizon,
            initial_body_quaternion=(
                initial_body_quaternion
            ),
            initial_alpha=(
                initial_alpha
            ),
            transform=transform,
        )
    )

    predictive, _ = (
        apply_predictive_low_thrust_preposition(
            baseline,
            initial_alpha=initial_alpha,
        )
    )

    return predictive


def build_predictive_effector_reference_horizon_with_diagnostics(
    trajectory: EndEffectorReferenceTrajectory,
    *,
    start_time_s: float,
    stage_step_s: float,
    n_horizon: int,
    initial_body_quaternion: np.ndarray,
    initial_alpha: np.ndarray,
    transform: RigidEndEffectorTransform = DEFAULT_END_EFFECTOR,
) -> tuple[
    list[StageReference],
    PredictivePrepositionDiagnostics,
]:
    baseline = (
        build_effector_reference_horizon(
            trajectory,
            start_time_s=start_time_s,
            stage_step_s=stage_step_s,
            n_horizon=n_horizon,
            initial_body_quaternion=(
                initial_body_quaternion
            ),
            initial_alpha=(
                initial_alpha
            ),
            transform=transform,
        )
    )

    return (
        apply_predictive_low_thrust_preposition(
            baseline,
            initial_alpha=initial_alpha,
        )
    )
