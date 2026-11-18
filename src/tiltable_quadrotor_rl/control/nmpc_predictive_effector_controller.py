from __future__ import annotations

import time

import numpy as np

from tiltable_quadrotor_rl.control.end_effector import (
    DEFAULT_END_EFFECTOR,
)
from tiltable_quadrotor_rl.control.nmpc_effector_controller import (
    EffectorNmpcController,
)
from tiltable_quadrotor_rl.control.nmpc_effector_reference import (
    EndEffectorReferenceTrajectory,
)
from tiltable_quadrotor_rl.control.nmpc_ocp import (
    N_HORIZON,
    PREDICTION_STEP,
)
from tiltable_quadrotor_rl.control.nmpc_predictive_preposition import (
    PredictivePrepositionDiagnostics,
    build_predictive_effector_reference_horizon_with_diagnostics,
)
from tiltable_quadrotor_rl.control.nmpc_reference import (
    StageReference,
)


class TimedEffectorNmpcController(
    EffectorNmpcController
):
    """
    Baseline end-effector NMPC with reference-horizon timing.

    The controller behavior is unchanged. The only addition is runtime
    instrumentation used by the experiment runner.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs
        )

        self.last_reference_generation_time_s = 0.0
        self.last_predictive_diagnostics = None

    def _build_horizon(
        self,
        *,
        state: np.ndarray,
        trajectory: EndEffectorReferenceTrajectory,
        time_s: float,
    ) -> list[StageReference]:
        start = time.perf_counter()

        horizon = super()._build_horizon(
            state=state,
            trajectory=trajectory,
            time_s=time_s,
        )

        self.last_reference_generation_time_s = (
            time.perf_counter()
            - start
        )

        self.last_predictive_diagnostics = None

        return horizon


class PredictivePrepositionEffectorNmpcController(
    EffectorNmpcController
):
    """
    Experimental end-effector NMPC wrapper with predictive low-thrust
    rotor pre-positioning.

    The OCP, dynamics, weights, constraints and solver are unchanged.
    Only the actuator-reference horizon is modified.

    Runtime instrumentation exposes:
      - last_reference_generation_time_s
      - last_predictive_diagnostics

    for the current 100 Hz control update.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs
        )

        self.last_reference_generation_time_s = 0.0

        self.last_predictive_diagnostics: (
            PredictivePrepositionDiagnostics
            | None
        ) = None

    def _build_horizon(
        self,
        *,
        state: np.ndarray,
        trajectory: EndEffectorReferenceTrajectory,
        time_s: float,
    ) -> list[StageReference]:
        start = time.perf_counter()

        (
            horizon,
            diagnostics,
        ) = (
            build_predictive_effector_reference_horizon_with_diagnostics(
                trajectory,
                start_time_s=time_s,
                stage_step_s=PREDICTION_STEP,
                n_horizon=N_HORIZON,
                initial_body_quaternion=(
                    state[6:10]
                ),
                initial_alpha=(
                    state[13:17]
                ),
                transform=DEFAULT_END_EFFECTOR,
            )
        )

        self.last_reference_generation_time_s = (
            time.perf_counter()
            - start
        )

        self.last_predictive_diagnostics = (
            diagnostics
        )

        return horizon
