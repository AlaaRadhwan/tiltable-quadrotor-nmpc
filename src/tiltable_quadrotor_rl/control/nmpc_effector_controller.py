from __future__ import annotations

from pathlib import Path

import numpy as np

from acados_template import AcadosOcpSolver

from tiltable_quadrotor_rl.control.end_effector import (
    DEFAULT_END_EFFECTOR,
)
from tiltable_quadrotor_rl.control.nmpc_controller import (
    NmpcCommand,
    NmpcController,
)
from tiltable_quadrotor_rl.control.nmpc_effector_ocp import (
    build_effector_ocp,
)
from tiltable_quadrotor_rl.control.nmpc_effector_reference import (
    EndEffectorReferenceTrajectory,
    build_effector_reference_horizon,
)
from tiltable_quadrotor_rl.control.nmpc_ocp import (
    N_HORIZON,
    PREDICTION_STEP,
)
from tiltable_quadrotor_rl.control.nmpc_reference import (
    StageReference,
)


class EffectorNmpcController(
    NmpcController
):
    """
    End-effector-centric extension of the validated NMPC wrapper.

    Warm starting, retry handling, hard-command checks and command
    extraction are inherited unchanged.

    Only:
      1. the OCP tracking residual,
      2. the reference-horizon construction

    are specialized for the rigid end effector.
    """

    def __init__(
        self,
        *,
        build_directory: str | Path,
        json_file: str | Path,
        generate: bool = True,
        build: bool = True,
        verbose: bool = False,
    ) -> None:

        build_directory = Path(
            build_directory
        ).resolve()

        json_file = Path(
            json_file
        ).resolve()

        build_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        ocp = build_effector_ocp(
            code_export_directory=build_directory,
        )

        self.solver = AcadosOcpSolver(
            ocp,
            json_file=str(
                json_file
            ),
            generate=generate,
            build=build,
            verbose=verbose,
        )

        self._initialized = False

    def _build_horizon(
        self,
        *,
        state: np.ndarray,
        trajectory: EndEffectorReferenceTrajectory,
        time_s: float,
    ) -> list[StageReference]:

        return build_effector_reference_horizon(
            trajectory,
            start_time_s=time_s,
            stage_step_s=PREDICTION_STEP,
            n_horizon=N_HORIZON,
            initial_body_quaternion=state[6:10],
            initial_alpha=state[13:17],
            transform=DEFAULT_END_EFFECTOR,
        )

    def solve(
        self,
        *,
        state: np.ndarray,
        trajectory: EndEffectorReferenceTrajectory,
        time_s: float,
    ) -> NmpcCommand:
        return super().solve(
            state=state,
            trajectory=trajectory,
            time_s=time_s,
        )
