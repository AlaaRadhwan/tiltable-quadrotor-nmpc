from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from acados_template import (
    AcadosOcpSolver,
)

from tiltable_quadrotor_rl.control.nmpc_model import (
    NU,
    NX,
)
from tiltable_quadrotor_rl.control.nmpc_ocp import (
    N_HORIZON,
    PREDICTION_STEP,
    build_ocp,
)
from tiltable_quadrotor_rl.control.nmpc_reference import (
    ReferenceTrajectory,
    StageReference,
    build_reference_horizon,
    normalize_quaternion,
)
from tiltable_quadrotor_rl.robot.parameters import (
    MAX_THRUST,
    SERVO_COMMAND_LIMIT,
)


@dataclass(frozen=True)
class NmpcCommand:
    thrust: np.ndarray
    alpha_cmd: np.ndarray

    status: int
    solve_time_s: float


class NmpcController:
    """
    General 6-DoF trajectory-tracking NMPC wrapper.

    One controller handles setpoint regulation and arbitrary
    time-varying 6-DoF trajectories. Different maneuvers are
    different references, not different controller code.
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

        ocp = build_ocp(
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


    @staticmethod
    def _prepare_state(
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Validate the measured 17-state vector and normalize q.

        Normalization is measurement-side housekeeping; the NMPC
        dynamics themselves remain the validated continuous model.
        """

        state = np.asarray(
            state,
            dtype=float,
        ).reshape(-1)

        if state.shape != (NX,):
            raise ValueError(
                f"Expected state shape ({NX},), "
                f"got {state.shape}."
            )

        if not np.all(
            np.isfinite(
                state
            )
        ):
            raise ValueError(
                "NMPC state contains non-finite values."
            )

        state = state.copy()

        state[6:10] = (
            normalize_quaternion(
                state[6:10]
            )
        )

        return state


    def _build_horizon(
        self,
        *,
        state: np.ndarray,
        trajectory: ReferenceTrajectory,
        time_s: float,
    ) -> list[StageReference]:
        """
        Build all N+1 references together so quaternion and
        actuator-angle references remain continuous across the horizon.
        """

        return build_reference_horizon(
            trajectory,
            start_time_s=time_s,
            stage_step_s=PREDICTION_STEP,
            n_horizon=N_HORIZON,
            initial_quaternion=state[6:10],
            initial_alpha=state[13:17],
        )


    def _set_stage_parameters(
        self,
        horizon: list[StageReference],
    ) -> None:
        """
        Update the time-varying reference at every shooting node.

        Parameters change every control cycle; the generated solver
        structure itself does not need to be rebuilt.
        """

        if len(horizon) != (
            N_HORIZON + 1
        ):
            raise ValueError(
                "Reference horizon has the wrong length."
            )

        for stage, reference in enumerate(
            horizon
        ):
            self.solver.set(
                stage,
                "p",
                reference.parameters,
            )


    def _initialize_iterate(
        self,
        *,
        state: np.ndarray,
        horizon: list[StageReference],
    ) -> None:
        """
        Initialize the first SQP_RTI solve from the trajectory
        references instead of the default all-zero iterate.

        This follows the practical idea in the paper: reference
        generation gives NMPC an initial solution already close
        to the desired motion.

        This method is also used as a one-time recovery retry if a
        later warm-started solve fails.
        """

        for stage in range(
            N_HORIZON + 1
        ):
            x_guess = (
                horizon[
                    stage
                ].x_reference.copy()
            )

            if stage == 0:
                x_guess = state.copy()

            self.solver.set(
                stage,
                "x",
                x_guess,
            )

        for stage in range(
            N_HORIZON
        ):
            self.solver.set(
                stage,
                "u",
                horizon[
                    stage
                ].u_reference,
            )


    def _set_initial_state(
        self,
        state: np.ndarray,
    ) -> None:
        """
        Enforce x_0 = measured state.

        Also update the stage-0 primal guess to the same measured
        state. The rest of the previous solution is intentionally
        left in place as the warm-start trajectory.
        """

        self.solver.constraints_set(
            0,
            "lbx",
            state,
        )

        self.solver.constraints_set(
            0,
            "ubx",
            state,
        )

        self.solver.set(
            0,
            "x",
            state,
        )


    def _solve_once(
        self,
    ) -> tuple[int, float]:
        status = int(
            self.solver.solve()
        )

        solve_time_s = float(
            self.solver.get_stats(
                "time_tot"
            )
        )

        return (
            status,
            solve_time_s,
        )


    def solve(
        self,
        *,
        state: np.ndarray,
        trajectory: ReferenceTrajectory,
        time_s: float,
    ) -> NmpcCommand:
        """
        Execute one 100 Hz NMPC update.

        Important:
        - The previous acados primal iterate is NOT shifted by one
          0.1 s shooting node. Only 0.01 s has elapsed between calls.
        - We keep the previous solution in the solver as the next
          warm-start trajectory, then replace stage 0 with the current
          measured state.
        - If that warm-started solve fails, we retry once after
          rebuilding the iterate from the current references.
        """

        state = self._prepare_state(
            state
        )

        horizon = self._build_horizon(
            state=state,
            trajectory=trajectory,
            time_s=time_s,
        )

        self._set_stage_parameters(
            horizon
        )

        if not self._initialized:
            self._initialize_iterate(
                state=state,
                horizon=horizon,
            )

        self._set_initial_state(
            state
        )

        status, solve_time_s = (
            self._solve_once()
        )

        if status != 0:
            # A previous solution can become a poor warm start after a
            # discontinuous or aggressive reference update. Retry once
            # from the current reference trajectory before declaring
            # controller failure.
            self._initialize_iterate(
                state=state,
                horizon=horizon,
            )

            self._set_initial_state(
                state
            )

            status, solve_time_s = (
                self._solve_once()
            )

        if status != 0:
            raise RuntimeError(
                "acados NMPC solve failed after "
                f"reference reinitialization; status={status}."
            )

        command = np.asarray(
            self.solver.get(
                0,
                "u",
            ),
            dtype=float,
        ).reshape(-1)

        if command.shape != (NU,):
            raise RuntimeError(
                "Unexpected NMPC command shape: "
                f"{command.shape}."
            )

        if not np.all(
            np.isfinite(
                command
            )
        ):
            raise RuntimeError(
                "NMPC returned non-finite actuator commands."
            )

        # Hard bounds are already enforced by acados. The final clip
        # is a numerical guard against tiny tolerance-level overshoot
        # before commands are passed to the plant or future hardware.
        thrust = np.clip(
            command[0:4],
            0.0,
            MAX_THRUST,
        )

        alpha_cmd = np.clip(
            command[4:8],
            -SERVO_COMMAND_LIMIT,
            SERVO_COMMAND_LIMIT,
        )

        self._initialized = True

        return NmpcCommand(
            thrust=thrust.copy(),
            alpha_cmd=alpha_cmd.copy(),
            status=status,
            solve_time_s=solve_time_s,
        )
