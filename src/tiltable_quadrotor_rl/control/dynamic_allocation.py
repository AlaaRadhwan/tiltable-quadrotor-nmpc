from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from tiltable_quadrotor_rl.control.allocation import (
    ARM_LENGTH,
    MAX_COMMAND_ANGLE,
    MAX_THRUST,
    thrust_allocation_matrix,
)
from tiltable_quadrotor_rl.robot.tilt_servo import (
    DEFAULT_TIME_CONSTANT,
)

DEFAULT_CONTROL_DT = 0.01
DEFAULT_HORIZON_STEPS = 3

def predict_tilt_angles(
        alpha: np.ndarray,
        alpha_cmd: np.ndarray,
        dt: float = DEFAULT_CONTROL_DT,
        time_constant: float = DEFAULT_TIME_CONSTANT
) -> np.ndarray:
    """
    Predict rotor tilt angles one controller step ahead.
    
    Use the exact discrete solution of 
        alpha_dot = (alpha_cmd - alpha) / tau
    
    """

    alpha = np.asarray(
        alpha,
        dtype=float,
    )

    alpha_cmd = np.asarray(
        alpha_cmd,
        dtype=float,
    )

    if alpha.shape != (4,):
        raise ValueError(
            f"Expected alpha shape (4,), got {alpha.shape}"
        )

    if alpha_cmd.shape != (4,):
        raise ValueError(
            f"Expected alpha_cmd shape (4,), got {alpha_cmd.shape}"
        )

    if dt <= 0.0:
        raise ValueError(
            "dt must be positive."
        )

    if time_constant <= 0.0:
        raise ValueError(
            "time_constant must be positive."
        )

    decay = math.exp(
        -dt / time_constant
    )

    return (
        alpha_cmd + (alpha - alpha_cmd) * decay
    )


def allocate_dynamic_one_step(
        desired_wrench: np.ndarray,
        current_alpha: np.ndarray,
        previous_thrust: np.ndarray,
        previous_alpha_cmd: np.ndarray,
        *,
        dt: float = DEFAULT_CONTROL_DT,
        time_constant: float = DEFAULT_TIME_CONSTANT,
        wrench_weights: float | None = None,
        thrust_regularization: float = 1e-4,
        angle_regularization: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    One-step dynamic actuator allocator.

    Jointly chooses:
        thrust = [f1, f2, f3, f4]
        alpha_cmd = [a1_cmd, ..., a4_cmd]

    While predicting the servo state one control step ahead.

    This is intentionally a first baseline implementation.
    It does not yet include:
        - angle winding / branch handling,
        - singularity avoidance,
        - multi-step prediction,
        - task-specific wrench priorities.
    """

    desired_wrench = np.asarray(
        desired_wrench,
        dtype=float,
    )

    current_alpha = np.asarray(
        current_alpha,
        dtype=float,
    )

    previous_thrust = np.asarray(
        previous_thrust,
        dtype=float,
    )

    previous_alpha_cmd = np.asarray(
        previous_alpha_cmd,
        dtype=float,
    )

    if desired_wrench.shape != (6,):
        raise ValueError(
            "desired_wrench must have shape (6,)"
        )

    if current_alpha.shape != (4,):
        raise ValueError(
            "current_alpha must have shape (4,)"
        )

    if previous_thrust.shape != (4,):
        raise ValueError(
            "previous_thrust must have shape (4,)"
        )

    if previous_alpha_cmd.shape != (4,):
        raise ValueError(
            "previous_alpha_cmd must have shape (4,)"
        )

    if wrench_weights is None:
        wrench_weights = np.ones(
            6,
            dtype=float,
        )
    else:
        wrench_weights = np.asarray(
            wrench_weights,
            dtype=float,
        )

        if wrench_weights.shape != (6,):
            raise ValueError(
                "wrench_weights must have shape (6,)"
            )

    # --------------------------------------------------------
    # Unit normalization
    #
    # Forces are normalized by vehicle weight.
    # Moments are normalized by weight * arm length.
    #
    # This makes the residual dimensionless.
    #
    # It is NOT intended to encode a permanent priority.
    # --------------------------------------------------------

    mass = 3.039
    gravity = 9.81

    force_scale = mass * gravity

    torque_scale = force_scale * ARM_LENGTH

    wrench_scale = np.array([
        force_scale,
        force_scale,
        force_scale,
        torque_scale,
        torque_scale,
        torque_scale,
    ])

    sqrt_thrust_reg = math.sqrt(
        thrust_regularization
    )

    sqrt_angle_reg = math.sqrt(
        angle_regularization
    )

    # Decision vector:
    #
    # x = [
    #       f1, f2, f3, f4,
    #       alpha_cmd1, ..., alpha_cmd4,
    #]

    def residual(
            x: np.ndarray,
    ) -> np.ndarray:

        thrust = x[:4]

        alpha_cmd = x[4:]

        predicted_alpha = (
            predict_tilt_angles(
                current_alpha,
                alpha_cmd,
                dt=dt,
                time_constant=time_constant,
            )
        )

        B = thrust_allocation_matrix(predicted_alpha)

        predicted_wrench = (
            B @ thrust
        )

        wrench_error = (
            predicted_wrench
            - desired_wrench
        )

        normalized_wrench_error = (
            wrench_weights
            * wrench_error
            / wrench_scale
        )

        thrust_change = (
            thrust
            - previous_thrust
        ) / MAX_THRUST

        angle_command_change = (
            alpha_cmd
            - previous_alpha_cmd
        ) / MAX_COMMAND_ANGLE

        return np.concatenate([
            normalized_wrench_error,
            sqrt_thrust_reg
            * thrust_change,
            sqrt_angle_reg
            * angle_command_change,
        ])

    x0 = np.concatenate([
        np.clip(
            previous_thrust,
            0.0,
            MAX_THRUST,
        ),
        np.clip(
            previous_alpha_cmd,
            -MAX_COMMAND_ANGLE,
            MAX_COMMAND_ANGLE,
        ),
    ])

    lower_bounds = np.concatenate([
        np.zeros(4),
        np.full(
            4,
            -MAX_COMMAND_ANGLE,
        ),
    ])

    upper_bounds = np.concatenate([
        np.full(
            4,
            MAX_THRUST,
        ),
        np.full(
            4,
            MAX_COMMAND_ANGLE,
        ),
    ])

    result = least_squares(
        residual,
        x0,
        bounds=(
            lower_bounds,
            upper_bounds,
        ),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=500,
    )

    thrust = (
        result.x[:4].copy()
    )

    alpha_cmd = (
        result.x[4:].copy()
    )

    predicted_alpha = (
        predict_tilt_angles(
            current_alpha,
            alpha_cmd,
            dt=dt,
            time_constant=time_constant
        )
    )

    B = thrust_allocation_matrix(predicted_alpha)

    predicted_wrench = B @ thrust

    wrench_error = (
        predicted_wrench
        - desired_wrench
    )

    info = {
        "success": bool(
            result.success
        ),

        "message": result.message,

        "cost": float(
            result.cost
        ),

        "function_evaluations": int(
            result.nfev
        ),

        "predicted_alpha": (
            predicted_alpha
        ),

        "predicted_wrench": (
            predicted_wrench
        ),

        "wrench_error": (
            wrench_error
        ),

        "wrench_error_norm": float(
            np.linalg.norm(
                wrench_error
            )
        ),
    }

    return (
        thrust,
        alpha_cmd,
        info,
    )


def allocate_dynamic_horizon(
    desired_wrench: np.ndarray,
    current_alpha: np.ndarray,
    previous_thrust: np.ndarray,
    previous_alpha_cmd: np.ndarray,
    *,
    horizon_steps: int = DEFAULT_HORIZON_STEPS,
    control_dt: float = DEFAULT_CONTROL_DT,
    time_constant: float = DEFAULT_TIME_CONSTANT,
    wrench_weights: np.ndarray | None = None,
    thrust_regularization: float = 1e-4,
    angle_regularization: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Finite-horizon dynamic control allocator.

    The optimizer chooses one thrust vector and one tilt-command
    vector for each control interval in the horizon.

    For a 3-step horizon at 100 Hz:

        interval 0:   0 -> 10 ms
        interval 1:  10 -> 20 ms
        interval 2:  20 -> 30 ms

    Only the FIRST actuator command is returned for application.
    The remaining planned commands are used only for prediction.

    Wrench tracking is evaluated at both the start and the end
    of every interval so that transient wrench errors cannot be
    hidden by a good terminal solution.
    """

    desired_wrench = np.asarray(
        desired_wrench,
        dtype=float,
    )

    current_alpha = np.asarray(
        current_alpha,
        dtype=float,
    )

    previous_thrust = np.asarray(
        previous_thrust,
        dtype=float,
    )

    previous_alpha_cmd = np.asarray(
        previous_alpha_cmd,
        dtype=float,
    )

    if desired_wrench.shape != (6,):
        raise ValueError(
            "desired_wrench must have shape (6,)"
        )

    if current_alpha.shape != (4,):
        raise ValueError(
            "current_alpha must have shape (4,)"
        )

    if previous_thrust.shape != (4,):
        raise ValueError(
            "previous_thrust must have shape (4,)"
        )

    if previous_alpha_cmd.shape != (4,):
        raise ValueError(
            "previous_alpha_cmd must have shape (4,)"
        )

    if horizon_steps < 1:
        raise ValueError(
            "horizon_steps must be >= 1"
        )

    if control_dt <= 0.0:
        raise ValueError(
            "control_dt must be positive"
        )

    if wrench_weights is None:
        wrench_weights = np.ones(
            6,
            dtype=float,
        )
    else:
        wrench_weights = np.asarray(
            wrench_weights,
            dtype=float,
        )

        if wrench_weights.shape != (6,):
            raise ValueError(
                "wrench_weights must have shape (6,)"
            )

    # --------------------------------------------------------
    # Dimensionless wrench scaling
    # --------------------------------------------------------

    mass = 3.039
    gravity = 9.81

    force_scale = (
        mass * gravity
    )

    torque_scale = (
        force_scale * ARM_LENGTH
    )

    wrench_scale = np.array([
        force_scale,
        force_scale,
        force_scale,
        torque_scale,
        torque_scale,
        torque_scale,
    ])

    # There are two wrench samples per interval:
    #
    #   start of interval
    #   end of interval
    #
    # Divide by sqrt(number of samples) so changing horizon
    # length does not automatically change the overall wrench
    # cost magnitude.
    wrench_sample_factor = (
        1.0
        / math.sqrt(
            2 * horizon_steps
        )
    )

    thrust_change_factor = math.sqrt(
        thrust_regularization
        / horizon_steps
    )

    angle_change_factor = math.sqrt(
        angle_regularization
        / horizon_steps
    )

    # --------------------------------------------------------
    # Decision vector
    #
    # For each horizon step:
    #
    #   [f1, f2, f3, f4,
    #    alpha_cmd1, ..., alpha_cmd4]
    #
    # 3 steps -> 24 decision variables.
    # --------------------------------------------------------

    previous_command = np.concatenate([
        previous_thrust,
        previous_alpha_cmd,
    ])

    x0 = np.tile(
        previous_command,
        horizon_steps,
    )

    single_lower = np.concatenate([
        np.zeros(4),
        np.full(
            4,
            -MAX_COMMAND_ANGLE,
        ),
    ])

    single_upper = np.concatenate([
        np.full(
            4,
            MAX_THRUST,
        ),
        np.full(
            4,
            MAX_COMMAND_ANGLE,
        ),
    ])

    lower_bounds = np.tile(
        single_lower,
        horizon_steps,
    )

    upper_bounds = np.tile(
        single_upper,
        horizon_steps,
    )

    x0 = np.clip(
        x0,
        lower_bounds,
        upper_bounds,
    )

    # --------------------------------------------------------
    # Objective residual
    # --------------------------------------------------------

    def residual(
        x: np.ndarray,
    ) -> np.ndarray:

        plan = x.reshape(
            horizon_steps,
            8,
        )

        thrust_plan = (
            plan[:, :4]
        )

        alpha_cmd_plan = (
            plan[:, 4:]
        )

        alpha = (
            current_alpha.copy()
        )

        wrench_residuals = []

        thrust_change_residuals = []
        angle_change_residuals = []

        prev_thrust = (
            previous_thrust
        )

        prev_alpha_cmd = (
            previous_alpha_cmd
        )

        for step in range(
            horizon_steps
        ):

            thrust = (
                thrust_plan[step]
            )

            alpha_cmd = (
                alpha_cmd_plan[step]
            )

            # --------------------------------------------
            # Wrench at START of interval.
            #
            # This directly captures the instantaneous
            # wrench when the new thrust is applied before
            # the servo has moved.
            # --------------------------------------------

            B_start = (
                thrust_allocation_matrix(
                    alpha
                )
            )

            wrench_start = (
                B_start @ thrust
            )

            start_error = (
                wrench_weights
                * (
                    wrench_start
                    - desired_wrench
                )
                / wrench_scale
            )

            wrench_residuals.append(
                wrench_sample_factor
                * start_error
            )

            # --------------------------------------------
            # Predict servo to END of interval.
            # --------------------------------------------

            alpha_next = (
                predict_tilt_angles(
                    alpha,
                    alpha_cmd,
                    dt=control_dt,
                    time_constant=time_constant,
                )
            )

            B_end = (
                thrust_allocation_matrix(
                    alpha_next
                )
            )

            wrench_end = (
                B_end @ thrust
            )

            end_error = (
                wrench_weights
                * (
                    wrench_end
                    - desired_wrench
                )
                / wrench_scale
            )

            wrench_residuals.append(
                wrench_sample_factor
                * end_error
            )

            # --------------------------------------------
            # Command continuity
            # --------------------------------------------

            thrust_change = (
                thrust
                - prev_thrust
            ) / MAX_THRUST

            angle_change = (
                alpha_cmd
                - prev_alpha_cmd
            ) / MAX_COMMAND_ANGLE

            thrust_change_residuals.append(
                thrust_change_factor
                * thrust_change
            )

            angle_change_residuals.append(
                angle_change_factor
                * angle_change
            )

            prev_thrust = (
                thrust
            )

            prev_alpha_cmd = (
                alpha_cmd
            )

            alpha = (
                alpha_next
            )

        return np.concatenate([
            *wrench_residuals,
            *thrust_change_residuals,
            *angle_change_residuals,
        ])

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    result = least_squares(
        residual,
        x0,
        bounds=(
            lower_bounds,
            upper_bounds,
        ),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=1000,
    )

    plan = result.x.reshape(
        horizon_steps,
        8,
    )

    thrust_plan = (
        plan[:, :4].copy()
    )

    alpha_cmd_plan = (
        plan[:, 4:].copy()
    )

    # --------------------------------------------------------
    # Reconstruct predicted trajectory for inspection.
    # --------------------------------------------------------

    alpha = (
        current_alpha.copy()
    )

    alpha_trajectory = [
        alpha.copy()
    ]

    wrench_start_trajectory = []
    wrench_end_trajectory = []

    for step in range(
        horizon_steps
    ):

        thrust = (
            thrust_plan[step]
        )

        alpha_cmd = (
            alpha_cmd_plan[step]
        )

        B_start = (
            thrust_allocation_matrix(
                alpha
            )
        )

        wrench_start = (
            B_start @ thrust
        )

        wrench_start_trajectory.append(
            wrench_start
        )

        alpha_next = (
            predict_tilt_angles(
                alpha,
                alpha_cmd,
                dt=control_dt,
                time_constant=time_constant,
            )
        )

        B_end = (
            thrust_allocation_matrix(
                alpha_next
            )
        )

        wrench_end = (
            B_end @ thrust
        )

        wrench_end_trajectory.append(
            wrench_end
        )

        alpha = (
            alpha_next
        )

        alpha_trajectory.append(
            alpha.copy()
        )

    alpha_trajectory = np.stack(
        alpha_trajectory
    )

    wrench_start_trajectory = np.stack(
        wrench_start_trajectory
    )

    wrench_end_trajectory = np.stack(
        wrench_end_trajectory
    )

    info = {
        "success": bool(
            result.success
        ),

        "message": (
            result.message
        ),

        "cost": float(
            result.cost
        ),

        "function_evaluations": int(
            result.nfev
        ),

        "thrust_plan": (
            thrust_plan
        ),

        "alpha_cmd_plan": (
            alpha_cmd_plan
        ),

        "alpha_trajectory": (
            alpha_trajectory
        ),

        "wrench_start_trajectory": (
            wrench_start_trajectory
        ),

        "wrench_end_trajectory": (
            wrench_end_trajectory
        ),
    }

    # Receding horizon:
    #
    # APPLY ONLY THE FIRST CONTROL ACTION.
    return (
        thrust_plan[0].copy(),
        alpha_cmd_plan[0].copy(),
        info,
    )