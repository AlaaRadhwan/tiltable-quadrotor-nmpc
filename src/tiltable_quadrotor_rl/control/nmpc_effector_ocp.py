from __future__ import annotations

from pathlib import Path

import casadi as ca
import numpy as np

from acados_template import AcadosOcp

from tiltable_quadrotor_rl.control.end_effector import (
    DEFAULT_END_EFFECTOR,
    body_to_end_effector_kinematics_ca,
)
from tiltable_quadrotor_rl.control.nmpc_model import (
    NU,
    NX,
)
from tiltable_quadrotor_rl.control.nmpc_ocp import (
    N_HORIZON,
    PREDICTION_HORIZON,
    Q_ALPHA,
    Q_ATTITUDE,
    Q_OMEGA,
    Q_POSITION,
    Q_VELOCITY,
    R_DELTA_ALPHA,
    R_THRUST,
    build_acados_model,
    quaternion_error_vector,
)
from tiltable_quadrotor_rl.control.nmpc_reference import (
    NP,
    P_ALPHA,
    P_OMEGA,
    P_POSITION,
    P_QUATERNION,
    P_THRUST,
    P_VELOCITY,
)
from tiltable_quadrotor_rl.robot.parameters import (
    HOVER_THRUST,
    MAX_THRUST,
    SERVO_COMMAND_LIMIT,
)


# ============================================================
# End-effector-specific tracking weights
# ============================================================
#
# We intentionally leave the generic CoG NMPC weights untouched.
#
# Previous end-effector controller reused:
#
#   Q_POSITION = [300, 300, 400]
#   Q_VELOCITY = [10, 10, 10]
#
# During the fixed-pivot pitch experiment the dominant visible error
# was dynamic end-effector position drift (~5 cm RMS), while the solver
# remained healthy and the original direct servo-angle cost handled the
# singularity region better than the periodic-angle experiment.
#
# First controlled tracking adjustment:
#
#   position weights: 2x
#   velocity weights: 3x
#
# Everything else stays unchanged:
#   attitude weights
#   angular-rate weights
#   direct alpha-alpha_ref tracking
#   thrust-reference weight
#   alpha_cmd-alpha regularization
#   constraints
#   horizon
#   dynamics
#
# This is deliberately conservative: we want to test whether the
# controller was simply accepting too much tool-space translation error
# before changing actuator penalties or architecture.

Q_EFFECTOR_POSITION = (
    2.0
    * np.asarray(
        Q_POSITION,
        dtype=float,
    )
)

Q_EFFECTOR_VELOCITY = (
    3.0
    * np.asarray(
        Q_VELOCITY,
        dtype=float,
    )
)


# End-effector-specific servo-motion regularization.
#
# The baseline inherited R_DELTA_ALPHA = 250 from the body-centered NMPC.
# In the servo model:
#
#     alpha_dot = (alpha_cmd - alpha) / tau
#
# therefore penalizing (alpha_cmd - alpha)^2 is effectively a servo-rate
# penalty. The fixed-pivot telemetry shows a strong rate-correlated X offset,
# so the first controlled test reduces only this term.
#
# Keep the base/body-centered controller unchanged.
R_EFFECTOR_DELTA_ALPHA = np.full(
    4,
    50.0,
    dtype=float,
)


def build_effector_ocp(
    *,
    code_export_directory: str | Path | None = None,
) -> AcadosOcp:
    """
    Build the end-effector-centric NMPC.

    State and input dynamics are unchanged from the validated rigid
    plant NMPC.

    Important choices in this version:

      1. Restore the original NON-PERIODIC physical servo-reference
         residual:

             alpha - alpha_ref

         The physical tilt servo is a bounded real mechanical state,
         so preserving its branch information produced better behavior
         than the periodic direction-only cost in our closed-loop test.

      2. Increase only the end-effector position and velocity tracking
         weights to reduce the ~5 cm dynamic fixed-pivot drift.

      3. Keep all actuator penalties, constraints, horizon length and
         reference generation unchanged for a clean A/B comparison.
    """

    model = build_acados_model()

    model.name = (
        "tiltable_quadrotor_effector_nmpc"
    )

    x = model.x
    u = model.u
    p = model.p

    position_B_W = x[0:3]
    velocity_B_W = x[3:6]
    quaternion_WB = x[6:10]
    omega_B = x[10:13]
    alpha = x[13:17]

    thrust = u[0:4]
    alpha_cmd = u[4:8]

    (
        position_E_W,
        velocity_E_W,
        quaternion_WE,
        omega_E,
    ) = body_to_end_effector_kinematics_ca(
        position_B_W,
        velocity_B_W,
        quaternion_WB,
        omega_B,
        transform=DEFAULT_END_EFFECTOR,
    )

    position_E_ref = p[P_POSITION]
    velocity_E_ref = p[P_VELOCITY]
    quaternion_WE_ref = p[P_QUATERNION]
    omega_E_ref = p[P_OMEGA]

    alpha_ref = p[P_ALPHA]
    thrust_ref = p[P_THRUST]

    attitude_error_E = (
        quaternion_error_vector(
            quaternion_WE,
            quaternion_WE_ref,
        )
    )

    # ========================================================
    # Stage residual
    # ========================================================
    #
    #   tool position              3
    #   tool velocity              3
    #   tool attitude              3
    #   tool angular velocity      3
    #   physical servo angle       4
    #   thrust magnitude           4
    #   servo command movement     4
    #
    # Total: 24

    y = ca.vertcat(
        position_E_W - position_E_ref,
        velocity_E_W - velocity_E_ref,
        attitude_error_E,
        omega_E - omega_E_ref,
        alpha - alpha_ref,
        thrust - thrust_ref,
        alpha_cmd - alpha,
    )

    # Terminal residual has no control input.
    #
    # Total: 16

    y_terminal = ca.vertcat(
        position_E_W - position_E_ref,
        velocity_E_W - velocity_E_ref,
        attitude_error_E,
        omega_E - omega_E_ref,
        alpha - alpha_ref,
    )

    W = np.diag(
        np.concatenate(
            [
                Q_EFFECTOR_POSITION,
                Q_EFFECTOR_VELOCITY,
                Q_ATTITUDE,
                Q_OMEGA,
                Q_ALPHA,
                R_THRUST,
                R_EFFECTOR_DELTA_ALPHA,
            ]
        )
    )

    W_terminal = np.diag(
        np.concatenate(
            [
                Q_EFFECTOR_POSITION,
                Q_EFFECTOR_VELOCITY,
                Q_ATTITUDE,
                Q_OMEGA,
                Q_ALPHA,
            ]
        )
    )

    ocp = AcadosOcp()
    ocp.model = model

    # --------------------------------------------------------
    # Nonlinear least-squares cost
    # --------------------------------------------------------

    ocp.cost.cost_type = (
        "NONLINEAR_LS"
    )

    ocp.cost.cost_type_e = (
        "NONLINEAR_LS"
    )

    ocp.model.cost_y_expr = y
    ocp.model.cost_y_expr_e = (
        y_terminal
    )

    ocp.cost.W = W
    ocp.cost.W_e = W_terminal

    ocp.cost.yref = np.zeros(
        int(
            y.shape[0]
        )
    )

    ocp.cost.yref_e = np.zeros(
        int(
            y_terminal.shape[0]
        )
    )

    # --------------------------------------------------------
    # Input constraints
    # --------------------------------------------------------

    ocp.constraints.idxbu = np.arange(
        NU,
        dtype=int,
    )

    ocp.constraints.lbu = np.concatenate(
        [
            np.zeros(4),
            np.full(
                4,
                -SERVO_COMMAND_LIMIT,
            ),
        ]
    )

    ocp.constraints.ubu = np.concatenate(
        [
            np.full(
                4,
                MAX_THRUST,
            ),
            np.full(
                4,
                SERVO_COMMAND_LIMIT,
            ),
        ]
    )

    # --------------------------------------------------------
    # State constraints
    # --------------------------------------------------------

    state_bound_indices = np.array(
        [
            3,
            4,
            5,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
        ],
        dtype=int,
    )

    state_lower = np.concatenate(
        [
            np.full(
                3,
                -1.0,
            ),
            np.full(
                3,
                -6.0,
            ),
            np.full(
                4,
                -SERVO_COMMAND_LIMIT,
            ),
        ]
    )

    state_upper = np.concatenate(
        [
            np.full(
                3,
                1.0,
            ),
            np.full(
                3,
                6.0,
            ),
            np.full(
                4,
                SERVO_COMMAND_LIMIT,
            ),
        ]
    )

    ocp.constraints.idxbx = (
        state_bound_indices
    )

    ocp.constraints.lbx = (
        state_lower
    )

    ocp.constraints.ubx = (
        state_upper
    )

    ocp.constraints.idxbx_e = (
        state_bound_indices
    )

    ocp.constraints.lbx_e = (
        state_lower
    )

    ocp.constraints.ubx_e = (
        state_upper
    )

    # Runtime replaces stage-0 bounds with the measured state.
    x0 = np.zeros(
        NX,
        dtype=float,
    )

    x0[6] = 1.0

    ocp.constraints.x0 = x0

    # --------------------------------------------------------
    # Default parameters used when generating the solver
    # --------------------------------------------------------

    parameter_values = np.zeros(
        NP,
        dtype=float,
    )

    parameter_values[P_QUATERNION] = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    parameter_values[P_THRUST] = np.full(
        4,
        HOVER_THRUST,
        dtype=float,
    )

    ocp.parameter_values = (
        parameter_values
    )

    # --------------------------------------------------------
    # Solver configuration
    # --------------------------------------------------------

    ocp.solver_options.N_horizon = (
        N_HORIZON
    )

    ocp.solver_options.tf = (
        PREDICTION_HORIZON
    )

    ocp.solver_options.integrator_type = (
        "ERK"
    )

    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1

    ocp.solver_options.qp_solver = (
        "PARTIAL_CONDENSING_HPIPM"
    )

    ocp.solver_options.nlp_solver_type = (
        "SQP_RTI"
    )

    ocp.solver_options.hessian_approx = (
        "GAUSS_NEWTON"
    )

    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.nlp_solver_warm_start_first_qp = True
    ocp.solver_options.nlp_solver_warm_start_first_qp_from_nlp = True

    ocp.solver_options.print_level = 0

    if code_export_directory is not None:
        ocp.code_gen_options.code_export_directory = str(
            Path(
                code_export_directory
            ).resolve()
        )

    return ocp
