from __future__ import annotations

from pathlib import Path

import casadi as ca
import numpy as np

from acados_template import (
    AcadosModel,
    AcadosOcp,
)

from tiltable_quadrotor_rl.control.nmpc_model import (
    NU,
    NX,
    continuous_dynamics,
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
# Controller timing
# ============================================================
#
# Paper baseline:
#   control loop:       0.01 s = 100 Hz
#   shooting interval:  0.10 s
#   shooting intervals: 20
#   prediction horizon: 2.0 s

CONTROL_PERIOD = 0.010
N_HORIZON = 20
PREDICTION_STEP = 0.100
PREDICTION_HORIZON = (
    N_HORIZON
    * PREDICTION_STEP
)


# ============================================================
# Paper baseline weights
# ============================================================

Q_POSITION = np.array(
    [
        300.0,
        300.0,
        400.0,
    ]
)

Q_VELOCITY = np.array(
    [
        10.0,
        10.0,
        10.0,
    ]
)

Q_ATTITUDE = np.array(
    [
        300.0,
        300.0,
        600.0,
    ]
)

Q_OMEGA = np.array(
    [
        5.0,
        5.0,
        5.0,
    ]
)

Q_ALPHA = np.full(
    4,
    2.0,
)

R_THRUST = np.full(
    4,
    2.0,
)

R_DELTA_ALPHA = np.full(
    4,
    250.0,
)


def quaternion_conjugate(
    quaternion: ca.SX,
) -> ca.SX:
    return ca.vertcat(
        quaternion[0],
        -quaternion[1],
        -quaternion[2],
        -quaternion[3],
    )


def quaternion_product(
    left: ca.SX,
    right: ca.SX,
) -> ca.SX:
    """
    Hamilton quaternion product for [w, x, y, z].
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


def quaternion_error_vector(
    quaternion: ca.SX,
    quaternion_ref: ca.SX,
) -> ca.SX:
    """
    Vector part of:
        q_error = conjugate(q_ref) * q

    Squaring this vector is invariant to q <-> -q.
    """

    error = quaternion_product(
        quaternion_conjugate(
            quaternion_ref
        ),
        quaternion,
    )

    return error[1:4]


def build_acados_model() -> AcadosModel:
    """
    Wrap the already-validated 17-state continuous model for acados.
    """

    x = ca.SX.sym(
        "x",
        NX,
    )

    u = ca.SX.sym(
        "u",
        NU,
    )

    xdot = ca.SX.sym(
        "xdot",
        NX,
    )

    p = ca.SX.sym(
        "p",
        NP,
    )

    f_expl = continuous_dynamics(
        x,
        u,
    )

    model = AcadosModel()

    model.name = (
        "tiltable_quadrotor_nmpc"
    )

    model.x = x
    model.u = u
    model.xdot = xdot
    model.p = p

    model.f_expl_expr = (
        f_expl
    )

    model.f_impl_expr = (
        xdot
        - f_expl
    )

    return model


def build_ocp(
    *,
    code_export_directory: str | Path | None = None,
) -> AcadosOcp:
    """
    Build one general 6-DoF trajectory-tracking NMPC.

    Hover, translation, rotation, and combined 6-DoF motion are
    references supplied to this same controller.
    """

    model = build_acados_model()

    x = model.x
    u = model.u
    p = model.p

    position = x[0:3]
    velocity = x[3:6]
    quaternion = x[6:10]
    omega = x[10:13]
    alpha = x[13:17]

    thrust = u[0:4]
    alpha_cmd = u[4:8]

    position_ref = p[P_POSITION]
    velocity_ref = p[P_VELOCITY]
    quaternion_ref = p[P_QUATERNION]
    omega_ref = p[P_OMEGA]
    alpha_ref = p[P_ALPHA]
    thrust_ref = p[P_THRUST]

    attitude_error = (
        quaternion_error_vector(
            quaternion,
            quaternion_ref,
        )
    )

    # Paper-style nonlinear least-squares residual:
    #
    # state errors:
    #   p - p_ref
    #   v - v_ref
    #   V(q_ref* * q)
    #   omega - omega_ref
    #   alpha - alpha_ref
    #
    # input errors:
    #   thrust - thrust_ref
    #   alpha_cmd - alpha
    #
    # The final term penalizes aggressive servo commands relative
    # to the actual physical servo state.

    y = ca.vertcat(
        position - position_ref,
        velocity - velocity_ref,
        attitude_error,
        omega - omega_ref,
        alpha - alpha_ref,
        thrust - thrust_ref,
        alpha_cmd - alpha,
    )

    y_terminal = ca.vertcat(
        position - position_ref,
        velocity - velocity_ref,
        attitude_error,
        omega - omega_ref,
        alpha - alpha_ref,
    )

    W = np.diag(
        np.concatenate(
            [
                Q_POSITION,
                Q_VELOCITY,
                Q_ATTITUDE,
                Q_OMEGA,
                Q_ALPHA,
                R_THRUST,
                R_DELTA_ALPHA,
            ]
        )
    )

    # The paper defines Q_N but does not publish a separate terminal
    # weight set in Table IV. Using the same state weights is a clear,
    # neutral baseline rather than inventing a second tuning set.
    W_terminal = np.diag(
        np.concatenate(
            [
                Q_POSITION,
                Q_VELOCITY,
                Q_ATTITUDE,
                Q_OMEGA,
                Q_ALPHA,
            ]
        )
    )

    ocp = AcadosOcp()
    ocp.model = model

    # --------------------------------------------------------
    # Cost
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

    ocp.cost.W_e = (
        W_terminal
    )

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
    #
    # u = [f1..f4, alpha_c1..alpha_c4]

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
    #
    # x:
    #   0:3    position
    #   3:6    world velocity
    #   6:10   quaternion
    #   10:13  body angular velocity
    #   13:17  physical servo angle
    #
    # Paper limits:
    #   |v|     <= 1 m/s
    #   |omega| <= 6 rad/s
    #   |alpha| <= 3.15 rad

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

    # acados has separate terminal-state bounds. Apply the same
    # physical limits at the final shooting node too.
    ocp.constraints.idxbx_e = (
        state_bound_indices
    )

    ocp.constraints.lbx_e = (
        state_lower
    )

    ocp.constraints.ubx_e = (
        state_upper
    )

    # Placeholder stage-0 state. At runtime, identical lower/upper
    # stage-0 bounds are replaced with the measured 17-state vector.
    x0 = np.zeros(
        NX,
        dtype=float,
    )

    x0[6] = 1.0

    ocp.constraints.x0 = x0

    # --------------------------------------------------------
    # Initial stage-parameter values
    # --------------------------------------------------------

    parameter_values = np.zeros(
        NP,
        dtype=float,
    )

    parameter_values[P_QUATERNION] = (
        np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        )
    )

    parameter_values[P_THRUST] = (
        np.full(
            4,
            HOVER_THRUST,
        )
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

    # The paper uses explicit RK4 with t_step = 0.1 s.
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

    # Keep the previous NLP iterate as the next linearization
    # trajectory. These options additionally allow HPIPM to reuse
    # QP information instead of forcing a cold first QP each cycle.
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
