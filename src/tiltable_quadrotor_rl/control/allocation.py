from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, lsq_linear

from tiltable_quadrotor_rl.robot.parameters import (
    ARM_LENGTH,
    KQ_OVER_KT,
    MAX_THRUST,
    SERVO_COMMAND_LIMIT,
)

MAX_COMMAND_ANGLE = SERVO_COMMAND_LIMIT

THRUST_TOLERANCE = 1e-6

def allocation_matrix(
        arm_length: float = ARM_LENGTH,
        kq_over_kt: float = KQ_OVER_KT,
) -> np.ndarray:
    """Return the 6x8 virtual-input allocation matrix.
    
    Virtual input ordering:

        z = [
            h1, v1
            h2, v2,
            h3, v3,
            h4, v4
        ]

    Wrench ordering:

        w = [
            Fx, Fy, Fz,
            Tx, Ty, Tz
        ]
    """

    l = arm_length
    gamma = kq_over_kt

    return np.array(
        [
            [0,  0,  1,  0,  0,  0, -1,  0],
            [-1, 0,  0,  0,  1,  0,  0,  0],
            [0,  1,  0,  1,  0,  1,  0,  1],

            [0,  0,  gamma,  l,  0,  0, -gamma, -l],
            [gamma, -l,  0,  0, -gamma,  l,  0,  0],
            [-l, -gamma, -l, gamma, -l, -gamma, -l, gamma],
        ],
        dtype=float,
    )

def rotor_commands_to_virtual(
    thrust: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Convert physical rotor commands into virtual force components.

    For each rotor:

        h_i = f_i * sin(alpha_i)
        v_i = f_i * cos(alpha_i)

    Returns:

        z = [
            h1, v1,
            h2, v2,
            h3, v3,
            h4, v4,
        ]
    """

    thrust = np.asarray(
        thrust,
        dtype=float,
    )

    alpha = np.asarray(
        alpha,
        dtype=float,
    )

    if thrust.shape != (4,):
        raise ValueError(
            f"Expected thrust shape (4,), got {thrust.shape}"
        )

    if alpha.shape != (4,):
        raise ValueError(
            f"Expected alpha shape (4,), got {alpha.shape}"
        )

    h = thrust * np.sin(alpha)
    v = thrust * np.cos(alpha)

    z = np.empty(8, dtype=float)

    z[0::2] = h
    z[1::2] = v

    return z

def virtual_to_wrench(
    z: np.ndarray,
) -> np.ndarray:
    """Convert virtual rotor force components into a body wrench."""

    z = np.asarray(
        z, dtype=float,
    )

    if z.shape != (8,):
        raise ValueError(
            f"Expected z shape (8,), got {z.shape}"
        )

    return allocation_matrix() @ z

def wrench_to_virtual(
    wrench: np.ndarray,
) -> np.ndarray:
    """
    Compute the minimum-norm virtual actuator solution
    fro a desired body wrench.

    Solves:
        A z = wrench

    using the Moore-Penrose pseudoinverse.
    """

    wrench = np.asarray(
        wrench,
        dtype=float,
    )

    if wrench.shape != (6,):
        raise ValueError(
            f"Expected wrench shape (6,), got {wrench.shape}"
        )

    A = allocation_matrix()

    return np.linalg.pinv(A) @ wrench

def virtual_to_rotor_commands(
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert virtual force components into physical rotor commands.

    Input ordering:

        z = [
            h1, v1,
            h2, v2,
            h3, v3,
            h4, v4,
        ]

    Returns
    -------
    thrust:
        Rotor thrust magnitudes [N], shape (4,)

    alpha:
        Rotor tilt angles [rad], shape (4,)
    """

    z = np.asarray(
        z,
        dtype=float,
    )

    if z.shape != (8,):
        raise ValueError(
            f"Expected z shape (8,), got {z.shape}"
        )

    h = z[0::2]
    v = z[1::2]

    thrust = np.sqrt(
        h**2 + v**2
    )

    alpha = np.arctan2(
        h,
        v,
    )

    return thrust, alpha

def check_rotor_commands(
    thrust: np.ndarray,
    alpha: np.ndarray,
    *,
    max_thrust: float = MAX_THRUST,
    max_command_angle: float = MAX_COMMAND_ANGLE,
) -> tuple[bool, dict]:
    """
    Check whether physical rotor commands satisfy actuator limits.
    """

    thrust = np.asarray(
        thrust,
        dtype=float,
    )

    alpha = np.asarray(
        alpha,
        dtype=float,
    )

    if thrust.shape != (4,):
        raise ValueError(
            f"Expected thrust shape (4,), got {thrust.shape}"
        )

    if alpha.shape != (4,):
        raise ValueError(
            f"Expected alpha shape (4,), got {alpha.shape}"
        )

    thrust_ok = np.all(
        (thrust >= 0.0)
        & (thrust <= max_thrust)
    )

    angle_ok = np.all(
        np.abs(alpha) <= max_command_angle
    )

    feasible = bool(
        thrust_ok and angle_ok
    )

    info = {
        "feasible": feasible,
        "thrust_ok": bool(thrust_ok),
        "angle_ok": bool(angle_ok),
        "max_thrust_used": float(np.max(thrust)),
        "max_abs_angle": float(np.max(np.abs(alpha))),
    }

    return feasible, info


def allocate_exact_wrench(
        wrench: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Allocate a desired 6-D body wrench using null-space
    optimization.

    The requested wrench is preserved exactly.

    Returns
    -------
    thrust:
        Rotor thrust magnitudes [N], shape (4,)

    alpha:
        Rotor tilt commands [rad], shape (4,)

    info:
        Diagnostic information about optimization and
        actuator feasibility.
    """

    wrench = np.asarray(
        wrench,
        dtype=float,
    )

    if wrench.shape != (6,):
        raise ValueError(
            f"Expected wrench shape (6,), got {wrench.shape}"
        )

    (
        z,
        c,
        thrust,
        alpha,
        result,
    ) = minimize_peak_thrust_slsqp(
        wrench
    )

    peak_thrust = float(
        np.max(thrust)
    )

    thrust_ok = bool(
        peak_thrust
        <= MAX_THRUST + THRUST_TOLERANCE
    )

    angle_ok = bool(
        np.all(
            np.abs(alpha)
            <= MAX_COMMAND_ANGLE
        )
    )

    feasible = bool(
        thrust_ok
        and angle_ok
    )

    info = {
        "feasible": feasible,
        "thrust_ok": thrust_ok,
        "angle_ok": angle_ok,

        "peak_thrust": peak_thrust,
        "thrust_limit": MAX_THRUST,

        "c": c,
        "virtual_input": z,

        "optimization_success": bool(
            result.success
        ),

        "optimization_message": (
            result.message
        ),

        "iterations": int(
            result.nit
        ),

        "function_evaluations": int(
            result.nfev
        ),
    }

    return (
        thrust,
        alpha,
        info,
    )


def null_space_basis(
    A: np.ndarray | None = None,
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """
    Compute an orthonormal basis for the null space of A.

    For our 6x8 full-row-rank allocation matrix, the result
    should have shape (8, 2).

    Each column n satisfies:

        A @ n = 0
    """

    if A is None:
        A = allocation_matrix()

    A = np.asarray(
        A,
        dtype=float,
    )

    # Singular Value Decomposition:
    #
    #   A = U S V^T
    #
    # The trailing columns of V span the null space.
    U, singular_values, Vt = np.linalg.svd(
        A,
        full_matrices=True,
    )

    rank = np.sum(
        singular_values > tolerance
    )

    N = Vt[rank:].T

    return N


def minimize_peak_thrust_grid(
        wrench: np.ndarray,
        *,
        search_limit: float = 25.0,
        grid_points: int = 401,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Search the 2-D null space for a solution that minimizes
    the largest rotor thrust.

    This is a diagnostic/reference implementation, not the
    intended final real-time optimizer.

    Returns
    -------
    z_best;
        Optimized virtual actuator vector, shape (8,)
    
    c_best:
        Best null-space coordinates [ c1, c2], shape (2,)
    
    thrust_best:
        Rotor thrust magnitudes, shape (4,)
    
    alpha_best:
        Rotor tilt angles, shape (4,)
    """

    wrench = np.asarray(
        wrench,
        dtype=float,
    )

    if wrench.shape != (6,):
        raise ValueError(
            f"Expected wrench shape (6,), got {wrench.shape}"
        )

    A = allocation_matrix()

    z0 = wrench_to_virtual(wrench)

    N = null_space_basis(A)

    values = np.linspace(
        -search_limit,
        search_limit,
        grid_points
    )

    best_peak = np.inf
    best_z = None
    best_c = None
    best_thrust = None
    best_alpha = None

    for c1 in values:

        # Evaluate every c2 for this c1 at once.
        Z = (
            z0[:, None]
            + N[:, 0, None] * c1
            + N[:, 1, None] * values[None, :]
        )

        H = Z[0::2, :]
        V = Z[1::2, :]

        thrust = np.sqrt(
            H**2 + V**2
        )

        peak = np.max(
            thrust,
            axis=0,
        )

        j = np.argmin(peak)

        if peak[j] < best_peak:

            best_peak = peak[j]

            best_z = Z[:, j].copy()

            best_c = np.array(
                [
                    c1,
                    values[j]
                ]
            )

            best_thrust, best_alpha = (
                virtual_to_rotor_commands(
                    best_z
                )
            )

    return (
        best_z,
        best_c,
        best_thrust,
        best_alpha,
    )

def minimize_peak_thrust_slsqp(
        wrench: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Minimize the maximum rotor thrust using the 2-D null space.

    Decision variables:
        x = [c1, c2, t]
    
    where:
        z = z0 + N @ [c1, c2]
    
    and t is an upper bound on all four rotor thrusts.

    The optimization problem is:

        minimize t

        subject to f_i <= t, for i = 1..4
    
    Returns:
        z_best
        c_best
        thrust_best
        alpha_best
        scipy_result
    """

    wrench = np.asarray(
        wrench,
        dtype=float
    )

    if wrench.shape != (6,):
        raise ValueError(
            f"Expected wrench shape (6,), got {wrench.shape}"
        )

    A = allocation_matrix()

    z0 = wrench_to_virtual(wrench)

    N = null_space_basis(A)

    # Helper: x -> virtual actoator vector z

    def z_from_x(x):
        c = x[:2]

        return (z0 + N @ c)

    # Helper: x -> four physical rotor thrusts

    def thrust_from_x(x):
        z = z_from_x(x)

        h = z[0::2]
        v = z[1::2]

        return np.sqrt(h**2 + v**2)
    
    # Objective

    def objective(x):
        return x[2]

    # Constraints 
    #
    # SciPy wants inequality constraints in the form:
    #
    #       constraint(x) >= 0
    #
    # We want:
    #       f_i <= t
    # which is equivalent to:
    #
    #       t - f_i >= 0

    def thrust_ceiling_constraint(x):
        t = x[2]

        thrust = thrust_from_x(x)

        return (t - thrust)

    # Initial guess

    thrust0, _ = virtual_to_rotor_commands(z0)

    t0 = float(np.max(thrust0))

    x0 = np.array(
        [
            0.0,
            0.0,
            t0,
        ]
    )

    # Run SLSQP

    result = minimize(
        fun=objective,
        x0=x0,
        method="SLSQP",
        bounds=[
            (None, None),
            (None, None),
            (0.0, None),
        ],
        constraints=[
            {
                "type": "ineq",
                "fun": thrust_ceiling_constraint,
            }
        ],
        options={
            "ftol": 1e-10,
            "maxiter": 200,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            "SLSQP optimization failed: " 
            + result.message
        )

    # Recover final solution

    c_best = result.x[:2]

    z_best = (
        z0 + N @ c_best
    )

    thrust_best, alpha_best = virtual_to_rotor_commands(z_best)

    return (
        z_best,
        c_best,
        thrust_best,
        alpha_best,
        result
    )


def maximize_feasible_wrench_scale(
        desired_wrench: np.ndarray,
        reference_wrench: np.ndarray,
) -> tuple[np.ndarray, float, dict]:

    """
    Find the largest scale s in [0, 1] such that

        achieved_wrench = 
            reference_wrench
            + s * (desired_wrench - reference_wrench)

    can be produced without exceeding MAX_THRUST.

    Returns:
        achieved_thrust,
        scale,
        diagnostic info
    """

    desired_wrench = np.asarray(
        desired_wrench,
        dtype=float,
    )

    reference_wrench = np.asarray(
        reference_wrench,
        dtype=float,
    )

    if desired_wrench.shape != (6,):
        raise ValueError(
            "desired_wrench must have shape (6,)"
        )

    if reference_wrench.shape != (6,):
        raise ValueError(
            "reference_wrench must have shape (6,)"
        )

    A = allocation_matrix()

    N = null_space_basis(
        A
    )

    # Desired change away from the reference wrench

    delta_wrench = (
        desired_wrench
        - reference_wrench
    )

    # Because the pseudoinverse is linear:
    #
    # z(s) = 
    #       z_reference + s * z_delta + N @ c

    z_reference = wrench_to_virtual(
        reference_wrench
    )

    z_delta = wrench_to_virtual(
        delta_wrench
    )

    # First make sure the reference wrench itself is feasible
    #
    # We also get a good starting null-space coordinate.

    (
        _,
        _,
        reference_info,
    ) = allocate_exact_wrench(reference_wrench)

    if not reference_info["feasible"]:
        raise ValueError(
            "Reference wrench itself is infeasible."
        )

    c_reference = reference_info["c"]

    # Decision vector:
    #
    # x = [s, c1, c2]

    def z_from_x(x):
        scale = x[0]
        c = x[1:3]

        return (
            z_reference
            + scale * z_delta
            + N @ c
        )

    def thrust_from_x(x):
        z = z_from_x(x)

        h = z[0::2]
        v = z[1::2]

        return np.sqrt(
            h**2 + v**2
        )


    # We want:
    #
    # maximize s
    #
    # SciPy minimizes, so minimize -s.

    def objective(x):
        return -x[0]

    # --------------------------------------------------------
    # Hard actuator constraint:
    #
    # f_i <= MAX_THRUST
    #
    # SciPy inequality convention:
    #
    # MAX_THRUST - f_i >= 0
    # --------------------------------------------------------

    def thrust_constraint(x):
        thrust = thrust_from_x(x)

        return (
            MAX_THRUST
            - thrust
        )

    # --------------------------------------------------------
    # Start at s = 0.
    #
    # This means:
    #
    # achieved_wrench = reference_wrench
    #
    # which we already know is feasible.
    # --------------------------------------------------------

    x0 = np.array(
        [
            0.0,
            c_reference[0],
            c_reference[1],
        ],
        dtype=float,
    )

    result = minimize(
        fun=objective,
        x0=x0,
        method="SLSQP",
        bounds=[
            (0.0, 1.0),
            (None, None),
            (None, None),
        ],
        constraints=[
            {
                "type": "ineq",
                "fun": thrust_constraint,
            }
        ],
        options={
            "ftol": 1e-10,
            "maxiter": 200,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Wrench scaling optimization failed: "
            + result.message
        )

    scale = float(
        result.x[0]
    )

    achieved_wrench = (
        reference_wrench
        + scale * delta_wrench
    )

    info = {
        "scale": scale,

        "c_scaling": (
            result.x[1:3].copy()
        ),

        "iterations": int(
            result.nit
        ),

        "function_evaluations": int(
            result.nfev
        ),

        "optimization_success": bool(
            result.success
        ),

        "optimization_message": (
            result.message
        ),
    }

    return (
        achieved_wrench,
        scale,
        info,
    )


def allocate_wrench(
        desired_wrench: np.ndarray,
        reference_wrench: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:

    """
    Hierarchical wrench allocator.

    Stage 1:
        Try to produce the desired wrench exactly using
        minimum peak-thrust allocation.

    Stage 2:
        If that wrench is infeasible, find the maximum
        feasible scale from reference_wrench toward
        desired_wrench.

    Stage 3:
        Run the minimum-peak-thrust allocator again for
        the resulting achievable wrench.
    """


    desired_wrench = np.asarray(
        desired_wrench,
        dtype=float,
    )

    reference_wrench = np.asarray(
        reference_wrench,
        dtype=float,
    )

    # Stage 1: Try the complete desired wrench.

    (
        thrust_exact,
        alpha_exact,
        exact_info,
    ) = allocate_exact_wrench(desired_wrench)

    if exact_info["feasible"]:
        info = {
            "requested_wrench": (
                desired_wrench.copy()
            ),

            "achieved_wrench": (
                desired_wrench.copy()
            ),

            "scaled": False,
            "scale": 1.0,

            "initial_peak_thrust": (
                exact_info["peak_thrust"]
            ),

            "final_peak_thrust": (
                exact_info["peak_thrust"]
            ),

            "final_c": (
                exact_info["c"]
            ),

            "exact_optimizer": (
                exact_info
            ),

            "scaling_optimizer": None,
        }

        return (
            thrust_exact,
            alpha_exact,
            info,
        )

    # Stage 2: Desired wrench is impossible
    #          Find the largest feasible fraction.

    (
        achieved_wrench,
        scale,
        scaling_info,
    ) = maximize_feasible_wrench_scale(
        desired_wrench,
        reference_wrench
    )

    # Stage 3: Redistribute the achieved wrench optimally.

    (
        thrust_final,
        alpha_final,
        final_info
    ) = allocate_exact_wrench(achieved_wrench)

    if not final_info["feasible"]:
        raise RuntimeError(
            "Internal allocator error: scaled wrench "
            "should be feasible but final exact "
            "allocation is not."
        )

    info = {
        "requested_wrench": (
            desired_wrench.copy()
        ),

        "achieved_wrench": (
            achieved_wrench.copy()
        ),

        "scaled": True,
        "scale": scale,

        # How much thrust the impossible original
        # request would have needed.
        "initial_peak_thrust": (
            exact_info["peak_thrust"]
        ),

        # Peak thrust after scaling + final redistribution.
        "final_peak_thrust": (
            final_info["peak_thrust"]
        ),

        "final_c": (
            final_info["c"]
        ),

        "exact_optimizer": (
            exact_info
        ),

        "scaling_optimizer": (
            scaling_info
        ),

        "final_optimizer": (
            final_info
        ),
    }

    return (
        thrust_final,
        alpha_final,
        info,
    )

def thrust_allocation_matrix(
        alpha: np.ndarray,
) -> np.ndarray:
    """
    Build the instantaneous wrench map

        w = B(alpha) @ thrust

    for fixed current rotor tilt angles.

    B has shape (6, 4).
    """
    
    alpha = np.asarray(
        alpha,
        dtype=float
    )

    if alpha.shape != (4,):
        raise ValueError(
            f"Expected alpha shape (4,), got {alpha.shape}"
        )

    A = allocation_matrix()

    T = np.zeros(
        (8, 4),
        dtype=float,
    )

    for i in range(4):

        T[2 * i, i] = np.sin(
            alpha[i]
        )

        T[2* i + 1, i] = np.cos(
            alpha[i]
        )

    return A @ T

def allocate_thrust_at_fixed_angles(
        alpha: np.ndarray,
        desired_wrench: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Find the best rotor thrust magnitudes for the current,
    fixed rotor tilt angles.

    The rotor angles are NOT changed here

    Sovles a bounded weighted least-squares problem:

        minimize || W (B(alpha) f - w_desired) ||^2

        subject to:
            0 <= f_i <= MAX_THRUST
    """

    alpha = np.asarray(
        alpha,
        dtype=float
    )

    desired_wrench = np.asarray(
        desired_wrench,
        dtype=float
    )

    if alpha.shape != (4,):
        raise ValueError(
            f"Expected alpha shape (4,), got {alpha.shape}"
        )

    if desired_wrench.shape != (6,):
        raise ValueError(
            f"Expected wrench shape (6,), got {desired_wrench.shape}"
        )

    B = thrust_allocation_matrix(
        alpha
    )

    # Wrench normalization
    # Force errors remain in N.
    #
    # Torque erros are divided by the arm length,
    # giving them an approximate force-equivalent scale.

    W = np.diag([
        1.0,
        1.0,
        1.0,
        1.0 / ARM_LENGTH,
        1.0 / ARM_LENGTH,
        1.0 / ARM_LENGTH,
    ])

    B_weighted = (
        W @ B
    )

    wrench_weighted = (
        W @ desired_wrench
    )

    # Solve:
    #
    # min || B_weighted @ f - wrench_weighted ||^2
    #
    # while enforcing:
    #
    # 0 <= f_i <= 23 N

    result = lsq_linear(
        B_weighted,
        wrench_weighted,
        bounds=(
            0.0,
            MAX_THRUST,
        ),
        method="trf",
        tol=1e-10,
        max_iter=200,
    )

    thrust = result.x

    achieved_wrench = (
        B @ thrust
    )

    wrench_error = (
        achieved_wrench
        - desired_wrench
    )
    
    weighted_error = (
        W @ wrench_error
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

        "iterations": int(
            result.nit
        ),

        "achieved_wrench": (
            achieved_wrench
        ),

        "wrench_error": (
            wrench_error
        ),

        "weighted_error_norm": float(
            np.linalg.norm(
                weighted_error
            )
        ),

        "matrix_rank": int(
            np.linalg.matrix_rank(
                B
            )
        ),
    }

    return (
        thrust,
        achieved_wrench,
        info,
    )