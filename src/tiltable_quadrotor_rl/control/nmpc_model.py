from __future__ import annotations

import casadi as ca
import numpy as np

from tiltable_quadrotor_rl.control.allocation import allocation_matrix
from tiltable_quadrotor_rl.robot.tilt_servo import DEFAULT_TIME_CONSTANT

from tiltable_quadrotor_rl.robot.parameters import (
    MASS,
    GRAVITY,
    INERTIA_DIAG,
)

NX = 17
NU = 8

def quaternion_to_rotation(q):
    """
    Rotation matrix from body frame B to world frame W.

    Quaternion convention:
        q = [qw, qx, qy, qz]
    """

    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    return ca.vertcat(
        ca.horzcat(
            1 - 2 * (qy**2 + qz**2),
            2 * (qx * qy - qw * qz),
            2 * (qx * qz + qw * qy),
        ),
        ca.horzcat(
            2 * (qx * qy + qw * qz),
            1 - 2 * (qx**2 + qz**2),
            2 * (qy * qz - qw * qx),
        ),
        ca.horzcat(
            2 * (qx * qz - qw * qy),
            2 * (qy * qz + qw * qx),
            1 - 2 * (qx**2 + qy**2),
        ),
    )

def quaternion_derivative(q, omega):
    """
    Quaternion derivative for body angular velocity omega.

        q_dot = 0.5 * q ⊗ [0, omega]
    """

    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    wx = omega[0]
    wy = omega[1]
    wz = omega[2]

    return 0.5 * ca.vertcat(
        -qx * wx - qy * wy - qz * wz,
         qw * wx + qy * wz - qz * wy,
         qw * wy + qz * wx - qx * wz,
         qw * wz + qx * wy - qy * wx,
    )


def rotor_wrench(thrust, alpha):
    """
    Compute body-frame wrench:

        [Fx, Fy, Fz, Tx, Ty, Tz]

    from physical rotor thrusts and physical servo angles.
    """

    z = ca.vertcat(
        thrust[0] * ca.sin(alpha[0]),
        thrust[0] * ca.cos(alpha[0]),

        thrust[1] * ca.sin(alpha[1]),
        thrust[1] * ca.cos(alpha[1]),

        thrust[2] * ca.sin(alpha[2]),
        thrust[2] * ca.cos(alpha[2]),

        thrust[3] * ca.sin(alpha[3]),
        thrust[3] * ca.cos(alpha[3]),
    )

    A = ca.DM(allocation_matrix())

    return A @ z

def continuous_dynamics(x, u):
    """
    Continuous-time tiltable-quadrotor model.

    State:
        x = [
            position_W(3),
            velocity_W(3),
            quaternion_WB(4),
            omega_B(3),
            alpha(4),
        ]

    input:
        U = [
            thrust(4),
            alpha_cmd(4),
        ]
    """

    velocity = x[3:6]
    q = x[6:10]
    omega = x[10:13]
    alpha = x[13:17]

    thrust = u[0:4]
    alpha_cmd = u[4:8]

    wrench = rotor_wrench(
        thrust,
        alpha,
    )

    force_B = wrench[0:3]
    torque_B = wrench[3:6]

    # Translational dynamics

    position_dot = velocity

    R_WB = quaternion_to_rotation(q)

    gravity_W = ca.DM(
        [0.0, 0.0, -GRAVITY]
    )

    velocity_dot = (
        R_WB @ force_B / MASS
        + gravity_W
    )

    # Rotational kinematics

    quaternion_dot = quaternion_derivative(q, omega)

    # Rotational rigid-body dynamics
    #
    # I * omega_dot = tau - omega x (I omega)

    Ixx = INERTIA_DIAG[0]
    Iyy = INERTIA_DIAG[1]
    Izz = INERTIA_DIAG[2]

    I_omega = ca.vertcat(
        Ixx * omega[0],
        Iyy * omega[1],
        Izz * omega[2],
    )

    gyro = ca.vertcat(
        omega[1] * I_omega[2]
        - omega[2] * I_omega[1],

        omega[2] * I_omega[0]
        - omega[0] * I_omega[2],

        omega[0] * I_omega[1]
        - omega[1] * I_omega[0],
    )

    omega_dot = ca.vertcat(
        (torque_B[0] - gyro[0]) / Ixx,
        (torque_B[1] - gyro[1]) / Iyy,
        (torque_B[2] - gyro[2]) / Izz,
    )

    # Sevo dynamics
    #
    # alpha_dot = (alpha_cmd - alpha) / tau

    alpha_dot = (
        alpha_cmd - alpha
    ) / DEFAULT_TIME_CONSTANT

    return ca.vertcat(
        position_dot,
        velocity_dot,
        quaternion_dot,
        omega_dot,
        alpha_dot,
    )


def build_dynamics_function():
    x = ca.SX.sym(
        "x",
        NX,
    )

    u = ca.SX.sym(
        "u",
        NU,
    )

    x_dot = continuous_dynamics(x, u)

    return ca.Function(
        "tiltable_quadrotor_dynamics",
        [x, u],
        [x_dot],
        ["x", "u"],
        ["x_dot"],
    )

def build_wrench_function():
    thrust = ca.SX.sym(
        "thrust",
        4,
    )

    alpha = ca.SX.sym(
        "alpha",
        4,
    )

    wrench = rotor_wrench(
        thrust,
        alpha,
    )

    return ca.Function(
        "tiltable_quadrotor_wrench",
        [thrust, alpha],
        [wrench],
        ["thrust", "alpha"],
        ["wrench"],
    )
