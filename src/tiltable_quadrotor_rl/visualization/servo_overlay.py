from __future__ import annotations

import mujoco
import numpy as np


# Rotor locations in the body frame, inferred from the validated plant.
ROTOR_OFFSETS_B = np.array(
    [
        [ 0.275,  0.000, 0.0],   # rotor 1
        [ 0.000,  0.275, 0.0],   # rotor 2
        [-0.275,  0.000, 0.0],   # rotor 3
        [ 0.000, -0.275, 0.0],   # rotor 4
    ],
    dtype=float,
)

# Horizontal direction each rotor tilts toward as alpha increases.
#
# These match the validated geometry:
#   rotor 1 -> -Y
#   rotor 2 -> +X
#   rotor 3 -> +Y
#   rotor 4 -> -X
ROTOR_TILT_DIRS_B = np.array(
    [
        [ 0.0, -1.0, 0.0],
        [ 1.0,  0.0, 0.0],
        [ 0.0,  1.0, 0.0],
        [-1.0,  0.0, 0.0],
    ],
    dtype=float,
)

DISC_RADIUS = 0.055
DISC_HALF_THICKNESS = 0.004

ARROW_RADIUS = 0.0055
ARROW_LENGTH = 0.140

DISC_COLOR = np.array([0.10, 0.45, 1.00, 0.65], dtype=float)
ARROW_COLOR = np.array([1.00, 0.80, 0.10, 0.90], dtype=float)


def quaternion_wxyz_to_matrix(
    quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to a 3x3 rotation matrix.
    """
    q = np.asarray(
        quaternion_wxyz,
        dtype=float,
    ).reshape(4)

    q = q / np.linalg.norm(q)

    w, x, y, z = q

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def make_rotation_with_z_axis(
    z_axis_world: np.ndarray,
) -> np.ndarray:
    """
    Create an orthonormal rotation matrix whose local +Z axis equals
    z_axis_world.

    MuJoCo cylinder/capsule geoms are aligned with the local Z axis, so
    this lets us orient thin discs and arrows along the current servo
    direction.
    """
    z_axis = np.asarray(
        z_axis_world,
        dtype=float,
    ).reshape(3)

    z_axis = z_axis / np.linalg.norm(z_axis)

    # Use a helper vector that is not nearly parallel to z_axis.
    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(z_axis, helper))) > 0.95:
        helper = np.array([1.0, 0.0, 0.0], dtype=float)

    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    return np.column_stack(
        [
            x_axis,
            y_axis,
            z_axis,
        ]
    )


def rotor_direction_body(
    alpha: float,
    rotor_index: int,
) -> np.ndarray:
    """
    Thrust-axis direction of one rotor in the body frame.

    At alpha = 0, the rotor points along +Z_B.
    As alpha increases, it tilts toward the rotor's horizontal tilt
    direction.
    """
    horizontal = ROTOR_TILT_DIRS_B[rotor_index]
    vertical = np.array([0.0, 0.0, 1.0], dtype=float)

    direction = (
        np.sin(alpha) * horizontal
        + np.cos(alpha) * vertical
    )

    return direction / np.linalg.norm(direction)


def update_servo_overlay(
    viewer,
    *,
    body_position_W: np.ndarray,
    body_quaternion_WB: np.ndarray,
    alpha: np.ndarray,
) -> None:
    """
    Draw a viewer-only overlay showing the current physical tilt of the
    four servo-mounted rotors.

    This does NOT modify the plant or XML. It only adds custom geoms to
    the viewer's user scene:
      - a blue thin disc for each rotor plane,
      - a yellow arrow showing the rotor +Z / thrust direction.

    Assumptions:
      - Geom slot 0 may already be used by the fixed tool marker.
      - We therefore start at slot 1 and write 8 geoms total.
    """
    position_W = np.asarray(
        body_position_W,
        dtype=float,
    ).reshape(3)

    quaternion_WB = np.asarray(
        body_quaternion_WB,
        dtype=float,
    ).reshape(4)

    alpha = np.asarray(
        alpha,
        dtype=float,
    ).reshape(4)

    R_WB = quaternion_wxyz_to_matrix(
        quaternion_WB
    )

    geom_index = 1

    for rotor_index in range(4):
        rotor_offset_B = ROTOR_OFFSETS_B[rotor_index]
        rotor_position_W = (
            position_W
            + R_WB @ rotor_offset_B
        )

        direction_B = rotor_direction_body(
            float(alpha[rotor_index]),
            rotor_index,
        )

        direction_W = R_WB @ direction_B
        direction_W = direction_W / np.linalg.norm(direction_W)

        orientation = make_rotation_with_z_axis(
            direction_W
        )

        # -----------------------------------------------------------------
        # 1) Blue thin disc representing the rotor tilt plane / site marker.
        # -----------------------------------------------------------------
        disc_geom = viewer.user_scn.geoms[geom_index]
        mujoco.mjv_initGeom(
            disc_geom,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.array(
                [
                    DISC_RADIUS,
                    DISC_HALF_THICKNESS,
                    0.0,
                ],
                dtype=float,
            ),
            rotor_position_W,
            orientation.reshape(-1),
            DISC_COLOR,
        )
        geom_index += 1

        # -----------------------------------------------------------------
        # 2) Yellow capsule arrow showing the rotor +Z / thrust direction.
        # -----------------------------------------------------------------
        arrow_center_W = (
            rotor_position_W
            + 0.5 * ARROW_LENGTH * direction_W
        )

        arrow_geom = viewer.user_scn.geoms[geom_index]
        mujoco.mjv_initGeom(
            arrow_geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array(
                [
                    ARROW_RADIUS,
                    0.5 * ARROW_LENGTH,
                    0.0,
                ],
                dtype=float,
            ),
            arrow_center_W,
            orientation.reshape(-1),
            ARROW_COLOR,
        )
        geom_index += 1

    viewer.user_scn.ngeom = geom_index
