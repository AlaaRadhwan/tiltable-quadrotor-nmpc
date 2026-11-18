from __future__ import annotations

from pathlib import Path

from tiltable_quadrotor_rl.control.nmpc_controller import (
    NmpcController,
)
from tiltable_quadrotor_rl.control.nmpc_ocp import (
    CONTROL_PERIOD,
    N_HORIZON,
    PREDICTION_HORIZON,
    PREDICTION_STEP,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

BUILD_DIR = (
    PROJECT_ROOT
    / "build"
    / "acados"
    / "tiltable_quadrotor_nmpc"
)

JSON_FILE = (
    BUILD_DIR
    / "tiltable_quadrotor_nmpc.json"
)


def main() -> None:

    print()
    print("=" * 76)
    print("BUILD GENERAL TILTABLE-QUADROTOR NMPC")
    print("=" * 76)

    print()
    print(
        f"Controller period : "
        f"{CONTROL_PERIOD:.3f} s "
        f"({1.0 / CONTROL_PERIOD:.0f} Hz)"
    )

    print(
        f"OCP stage spacing : "
        f"{PREDICTION_STEP:.3f} s"
    )

    print(
        f"OCP stages        : "
        f"{N_HORIZON}"
    )

    print(
        f"OCP horizon       : "
        f"{PREDICTION_HORIZON:.3f} s"
    )

    print()
    print("Build directory:")
    print(BUILD_DIR)

    print()
    print(
        "Generating and compiling the general acados solver..."
    )

    controller = NmpcController(
        build_directory=BUILD_DIR,
        json_file=JSON_FILE,
        generate=True,
        build=True,
        verbose=True,
    )

    del controller

    print()
    print("=" * 76)
    print("GENERAL NMPC SOLVER BUILD PASSED")
    print("=" * 76)


if __name__ == "__main__":
    main()
