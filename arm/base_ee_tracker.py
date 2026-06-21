"""Continuously write T_base_ee from the UFACTORY xArm TCP pose."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from arm.xarm_controller import DEFAULT_ENV_FILE, XArmCommandError, XArmController


DEFAULT_OUTPUT = Path("calibration/base_ee.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("xarm", "file", "stdin"),
        default="xarm",
        help="pose source to track (default: xarm)",
    )
    parser.add_argument("--pose-file", type=Path, help="YAML/JSON pose file for --source file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--ip", help="override XARM_IP from .env for --source xarm")
    parser.add_argument("--base-frame", default="base")
    parser.add_argument("--ee-frame", default="xarm_tcp")
    parser.add_argument(
        "--interval",
        type=positive_float,
        default=0.05,
        help="poll/write interval in seconds (default: 0.05)",
    )
    parser.add_argument("--once", action="store_true", help="write one sample and exit")
    parser.add_argument("--pretty", action="store_true", help="also print each output YAML")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.source == "xarm":
            return track_xarm(args)
        if args.source == "stdin":
            return track_stdin(args)
        return track_file(args)
    except (
        FileNotFoundError,
        ImportError,
        RuntimeError,
        ValueError,
        XArmCommandError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as exc:
        print(f"base-ee-tracker: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


def track_xarm(args: argparse.Namespace) -> int:
    controller = XArmController(ip=args.ip, env_file=args.env_file)
    controller.connect()
    try:
        while True:
            sample = sample_from_xarm_tcp_pose(controller.get_tcp_pose())
            write_sample(args, sample, source="xarm.get_tcp_pose")
            if args.once:
                return 0
            time.sleep(args.interval)
    finally:
        controller.disconnect()


def track_file(args: argparse.Namespace) -> int:
    if args.pose_file is None:
        raise ValueError("--source file requires --pose-file")
    while True:
        write_sample(args, load_pose_sample(args.pose_file), source=str(args.pose_file))
        if args.once:
            return 0
        time.sleep(args.interval)


def track_stdin(args: argparse.Namespace) -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        write_sample(args, parse_pose_data(yaml.safe_load(line)), source="stdin")
        if args.once:
            return 0
    return 0


def write_sample(args: argparse.Namespace, sample: dict[str, Any], source: str) -> None:
    document = base_ee_document(sample, args.base_frame, args.ee_frame, source=source)
    serialized = yaml.safe_dump(document, sort_keys=False)
    atomic_write(args.output, serialized)
    if args.pretty:
        print(serialized, end="")


def sample_from_xarm_tcp_pose(tcp_pose: list[float]) -> dict[str, Any]:
    if len(tcp_pose) != 6:
        raise ValueError("xArm TCP pose must contain [x, y, z, roll, pitch, yaw]")
    x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = [float(value) for value in tcp_pose]
    translation_m = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
    rotation = rpy_to_rotation(np.radians([roll_deg, pitch_deg, yaw_deg]))
    return {
        "matrix": make_transform(rotation, translation_m),
        "tcp_pose_mm_deg": [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg],
    }


def load_pose_sample(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        if path.suffix.lower() == ".json":
            data = json.load(file)
        else:
            data = yaml.safe_load(file)
    return parse_pose_data(data)


def parse_pose_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("pose source must contain a mapping")

    matrix = transform_matrix_from_data(data)
    if matrix is not None:
        return {"matrix": matrix}

    if "tcp_pose_mm_deg" in data:
        return sample_from_xarm_tcp_pose(list(data["tcp_pose_mm_deg"]))

    position = vector3_from_data(data, "position_m", ("x", "y", "z"))
    quaternion = quaternion_from_data(data)
    if quaternion is not None:
        return {"matrix": make_transform(quaternion_to_rotation(quaternion), position)}

    rpy = vector3_from_data(data, "rpy_rad", ("roll", "pitch", "yaw"), required=False)
    if rpy is None:
        rpy = vector3_from_data(data, "orientation_rpy_rad", ("roll", "pitch", "yaw"), required=False)
    if rpy is not None:
        return {"matrix": make_transform(rpy_to_rotation(rpy), position)}

    raise ValueError(
        "pose source must contain T_base_ee, tcp_pose_mm_deg, or position_m plus quaternion/RPY"
    )


def transform_matrix_from_data(data: dict[str, Any]) -> np.ndarray | None:
    for key in ("T_base_ee", "transform_matrix", "matrix", "T"):
        if key not in data:
            continue
        matrix = np.asarray(data[key], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"{key} must be a 4x4 matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{key} must contain only finite numbers")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError(f"{key} last row must be [0, 0, 0, 1]")
        return matrix
    return None


def vector3_from_data(
    data: dict[str, Any],
    key: str,
    component_names: tuple[str, str, str],
    required: bool = True,
) -> np.ndarray | None:
    if key in data:
        value = data[key]
        if isinstance(value, dict):
            vector = [value[name] for name in component_names]
        else:
            vector = value
    elif all(name in data for name in component_names):
        vector = [data[name] for name in component_names]
    elif required:
        raise ValueError(f"pose source must contain {key}")
    else:
        return None

    array = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{key} must contain only finite numbers")
    return array


def quaternion_from_data(data: dict[str, Any]) -> np.ndarray | None:
    for key in ("orientation_quaternion_xyzw", "quaternion_xyzw", "q_xyzw"):
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, dict):
            quaternion = [value[name] for name in ("x", "y", "z", "w")]
        else:
            quaternion = value
        return validate_quaternion(quaternion, key)

    if all(key in data for key in ("qx", "qy", "qz", "qw")):
        return validate_quaternion([data["qx"], data["qy"], data["qz"], data["qw"]], "quaternion")
    return None


def validate_quaternion(value: Any, name: str) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{name} must be a non-zero finite quaternion")
    return quaternion / norm


def quaternion_to_rotation(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = validate_quaternion(quaternion_xyzw, "quaternion")
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rpy_to_rotation(rpy_rad: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy_rad, dtype=np.float64).reshape(3)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def rotation_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif index == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    return validate_quaternion([x, y, z, w], "rotation quaternion").tolist()


def base_ee_document(
    sample: dict[str, Any],
    base_frame: str,
    ee_frame: str,
    source: str | None = None,
) -> dict[str, Any]:
    matrix = np.asarray(sample["matrix"], dtype=np.float64).reshape(4, 4)
    translation = matrix[:3, 3]
    quaternion = rotation_to_quaternion_xyzw(matrix[:3, :3])
    document: dict[str, Any] = {
        "timestamp": time.time(),
        "frames": {"parent": base_frame, "child": ee_frame},
        "translation_m": {
            "x": float(translation[0]),
            "y": float(translation[1]),
            "z": float(translation[2]),
        },
        "rotation_quaternion_xyzw": {
            "x": float(quaternion[0]),
            "y": float(quaternion[1]),
            "z": float(quaternion[2]),
            "w": float(quaternion[3]),
        },
        "T_base_ee": matrix.tolist(),
    }
    if source is not None:
        document["source"] = source
    if "tcp_pose_mm_deg" in sample:
        document["xarm_tcp_pose_mm_deg"] = sample["tcp_pose_mm_deg"]
    return document


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as file:
        file.write(content)
        temp_path = Path(file.name)
    temp_path.replace(path)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
