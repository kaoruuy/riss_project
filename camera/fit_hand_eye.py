"""Fit base-to-camera and end-effector-to-marker transforms from paired samples."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_SAMPLES = Path("calibration/hand_eye_samples.yaml")
DEFAULT_BASE_TO_CAMERA = Path("calibration/base_to_camera.yaml")
DEFAULT_EE_MARKER = Path("calibration/ee_marker_estimated.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--base-to-camera-output", type=Path, default=DEFAULT_BASE_TO_CAMERA)
    parser.add_argument("--ee-marker-output", type=Path, default=DEFAULT_EE_MARKER)
    parser.add_argument("--base-frame", default="base")
    parser.add_argument("--ee-frame", default="xarm_tcp")
    parser.add_argument("--camera-frame", default="zed_left_camera_optical_frame")
    parser.add_argument("--marker-frame", default="aruco_marker")
    parser.add_argument(
        "--method",
        choices=("shah", "li"),
        default="shah",
        help="OpenCV robot-world-hand-eye method (default: shah)",
    )
    parser.add_argument("--pretty", action="store_true", help="print estimated transforms")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        samples = load_samples(args.samples)
        result = fit_hand_eye(samples, method=args.method)
        save_base_to_camera(args.base_to_camera_output, result, samples, args)
        save_ee_marker(args.ee_marker_output, result, samples, args)
        if args.pretty:
            print(yaml.safe_dump(result_payload(result), sort_keys=False), end="")
        return 0
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(f"fit-hand-eye: {exc}", file=sys.stderr)
        return 1


def load_samples(path: Path) -> list[dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8") as file:
        if path.suffix.lower() == ".json":
            data = json.load(file)
        else:
            data = yaml.safe_load(file)

    if isinstance(data, dict):
        raw_samples = data.get("samples")
        if raw_samples is None and "transforms" in data:
            raw_samples = [data]
    else:
        raw_samples = data

    if not isinstance(raw_samples, list):
        raise ValueError(f"{path} must contain a list of samples or a 'samples' list")

    top_level_zed_settings = data.get("zed_settings") if isinstance(data, dict) else None
    if top_level_zed_settings is not None and not isinstance(top_level_zed_settings, dict):
        raise ValueError(f"{path} zed_settings must be a mapping")

    samples = [
        parse_sample(sample, index, top_level_zed_settings)
        for index, sample in enumerate(raw_samples, start=1)
    ]
    if len(samples) < 3:
        raise ValueError("at least 3 paired samples are required")
    return samples


def parse_sample(
    sample: Any,
    index: int,
    default_zed_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(sample, dict):
        raise ValueError(f"sample {index} must be a mapping")
    transforms = sample.get("transforms", sample)
    if not isinstance(transforms, dict):
        raise ValueError(f"sample {index} transforms must be a mapping")

    zed_settings = sample.get("zed_settings", default_zed_settings)
    if zed_settings is not None and not isinstance(zed_settings, dict):
        raise ValueError(f"sample {index} zed_settings must be a mapping")
    return {
        "T_base_ee": get_transform(transforms, "T_base_ee", index),
        "T_cam_marker": get_transform(transforms, "T_cam_marker", index),
        "zed_settings": zed_settings,
    }


def get_transform(transforms: dict[str, Any], key: str, index: int) -> np.ndarray:
    if key not in transforms:
        raise ValueError(f"sample {index} is missing {key}")
    return as_transform_matrix(transforms[key], f"sample {index} {key}")


def as_transform_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite numbers")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")
    return matrix


def fit_hand_eye(samples: list[dict[str, np.ndarray]], method: str = "shah") -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required; install opencv-contrib-python") from exc
    if not hasattr(cv2, "calibrateRobotWorldHandEye"):
        raise ImportError("OpenCV build does not include calibrateRobotWorldHandEye")

    method_id = {
        "shah": cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
        "li": cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI,
    }[method]

    convention_results = []
    for invert_cam_marker in (False, True):
        for invert_base_ee in (False, True):
            convention_results.append(
                fit_direction_convention(
                    cv2,
                    samples,
                    method_id,
                    method,
                    invert_cam_marker=invert_cam_marker,
                    invert_base_ee=invert_base_ee,
                )
            )

    best = min(convention_results, key=residual_score)
    best["tested_conventions"] = convention_summary(convention_results)
    return best


def fit_direction_convention(
    cv2: Any,
    samples: list[dict[str, np.ndarray]],
    method_id: int,
    method: str,
    invert_cam_marker: bool,
    invert_base_ee: bool,
) -> dict[str, Any]:
    t_world2cam_input = [
        invert_transform(sample["T_cam_marker"]) if invert_cam_marker else sample["T_cam_marker"]
        for sample in samples
    ]
    t_base2gripper_input = [
        invert_transform(sample["T_base_ee"]) if invert_base_ee else sample["T_base_ee"]
        for sample in samples
    ]

    r_world2cam = [transform[:3, :3] for transform in t_world2cam_input]
    p_world2cam = [transform[:3, 3].reshape(3, 1) for transform in t_world2cam_input]
    r_base2gripper = [transform[:3, :3] for transform in t_base2gripper_input]
    p_base2gripper = [transform[:3, 3].reshape(3, 1) for transform in t_base2gripper_input]

    r_x, p_x, r_z, p_z = cv2.calibrateRobotWorldHandEye(
        r_world2cam,
        p_world2cam,
        r_base2gripper,
        p_base2gripper,
        method=method_id,
    )

    x_transform = make_transform(r_x, p_x)
    z_transform = make_transform(r_z, p_z)
    mapping = best_output_mapping(samples, x_transform, z_transform)
    return {
        "T_base_cam": mapping["T_base_cam"],
        "T_ee_marker": mapping["T_ee_marker"],
        "opencv_X": x_transform,
        "opencv_Z": z_transform,
        "residuals": mapping["residuals"],
        "residual_score": mapping["score"],
        "method": method,
        "convention": {
            "T_cam_marker_input": "inverted" if invert_cam_marker else "as_given",
            "T_base_ee_input": "inverted" if invert_base_ee else "as_given",
            "T_base_cam_from": mapping["T_base_cam_from"],
            "T_ee_marker_from": mapping["T_ee_marker_from"],
        },
    }


def best_output_mapping(
    samples: list[dict[str, np.ndarray]],
    x_transform: np.ndarray,
    z_transform: np.ndarray,
) -> dict[str, Any]:
    candidates = {
        "opencv_X": x_transform,
        "inverse_opencv_X": invert_transform(x_transform),
        "opencv_Z": z_transform,
        "inverse_opencv_Z": invert_transform(z_transform),
    }
    best: dict[str, Any] | None = None
    for base_name, t_base_cam in candidates.items():
        for ee_name, t_ee_marker in candidates.items():
            residuals = residual_errors(samples, t_base_cam, t_ee_marker)
            score = residual_score({"residuals": residuals})
            candidate = {
                "T_base_cam": t_base_cam,
                "T_ee_marker": t_ee_marker,
                "residuals": residuals,
                "score": score,
                "T_base_cam_from": base_name,
                "T_ee_marker_from": ee_name,
            }
            if best is None or score < best["score"]:
                best = candidate
    assert best is not None
    return best


def residual_score(result: dict[str, Any]) -> float:
    residuals = result["residuals"]
    return float(residuals["translation_m"]["rms"] + 0.01 * residuals["rotation_deg"]["rms"])


def convention_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows = []
    for result in sorted(results, key=residual_score):
        summary_rows.append(
            {
                "convention": result["convention"],
                "residual_score": result["residual_score"],
                "translation_rms_m": result["residuals"]["translation_m"]["rms"],
                "translation_max_m": result["residuals"]["translation_m"]["max"],
                "rotation_rms_deg": result["residuals"]["rotation_deg"]["rms"],
                "rotation_max_deg": result["residuals"]["rotation_deg"]["max"],
            }
        )
    return summary_rows


def residual_errors(
    samples: list[dict[str, np.ndarray]],
    t_base_cam: np.ndarray,
    t_ee_marker: np.ndarray,
) -> dict[str, Any]:
    translation_errors = []
    rotation_errors_deg = []
    for sample in samples:
        left = t_base_cam @ sample["T_cam_marker"]
        right = sample["T_base_ee"] @ t_ee_marker
        delta = invert_transform(left) @ right
        translation_errors.append(float(np.linalg.norm(delta[:3, 3])))
        rotation_errors_deg.append(float(rotation_angle_deg(delta[:3, :3])))

    return {
        "translation_m": summary(translation_errors),
        "rotation_deg": summary(rotation_errors_deg),
    }


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "rms": float(np.sqrt(np.mean(array * array))),
    }


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (float(np.trace(rotation)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def save_base_to_camera(
    path: Path,
    result: dict[str, Any],
    samples: list[dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> None:
    matrix = result["T_base_cam"]
    translation = matrix[:3, 3]
    quaternion = rotation_to_quaternion_xyzw(matrix[:3, :3])
    timestamp = time.time()
    zed_settings = common_zed_settings(samples)
    quality = result_quality(result)
    document = {
        "frames": {"parent": args.base_frame, "child": args.camera_frame},
        "camera": {
            "model": "ZED Mini",
            "serial_number": 14778242,
            "intrinsics_file": "calibration/zed_intrinsics.yaml",
            "calibration_source": "/usr/local/zed/settings/SN14778242.conf",
        },
        "calibrated": True,
        "calibration": {
            "date": dt.datetime.fromtimestamp(timestamp).date().isoformat(),
            "timestamp": timestamp,
            "sample_count": len(samples),
            "method": f"cv2.calibrateRobotWorldHandEye:{result['method']}",
        },
        "zed": zed_settings,
        "results": quality,
        "method": f"cv2.calibrateRobotWorldHandEye:{result['method']}",
        "selected_convention": result["convention"],
        "sample_count": len(samples),
        "timestamp": timestamp,
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
        "transform_matrix": matrix.tolist(),
        "residuals": result["residuals"],
        "tested_conventions": result["tested_conventions"],
        "notes": [
            "Estimated from multiple paired T_base_ee and T_cam_marker samples.",
            "All combinations of inverting T_base_ee and T_cam_marker were tested; this file stores only the lowest-residual result.",
            "T_base_ee samples are frozen snapshots; do not use a live-updating base_ee.yaml during fitting.",
            "Units are meters for translations.",
        ],
    }
    write_yaml(path, document)


def save_ee_marker(
    path: Path,
    result: dict[str, Any],
    samples: list[dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> None:
    matrix = result["T_ee_marker"]
    translation = matrix[:3, 3]
    quaternion = rotation_to_quaternion_xyzw(matrix[:3, :3])
    timestamp = time.time()
    quality = result_quality(result)
    document = {
        "frames": {"parent": args.ee_frame, "child": args.marker_frame},
        "calibrated": True,
        "calibration": {
            "date": dt.datetime.fromtimestamp(timestamp).date().isoformat(),
            "timestamp": timestamp,
            "sample_count": len(samples),
            "method": f"cv2.calibrateRobotWorldHandEye:{result['method']}",
        },
        "zed": common_zed_settings(samples),
        "results": quality,
        "method": f"cv2.calibrateRobotWorldHandEye:{result['method']}",
        "selected_convention": result["convention"],
        "sample_count": len(samples),
        "timestamp": timestamp,
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
        "T_ee_marker": matrix.tolist(),
        "residuals": result["residuals"],
        "tested_conventions": result["tested_conventions"],
        "notes": [
            "Estimated from multiple paired T_base_ee and T_cam_marker samples.",
            "All combinations of inverting T_base_ee and T_cam_marker were tested; this file stores only the lowest-residual result.",
            "Use this file as --ee-marker-transform for ArUco pose workflows when appropriate.",
        ],
    }
    write_yaml(path, document)


def result_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "T_base_cam": result["T_base_cam"].tolist(),
        "T_ee_marker": result["T_ee_marker"].tolist(),
        "selected_convention": result["convention"],
        "residuals": result["residuals"],
        "tested_conventions": result["tested_conventions"],
    }


def result_quality(result: dict[str, Any]) -> dict[str, float]:
    residuals = result["residuals"]
    return {
        "translation_rms_m": float(residuals["translation_m"]["rms"]),
        "translation_max_m": float(residuals["translation_m"]["max"]),
        "rotation_rms_deg": float(residuals["rotation_deg"]["rms"]),
        "rotation_max_deg": float(residuals["rotation_deg"]["max"]),
    }


def common_zed_settings(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    common = None
    for sample in samples:
        settings = sample.get("zed_settings")
        if settings is None:
            continue
        if common is None:
            common = settings
            continue
        if settings != common:
            raise ValueError("samples contain mismatched zed_settings")
    return common


def write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(document, file, sort_keys=False)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    inverse = np.eye(4, dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


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
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.tolist()


if __name__ == "__main__":
    raise SystemExit(main())
