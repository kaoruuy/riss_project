"""Utilities for table-marker based camera pose recovery."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from camera.aruco_pose import (
    as_transform_matrix,
    detect_markers,
    estimate_camera_to_marker,
    invert_transform,
    load_transform_matrix,
    select_marker,
)
from camera.fit_hand_eye import rotation_angle_deg, rotation_to_quaternion_xyzw
from camera.zed_config import DEFAULT_EYE, DEFAULT_RESOLUTION, camera_parameters_from_config


DEFAULT_TABLE_REFERENCES = Path("calibration/table_marker_references.yaml")
DEFAULT_REFERENCE_IMAGE = Path("calibration/table_marker_references.png")
DEFAULT_RECOVERED_BASE_TO_CAMERA = Path("calibration/base_to_camera_recovered.yaml")
DEFAULT_TABLE_MARKER_IDS = (1, 2, 3, 4)


def parse_marker_ids(value: str) -> list[int]:
    marker_ids = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        marker_id = int(text)
        if marker_id < 0:
            raise ValueError("marker ids must be non-negative")
        marker_ids.append(marker_id)
    if not marker_ids:
        raise ValueError("at least one marker id is required")
    return marker_ids


def detect_marker_poses(
    image_path: Path,
    config: dict[str, Any],
    dictionary_name: str,
    marker_length_m: float,
    marker_ids: list[int],
    ignore_distortion: bool = False,
    require_all: bool = True,
    resolution: str = DEFAULT_RESOLUTION,
    eye: str = DEFAULT_EYE,
) -> dict[int, np.ndarray]:
    camera = camera_parameters_from_config(
        config,
        resolution=resolution,
        eye=eye,
        ignore_distortion=ignore_distortion,
    )
    corners, ids = detect_markers(image_path, dictionary_name)
    poses = {}
    for marker_id in marker_ids:
        try:
            corner_set, detected_id = select_marker(corners, ids, marker_id)
        except RuntimeError:
            if require_all:
                raise
            continue
        transform = estimate_camera_to_marker(corner_set, marker_length_m, camera)
        poses[int(detected_id)] = transform["marker_to_camera"]
    if require_all and set(poses) != set(marker_ids):
        missing = sorted(set(marker_ids) - set(poses))
        raise RuntimeError(f"missing required table marker ids: {missing}")
    return poses


def compute_table_references(
    t_base_cam: np.ndarray,
    marker_poses: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    t_base_cam = require_transform(t_base_cam, "T_base_cam")
    return {
        marker_id: t_base_cam @ require_transform(t_cam_table, f"T_cam_table_{marker_id}")
        for marker_id, t_cam_table in marker_poses.items()
    }


def recover_base_to_camera(
    table_references: dict[int, np.ndarray],
    marker_poses: dict[int, np.ndarray],
    min_markers: int = 1,
) -> dict[str, Any]:
    common_ids = sorted(set(table_references) & set(marker_poses))
    if len(common_ids) < min_markers:
        raise ValueError(
            f"only {len(common_ids)} referenced table markers detected; need at least {min_markers}"
        )

    estimates = {}
    for marker_id in common_ids:
        t_base_table = require_transform(table_references[marker_id], f"T_base_table_{marker_id}")
        t_cam_table = require_transform(marker_poses[marker_id], f"T_cam_table_{marker_id}")
        estimates[marker_id] = t_base_table @ invert_transform(t_cam_table)

    t_base_cam = average_transforms(list(estimates.values()))
    residuals = residuals_to_average(t_base_cam, estimates)
    return {
        "T_base_cam": t_base_cam,
        "used_marker_ids": common_ids,
        "per_marker_estimates": estimates,
        "residuals": residuals,
    }


def load_table_references(path: Path) -> dict[int, np.ndarray]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    markers = data.get("table_markers")
    if not isinstance(markers, list):
        raise ValueError(f"{path} must contain a table_markers list")

    references = {}
    for marker in markers:
        if not isinstance(marker, dict):
            raise ValueError("each table_markers item must be a mapping")
        marker_id = int(marker["marker_id"])
        references[marker_id] = require_transform(
            marker.get("T_base_table"), f"table marker {marker_id} T_base_table"
        )
    return references


def load_base_to_camera(path: Path) -> np.ndarray:
    return load_transform_matrix(path, preferred_key="transform_matrix")


def table_references_document(
    table_references: dict[int, np.ndarray],
    marker_poses: dict[int, np.ndarray],
    *,
    image_path: Path,
    dictionary_name: str,
    marker_length_m: float,
    base_to_camera_path: Path,
    base_frame: str,
    camera_frame: str,
    zed_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "frames": {
            "base": base_frame,
            "camera": camera_frame,
            "table_marker_prefix": "table_marker_",
        },
        "timestamp": time.time(),
        "image": str(image_path),
        "dictionary": dictionary_name,
        "marker_length_m": float(marker_length_m),
        "base_to_camera_file": str(base_to_camera_path),
        "zed_settings": zed_settings,
        "table_markers": [
            {
                "marker_id": marker_id,
                "frame": f"table_marker_{marker_id}",
                "T_cam_table": marker_poses[marker_id].tolist(),
                "T_base_table": table_references[marker_id].tolist(),
            }
            for marker_id in sorted(table_references)
        ],
        "notes": [
            "These are fixed table marker poses in the robot base frame.",
            "Regenerate this file only after a successful hand-eye calibration and while the camera has not moved.",
            "Future recovery can estimate T_base_cam from T_base_table_i @ inverse(T_cam_table_i).",
        ],
    }


def recovered_base_to_camera_document(
    result: dict[str, Any],
    *,
    image_path: Path,
    references_path: Path,
    base_frame: str,
    camera_frame: str,
    zed_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    matrix = result["T_base_cam"]
    translation = matrix[:3, 3]
    quaternion = rotation_to_quaternion_xyzw(matrix[:3, :3])
    return {
        "frames": {"parent": base_frame, "child": camera_frame},
        "calibrated": True,
        "method": "table_marker_recovery",
        "timestamp": time.time(),
        "image": str(image_path),
        "table_references_file": str(references_path),
        "zed_settings": zed_settings,
        "used_marker_ids": result["used_marker_ids"],
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
        "per_marker_estimates": [
            {
                "marker_id": marker_id,
                "T_base_cam": result["per_marker_estimates"][marker_id].tolist(),
            }
            for marker_id in result["used_marker_ids"]
        ],
        "residuals": result["residuals"],
        "notes": [
            "Recovered from fixed table-marker reference poses.",
            "This updates camera extrinsics after camera motion without redoing hand-eye calibration.",
        ],
    }


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("at least one transform is required")
    rotations = [require_transform(transform, "transform")[:3, :3] for transform in transforms]
    translations = [require_transform(transform, "transform")[:3, 3] for transform in transforms]
    average = np.eye(4, dtype=np.float64)
    average[:3, :3] = average_rotation(rotations)
    average[:3, 3] = np.mean(np.asarray(translations, dtype=np.float64), axis=0)
    return average


def average_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    matrix = np.sum(np.asarray(rotations, dtype=np.float64), axis=0)
    u, _singular_values, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation


def residuals_to_average(
    t_base_cam: np.ndarray,
    estimates: dict[int, np.ndarray],
) -> dict[str, Any]:
    per_marker = []
    translation_errors = []
    rotation_errors = []
    for marker_id, estimate in sorted(estimates.items()):
        delta = invert_transform(t_base_cam) @ estimate
        translation_error = float(np.linalg.norm(delta[:3, 3]))
        rotation_error = rotation_angle_deg(delta[:3, :3])
        translation_errors.append(translation_error)
        rotation_errors.append(rotation_error)
        per_marker.append(
            {
                "marker_id": marker_id,
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
            }
        )
    return {
        "translation_m": summarize(translation_errors),
        "rotation_deg": summarize(rotation_errors),
        "per_marker": per_marker,
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "rms": float(np.sqrt(np.mean(array * array))),
    }


def require_transform(value: Any, name: str) -> np.ndarray:
    matrix = as_transform_matrix(value)
    if matrix is None:
        raise ValueError(f"{name} must be a 4x4 transform matrix")
    return matrix


def write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(document, file, sort_keys=False)
