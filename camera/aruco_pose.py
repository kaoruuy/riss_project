"""Estimate a camera-to-ArUco-marker transform with OpenCV PnP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_CONFIG = Path("calibration/aruco_config.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="image containing the ArUco marker")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dictionary", help="OpenCV ArUco dictionary name, e.g. DICT_6X6_250")
    parser.add_argument("--marker-length-m", type=float, help="marker black-square side length")
    parser.add_argument("--marker-id", type=int, help="specific marker id to use if multiple are visible")
    parser.add_argument(
        "--ignore-distortion",
        action="store_true",
        help="use zero distortion coefficients, useful for rectified ZED SDK images",
    )
    parser.add_argument(
        "--save-transforms",
        type=Path,
        help="write calculated calibration transforms to a YAML file",
    )
    parser.add_argument(
        "--ee-marker-transform",
        type=Path,
        help="YAML/JSON file containing T_ee_marker as a 4x4 transform matrix",
    )
    parser.add_argument(
        "--base-ee-transform",
        type=Path,
        help="YAML/JSON file containing T_base_ee as a 4x4 transform matrix",
    )
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
        dictionary_name = args.dictionary or required_value(config.get("dictionary"), "dictionary")
        marker_length = args.marker_length_m or required_value(
            config.get("marker_length_m"), "marker_length_m"
        )
        camera = camera_parameters(config, ignore_distortion=args.ignore_distortion)

        corners, ids = detect_markers(args.image, dictionary_name)
        corner_set, marker_id = select_marker(corners, ids, args.marker_id)
        transform = estimate_camera_to_marker(corner_set, marker_length, camera)

        payload = {
            "marker_id": int(marker_id),
            "marker_length_m": float(marker_length),
            "camera_frame": config.get("camera", {}).get("frame"),
            "marker_translation_in_camera_m": transform["translation"].reshape(3).tolist(),
            "marker_rotation_in_camera": transform["rotation"].tolist(),
            "marker_to_camera_matrix": transform["marker_to_camera"].tolist(),
            "camera_to_marker_matrix": transform["camera_to_marker"].tolist(),
        }
        if args.save_transforms:
            save_transforms(
                args.save_transforms,
                transform,
                args.ee_marker_transform,
                args.base_ee_transform,
                payload,
            )
        print(json.dumps(payload, indent=2 if args.pretty else None))
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_transform_matrix(path: Path, preferred_key: str | None = None) -> np.ndarray:
    with path.open("r", encoding="utf-8") as file:
        if path.suffix.lower() == ".json":
            data = json.load(file)
        else:
            data = yaml.safe_load(file)

    matrix = transform_matrix_from_data(data, preferred_key=preferred_key)
    if matrix is None:
        raise ValueError(f"{path} must contain a 4x4 transform matrix")
    return matrix


def transform_matrix_from_data(data: Any, preferred_key: str | None = None) -> np.ndarray | None:
    if preferred_key and isinstance(data, dict) and preferred_key in data:
        return as_transform_matrix(data[preferred_key])

    if isinstance(data, dict):
        for key in ("transform_matrix", "matrix", "T", "T_ee_marker", "T_base_ee"):
            if key in data:
                return as_transform_matrix(data[key])
    return as_transform_matrix(data)


def as_transform_matrix(value: Any) -> np.ndarray | None:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (4, 4):
        return None
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transform matrix must contain only finite numbers")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("transform matrix last row must be [0, 0, 0, 1]")
    return matrix


def save_transforms(
    path: Path,
    transform: dict[str, np.ndarray],
    ee_marker_path: Path | None,
    base_ee_path: Path | None,
    metadata: dict[str, Any],
) -> None:
    t_cam_marker = transform["marker_to_camera"]
    t_marker_cam = transform["camera_to_marker"]

    transforms = {
        "T_cam_marker": t_cam_marker.tolist(),
        "T_marker_cam": t_marker_cam.tolist(),
    }
    sources: dict[str, str] = {"T_cam_marker": "aruco_pose"}

    if ee_marker_path:
        t_ee_marker = load_transform_matrix(ee_marker_path, preferred_key="T_ee_marker")
        transforms["T_ee_marker"] = t_ee_marker.tolist()
        sources["T_ee_marker"] = str(ee_marker_path)
    else:
        t_ee_marker = None

    if base_ee_path:
        t_base_ee = load_transform_matrix(base_ee_path, preferred_key="T_base_ee")
        transforms["T_base_ee"] = t_base_ee.tolist()
        sources["T_base_ee"] = str(base_ee_path)
    else:
        t_base_ee = None

    if t_ee_marker is not None and t_base_ee is not None:
        transforms["T_base_cam"] = (t_base_ee @ t_ee_marker @ t_marker_cam).tolist()
        sources["T_base_cam"] = "T_base_ee @ T_ee_marker @ T_marker_cam"

    document = {
        "frames": {
            "base": "base",
            "ee": "end_effector",
            "marker": f"aruco_marker_{metadata['marker_id']}",
            "camera": metadata.get("camera_frame"),
        },
        "marker_id": metadata["marker_id"],
        "marker_length_m": metadata["marker_length_m"],
        "sources": sources,
        "transforms": transforms,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(document, file, sort_keys=False)


def required_value(value: Any, name: str) -> Any:
    if value is None:
        raise ValueError(f"{name} must be set in the config or provided on the command line")
    return value


def camera_parameters(config: dict[str, Any], ignore_distortion: bool = False) -> dict[str, np.ndarray]:
    camera = config.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("config must contain a camera mapping")

    fx = float(required_value(camera.get("fx"), "camera.fx"))
    fy = float(required_value(camera.get("fy"), "camera.fy"))
    cx = float(required_value(camera.get("cx"), "camera.cx"))
    cy = float(required_value(camera.get("cy"), "camera.cy"))
    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    if ignore_distortion:
        distortion = np.zeros((4, 1), dtype=np.float64)
    else:
        values = required_value(camera.get("distortion_vector"), "camera.distortion_vector")
        distortion = np.asarray(values, dtype=np.float64).reshape(-1, 1)

    return {"camera_matrix": camera_matrix, "distortion": distortion}


def detect_markers(image_path: Path, dictionary_name: str) -> tuple[list[np.ndarray], np.ndarray]:
    cv2 = import_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    aruco = cv2.aruco
    dictionary_id = getattr(aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")

    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(aruco, "ArucoDetector"):
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _rejected = detector.detectMarkers(image)
    else:
        parameters = aruco.DetectorParameters_create()
        corners, ids, _rejected = aruco.detectMarkers(image, dictionary, parameters=parameters)

    if ids is None or len(corners) == 0:
        raise RuntimeError(f"no ArUco markers detected in {image_path}")
    return corners, ids.reshape(-1)


def select_marker(
    corners: list[np.ndarray], ids: np.ndarray, requested_id: int | None = None
) -> tuple[np.ndarray, int]:
    if requested_id is None:
        return normalize_corners(corners[0]), int(ids[0])

    matches = np.where(ids == requested_id)[0]
    if matches.size == 0:
        visible = ", ".join(str(int(marker_id)) for marker_id in ids)
        raise RuntimeError(f"marker id {requested_id} was not detected; visible ids: {visible}")
    index = int(matches[0])
    return normalize_corners(corners[index]), int(ids[index])


def estimate_camera_to_marker(
    image_corners: np.ndarray, marker_length_m: float, camera: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    cv2 = import_cv2()
    if marker_length_m <= 0:
        raise ValueError("marker_length_m must be greater than zero")

    object_points = marker_object_points(marker_length_m)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        normalize_corners(image_corners),
        camera["camera_matrix"],
        camera["distortion"],
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        raise RuntimeError("cv2.solvePnP failed")

    rotation, _jacobian = cv2.Rodrigues(rvec)
    marker_to_camera = make_transform(rotation, tvec)
    camera_to_marker = invert_transform(marker_to_camera)
    return {
        "rotation": rotation,
        "translation": tvec,
        "marker_to_camera": marker_to_camera,
        "camera_to_marker": camera_to_marker,
    }


def marker_object_points(marker_length_m: float) -> np.ndarray:
    half = marker_length_m / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def normalize_corners(corners: np.ndarray) -> np.ndarray:
    return np.asarray(corners, dtype=np.float64).reshape(4, 2)


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


def import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV with ArUco support is required; install opencv-contrib-python") from exc
    if not hasattr(cv2, "aruco"):
        raise ImportError("installed OpenCV does not include cv2.aruco; install opencv-contrib-python")
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
