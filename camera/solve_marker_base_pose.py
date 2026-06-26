"""Solve an ArUco marker pose in robot base coordinates from one image."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from camera.aruco_pose import DEFAULT_CONFIG, detect_markers, estimate_camera_to_marker, load_config, required_value, select_marker
from camera.fit_hand_eye import rotation_to_quaternion_xyzw
from camera.table_marker_recovery import load_base_to_camera
from camera.zed_config import DEFAULT_EYE, DEFAULT_RESOLUTION, camera_parameters_from_config


DEFAULT_BASE_TO_CAMERA = Path("calibration/base_to_camera.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-to-camera", type=Path, default=DEFAULT_BASE_TO_CAMERA)
    parser.add_argument("--dictionary", help="override ArUco dictionary from config")
    parser.add_argument("--marker-id", type=int, required=True)
    parser.add_argument("--marker-length-m", type=float, help="override marker length from config")
    parser.add_argument("--ignore-distortion", action="store_true")
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    parser.add_argument("--eye", choices=("left", "right"), default=DEFAULT_EYE)
    parser.add_argument("--base-frame", default="base")
    parser.add_argument("--camera-frame", help="override camera frame name")
    parser.add_argument("--output", type=Path, help="optional YAML output path")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        dictionary_name = args.dictionary or required_value(config.get("dictionary"), "dictionary")
        marker_length = float(
            args.marker_length_m or required_value(config.get("marker_length_m"), "marker_length_m")
        )
        camera_frame = args.camera_frame or config.get("camera", {}).get(
            "frame", "zed_left_camera_optical_frame"
        )
        t_base_cam = load_base_to_camera(args.base_to_camera)
        result = solve_marker_base_pose(
            image_path=args.image,
            config=config,
            dictionary_name=dictionary_name,
            marker_id=args.marker_id,
            marker_length_m=marker_length,
            t_base_cam=t_base_cam,
            ignore_distortion=args.ignore_distortion,
            resolution=args.resolution,
            eye=args.eye,
        )
        document = marker_base_pose_document(
            result,
            image_path=args.image,
            base_to_camera_path=args.base_to_camera,
            dictionary_name=dictionary_name,
            marker_length_m=marker_length,
            base_frame=args.base_frame,
            camera_frame=camera_frame,
        )
        if args.output:
            write_yaml(args.output, document)
        print(json.dumps(document, indent=2 if args.pretty else None))
        return 0
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, yaml.YAMLError) as exc:
        print(f"solve-marker-base-pose: {exc}", file=sys.stderr)
        return 1


def solve_marker_base_pose(
    *,
    image_path: Path,
    config: dict[str, Any],
    dictionary_name: str,
    marker_id: int,
    marker_length_m: float,
    t_base_cam: np.ndarray,
    ignore_distortion: bool = False,
    resolution: str = DEFAULT_RESOLUTION,
    eye: str = DEFAULT_EYE,
) -> dict[str, Any]:
    camera = camera_parameters_from_config(
        config,
        resolution=resolution,
        eye=eye,
        ignore_distortion=ignore_distortion,
    )
    corners, ids = detect_markers(image_path, dictionary_name)
    corner_set, detected_id = select_marker(corners, ids, marker_id)
    transform = estimate_camera_to_marker(corner_set, marker_length_m, camera)
    t_cam_marker = transform["marker_to_camera"]
    t_base_marker = np.asarray(t_base_cam, dtype=np.float64).reshape(4, 4) @ t_cam_marker
    return {
        "marker_id": int(detected_id),
        "T_cam_marker": t_cam_marker,
        "T_base_marker": t_base_marker,
    }


def marker_base_pose_document(
    result: dict[str, Any],
    *,
    image_path: Path,
    base_to_camera_path: Path,
    dictionary_name: str,
    marker_length_m: float,
    base_frame: str,
    camera_frame: str,
) -> dict[str, Any]:
    matrix = result["T_base_marker"]
    translation = matrix[:3, 3]
    quaternion = rotation_to_quaternion_xyzw(matrix[:3, :3])
    marker_frame = f"aruco_marker_{result['marker_id']}"
    return {
        "frames": {
            "parent": base_frame,
            "child": marker_frame,
            "camera": camera_frame,
        },
        "timestamp": time.time(),
        "image": str(image_path),
        "base_to_camera_file": str(base_to_camera_path),
        "dictionary": dictionary_name,
        "marker_id": result["marker_id"],
        "marker_length_m": float(marker_length_m),
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
        "T_base_marker": matrix.tolist(),
        "T_cam_marker": result["T_cam_marker"].tolist(),
        "calculation": "T_base_marker = T_base_cam @ T_cam_marker",
    }


def write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(document, file, sort_keys=False)


if __name__ == "__main__":
    raise SystemExit(main())
