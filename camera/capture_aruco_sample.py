"""Capture a synchronized ZED ArUco image and xArm base-to-EE pose sample."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from arm.base_ee_tracker import base_ee_document, sample_from_xarm_tcp_pose
from arm.xarm_controller import DEFAULT_ENV_FILE, XArmCommandError, XArmController
from camera.aruco_pose import (
    DEFAULT_CONFIG,
    detect_markers,
    estimate_camera_to_marker,
    load_config,
    required_value,
    select_marker,
)
from camera.zed_config import (
    ZedRuntimeConfig,
    add_zed_runtime_args,
    camera_parameters_from_config,
    config_from_args,
    left_view_value,
    open_zed_camera,
)


DEFAULT_OUTPUT_DIR = Path("calibration/samples")
SAMPLE_RE = re.compile(r"^pose_(\d{3})\.png$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dictionary", help="override ArUco dictionary from config")
    parser.add_argument("--marker-length-m", type=float, help="override marker length from config")
    parser.add_argument(
        "--marker-id",
        type=int,
        default=0,
        help="specific marker id to use for hand-eye calibration; defaults to hand marker 0",
    )
    parser.add_argument("--ignore-distortion", action="store_true")
    add_zed_runtime_args(parser)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--ip", help="override XARM_IP from .env")
    parser.add_argument("--base-frame", default="base")
    parser.add_argument("--ee-frame", default="xarm_tcp")
    parser.add_argument("--camera-frame", help="override camera frame name")
    parser.add_argument("--dry-run", action="store_true", help="print next sample paths without capture")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = next_sample_paths(args.output_dir)
        if args.dry_run:
            print(paths["image"])
            print(paths["annotated_image"])
            print(paths["marker"])
            print(paths["base_ee"])
            return 0

        config = load_config(args.config)
        dictionary_name = args.dictionary or required_value(config.get("dictionary"), "dictionary")
        marker_length = args.marker_length_m or required_value(
            config.get("marker_length_m"), "marker_length_m"
        )
        camera_frame = args.camera_frame or config.get("camera", {}).get(
            "frame", "zed_left_camera_optical_frame"
        )
        zed_config = config_from_args(args)

        controller = XArmController(ip=args.ip, env_file=args.env_file)
        controller.connect()
        try:
            zed_settings = capture_zed_left_image(paths["image"], zed_config)
            tcp_pose = controller.get_tcp_pose()
        finally:
            controller.disconnect()

        marker_document = marker_pose_document(
            image_path=paths["image"],
            config=config,
            dictionary_name=dictionary_name,
            marker_length_m=float(marker_length),
            marker_id=args.marker_id,
            ignore_distortion=args.ignore_distortion,
            camera_frame=camera_frame,
            sample_index=paths["index"],
            zed_settings=zed_settings,
            resolution=zed_config.resolution,
            eye=zed_config.eye,
            annotated_image_path=paths["annotated_image"],
        )
        write_yaml(paths["marker"], marker_document)

        base_ee_sample = sample_from_xarm_tcp_pose(tcp_pose)
        base_ee_document_ = base_ee_document(
            base_ee_sample,
            args.base_frame,
            args.ee_frame,
            source="xarm.get_tcp_pose",
        )
        base_ee_document_["sample_index"] = paths["index"]
        base_ee_document_["paired_image"] = str(paths["image"])
        base_ee_document_["zed_settings"] = zed_settings
        write_yaml(paths["base_ee"], base_ee_document_)

        print(f"Saved image: {paths['image']}")
        print(f"Saved annotated image: {paths['annotated_image']}")
        print(f"Saved marker pose: {paths['marker']}")
        print(f"Saved xArm base-EE pose: {paths['base_ee']}")
        return 0
    except (
        FileNotFoundError,
        ImportError,
        RuntimeError,
        ValueError,
        XArmCommandError,
        yaml.YAMLError,
        OSError,
    ) as exc:
        print(f"capture-aruco-sample: {exc}", file=sys.stderr)
        return 1


def next_sample_paths(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index = next_sample_index(output_dir)
    stem = f"pose_{index:03d}"
    paths = {
        "index": index,
        "image": output_dir / f"{stem}.png",
        "annotated_image": output_dir / f"{stem}_annotated.png",
        "marker": output_dir / f"{stem}_marker.yaml",
        "base_ee": output_dir / f"{stem}_base_ee.yaml",
    }
    for key, path in paths.items():
        if key != "index" and path.exists():
            raise FileExistsError(f"refusing to overwrite existing sample file: {path}")
    return paths


def next_sample_index(output_dir: Path) -> int:
    indices = []
    if output_dir.exists():
        for path in output_dir.iterdir():
            match = SAMPLE_RE.match(path.name)
            if match:
                indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def capture_zed_left_image(path: Path, zed_config: ZedRuntimeConfig) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for ZED image capture") from exc

    zed, sl = open_zed_camera(zed_config)

    try:
        image = sl.Mat()
        result = zed.grab()
        if result != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED grab failed: {result}")
        result = zed.retrieve_image(image, left_view_value(sl, zed_config))
        if result != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED image retrieval failed: {result}")

        path.parent.mkdir(parents=True, exist_ok=True)
        frame = np.asarray(image.get_data())
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"failed to write image: {path}")
        return zed_config.metadata_with_sdk(sl)
    finally:
        zed.close()


def marker_pose_document(
    image_path: Path,
    config: dict[str, Any],
    dictionary_name: str,
    marker_length_m: float,
    marker_id: int | None,
    ignore_distortion: bool,
    camera_frame: str,
    sample_index: int,
    zed_settings: dict[str, Any] | None = None,
    resolution: str = "HD720",
    eye: str = "left",
    annotated_image_path: Path | None = None,
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
    if annotated_image_path is not None:
        write_annotated_marker_image(
            image_path=image_path,
            output_path=annotated_image_path,
            corners=corners,
            ids=ids,
            selected_corners=corner_set,
            selected_marker_id=int(detected_id),
            transform=transform,
            camera=camera,
            axis_length_m=min(marker_length_m * 0.75, 0.03),
        )
    return {
        "sample_index": sample_index,
        "timestamp": time.time(),
        "image": str(image_path),
        "annotated_image": str(annotated_image_path) if annotated_image_path else None,
        "marker_id": int(detected_id),
        "marker_length_m": float(marker_length_m),
        "dictionary": dictionary_name,
        "camera_frame": camera_frame,
        "zed_settings": zed_settings,
        "T_cam_marker": transform["marker_to_camera"].tolist(),
        "T_marker_cam": transform["camera_to_marker"].tolist(),
        "marker_translation_in_camera_m": transform["translation"].reshape(3).tolist(),
        "marker_rotation_in_camera": transform["rotation"].tolist(),
    }


def write_annotated_marker_image(
    *,
    image_path: Path,
    output_path: Path,
    corners: list[np.ndarray],
    ids: np.ndarray,
    selected_corners: np.ndarray,
    selected_marker_id: int,
    transform: dict[str, np.ndarray],
    camera: dict[str, np.ndarray],
    axis_length_m: float,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for annotated ArUco images") from exc

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image for annotation: {image_path}")

    cv2.aruco.drawDetectedMarkers(image, corners, ids.reshape(-1, 1))
    rotation = transform["rotation"]
    translation = transform["translation"]
    rvec, _jacobian = cv2.Rodrigues(rotation)
    cv2.drawFrameAxes(
        image,
        camera["camera_matrix"],
        camera["distortion"],
        rvec,
        translation,
        axis_length_m,
    )
    anchor = tuple(np.round(selected_corners.reshape(4, 2)[0]).astype(int))
    cv2.putText(
        image,
        f"selected ID {selected_marker_id}",
        anchor,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"failed to write annotated image: {output_path}")


def write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(document, file, sort_keys=False)


if __name__ == "__main__":
    raise SystemExit(main())
