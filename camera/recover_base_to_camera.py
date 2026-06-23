"""Recover T_base_cam from fixed table marker references after camera motion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from camera.aruco_pose import DEFAULT_CONFIG, load_config, required_value
from camera.capture_aruco_sample import capture_zed_left_image
from camera.table_marker_recovery import (
    DEFAULT_RECOVERED_BASE_TO_CAMERA,
    DEFAULT_TABLE_REFERENCES,
    detect_marker_poses,
    load_table_references,
    recover_base_to_camera,
    recovered_base_to_camera_document,
    write_yaml,
)


DEFAULT_RECOVERY_IMAGE = Path("calibration/table_marker_recovery.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="existing image to use instead of capturing a fresh ZED image",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--references", type=Path, default=DEFAULT_TABLE_REFERENCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECOVERED_BASE_TO_CAMERA)
    parser.add_argument("--recovery-image", type=Path, default=DEFAULT_RECOVERY_IMAGE)
    parser.add_argument("--dictionary", help="override ArUco dictionary from config")
    parser.add_argument("--marker-length-m", type=float, help="override marker length from config")
    parser.add_argument("--ignore-distortion", action="store_true")
    parser.add_argument("--min-markers", type=int, default=1)
    parser.add_argument("--resolution", default="HD720")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--open-timeout", type=float, default=30.0)
    parser.add_argument("--base-frame", default="base")
    parser.add_argument("--camera-frame", help="override camera frame name")
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

        image_path = args.image or args.recovery_image
        if args.image is None:
            capture_zed_left_image(image_path, args.resolution, args.fps, args.open_timeout)

        table_references = load_table_references(args.references)
        marker_poses = detect_marker_poses(
            image_path=image_path,
            config=config,
            dictionary_name=dictionary_name,
            marker_length_m=marker_length,
            marker_ids=sorted(table_references),
            ignore_distortion=args.ignore_distortion,
            require_all=False,
        )
        result = recover_base_to_camera(
            table_references,
            marker_poses,
            min_markers=args.min_markers,
        )
        document = recovered_base_to_camera_document(
            result,
            image_path=image_path,
            references_path=args.references,
            base_frame=args.base_frame,
            camera_frame=camera_frame,
        )
        write_yaml(args.output, document)
        print(
            f"Recovered T_base_cam from markers {result['used_marker_ids']} and saved {args.output}"
        )
        return 0
    except (
        FileNotFoundError,
        ImportError,
        RuntimeError,
        ValueError,
        yaml.YAMLError,
        OSError,
    ) as exc:
        print(f"recover-base-to-camera: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
