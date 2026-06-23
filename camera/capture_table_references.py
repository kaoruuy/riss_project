"""Capture fixed table-marker poses in the robot base frame after hand-eye calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from camera.aruco_pose import DEFAULT_CONFIG, load_config, required_value
from camera.capture_aruco_sample import capture_zed_left_image
from camera.table_marker_recovery import (
    DEFAULT_REFERENCE_IMAGE,
    DEFAULT_TABLE_MARKER_IDS,
    DEFAULT_TABLE_REFERENCES,
    compute_table_references,
    detect_marker_poses,
    load_base_to_camera,
    parse_marker_ids,
    table_references_document,
    write_yaml,
)


DEFAULT_BASE_TO_CAMERA = Path("calibration/base_to_camera.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="existing image to use instead of capturing a fresh ZED image",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-to-camera", type=Path, default=DEFAULT_BASE_TO_CAMERA)
    parser.add_argument("--output", type=Path, default=DEFAULT_TABLE_REFERENCES)
    parser.add_argument("--reference-image", type=Path, default=DEFAULT_REFERENCE_IMAGE)
    parser.add_argument("--dictionary", help="override ArUco dictionary from config")
    parser.add_argument("--marker-length-m", type=float, help="override marker length from config")
    parser.add_argument(
        "--marker-ids",
        default=",".join(str(marker_id) for marker_id in DEFAULT_TABLE_MARKER_IDS),
        help="comma-separated table marker ids to record; defaults to 1,2,3,4",
    )
    parser.add_argument("--ignore-distortion", action="store_true")
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
        marker_ids = parse_marker_ids(args.marker_ids)
        camera_frame = args.camera_frame or config.get("camera", {}).get(
            "frame", "zed_left_camera_optical_frame"
        )

        image_path = args.image or args.reference_image
        if args.image is None:
            capture_zed_left_image(image_path, args.resolution, args.fps, args.open_timeout)

        t_base_cam = load_base_to_camera(args.base_to_camera)
        marker_poses = detect_marker_poses(
            image_path=image_path,
            config=config,
            dictionary_name=dictionary_name,
            marker_length_m=marker_length,
            marker_ids=marker_ids,
            ignore_distortion=args.ignore_distortion,
        )
        table_references = compute_table_references(t_base_cam, marker_poses)
        document = table_references_document(
            table_references,
            marker_poses,
            image_path=image_path,
            dictionary_name=dictionary_name,
            marker_length_m=marker_length,
            base_to_camera_path=args.base_to_camera,
            base_frame=args.base_frame,
            camera_frame=camera_frame,
        )
        write_yaml(args.output, document)
        print(f"Saved {len(table_references)} table marker references to {args.output}")
        return 0
    except (
        FileNotFoundError,
        ImportError,
        RuntimeError,
        ValueError,
        yaml.YAMLError,
        OSError,
    ) as exc:
        print(f"capture-table-references: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
