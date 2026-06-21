"""Generate printable ArUco marker images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


DEFAULT_DICTIONARY = "DICT_4X4_50"
DEFAULT_OUTPUT = Path("aruco_marker.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("marker_id", type=int, help="marker id to generate")
    parser.add_argument("--dictionary", default=DEFAULT_DICTIONARY, help="OpenCV ArUco dictionary name")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output image path")
    parser.add_argument("--size-px", type=int, default=800, help="marker code size in pixels")
    parser.add_argument("--border-px", type=int, default=100, help="white page border around the marker")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        image = generate_marker(
            marker_id=args.marker_id,
            dictionary_name=args.dictionary,
            size_px=args.size_px,
            border_px=args.border_px,
        )
        write_image(args.output, image)
        print(f"Wrote ArUco marker {args.marker_id} to {args.output}")
    except (ImportError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def generate_marker(
    marker_id: int,
    dictionary_name: str = DEFAULT_DICTIONARY,
    size_px: int = 800,
    border_px: int = 100,
) -> Any:
    if marker_id < 0:
        raise ValueError("marker_id must be non-negative")
    if size_px <= 0:
        raise ValueError("--size-px must be greater than zero")
    if border_px < 0:
        raise ValueError("--border-px must be zero or greater")

    cv2 = import_cv2()
    aruco = cv2.aruco
    dictionary_id = getattr(aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")

    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    marker_count = getattr(dictionary, "bytesList", [])
    if marker_count is not None and len(marker_count) and marker_id >= len(marker_count):
        raise ValueError(f"marker_id {marker_id} is outside dictionary range 0..{len(marker_count) - 1}")

    marker = draw_marker(aruco, dictionary, marker_id, size_px)
    if border_px == 0:
        return marker

    import numpy as np

    page = np.full((size_px + 2 * border_px, size_px + 2 * border_px), 255, dtype=marker.dtype)
    page[border_px : border_px + size_px, border_px : border_px + size_px] = marker
    return page


def draw_marker(aruco: Any, dictionary: Any, marker_id: int, size_px: int) -> Any:
    if hasattr(aruco, "generateImageMarker"):
        return aruco.generateImageMarker(dictionary, marker_id, size_px)
    if hasattr(aruco, "drawMarker"):
        return aruco.drawMarker(dictionary, marker_id, size_px)
    raise ImportError("installed OpenCV ArUco module cannot generate markers")


def write_image(path: Path, image: Any) -> None:
    cv2 = import_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


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
