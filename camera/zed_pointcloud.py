"""Capture a colored point cloud from ZED RGB and depth frames."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from camera.zed_config import (
    ZedRuntimeConfig,
    add_zed_runtime_args,
    config_from_args,
    left_view_value,
    open_zed_camera,
)


DEFAULT_OUTPUT = Path("zed_pointcloud.ply")


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class ColoredPointCloud:
    points: np.ndarray
    colors: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    add_zed_runtime_args(parser)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=2, help="sample every N pixels before filtering")
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument(
        "--color-order",
        default="bgra",
        choices=["bgra", "rgba", "bgr", "rgb"],
        help="channel order returned by the image source (ZED/OpenCV is usually bgra)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cloud = capture_zed_point_cloud(
            zed_config=config_from_args(args),
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            stride=args.stride,
            max_points=args.max_points,
            color_order=args.color_order,
        )
        write_point_cloud(args.output, cloud)
        print(f"Wrote {len(cloud.points)} points to {args.output}")
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def capture_zed_point_cloud(
    zed_config: ZedRuntimeConfig,
    min_depth_m: float,
    max_depth_m: float,
    stride: int,
    max_points: int,
    color_order: str,
) -> ColoredPointCloud:
    zed, sl = open_zed_camera(zed_config)

    try:
        intrinsics = zed_left_intrinsics(zed)
        image = sl.Mat()
        depth = sl.Mat()

        result = zed.grab()
        if result != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED grab failed: {result}")

        result = zed.retrieve_image(image, left_view_value(sl, zed_config))
        if result != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED image retrieval failed: {result}")

        result = zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        if result != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED depth retrieval failed: {result}")

        rgb = rgb_from_image(np.asarray(image.get_data()), color_order)
        depth_m = np.asarray(depth.get_data(), dtype=np.float32)
        return point_cloud_from_rgbd(
            depth_m,
            rgb,
            intrinsics,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            stride=stride,
            max_points=max_points,
        )
    finally:
        zed.close()


def zed_left_intrinsics(zed: Any) -> Intrinsics:
    info = zed.get_camera_information()
    calibration = getattr(getattr(info, "camera_configuration", info), "calibration_parameters", None)
    if calibration is None:
        calibration = getattr(info, "calibration_parameters", None)
    if calibration is None:
        raise RuntimeError("Could not read ZED calibration parameters")
    left = calibration.left_cam
    return Intrinsics(fx=float(left.fx), fy=float(left.fy), cx=float(left.cx), cy=float(left.cy))


def rgb_from_image(image: np.ndarray, color_order: str) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("RGB image must have shape (height, width, channels>=3)")
    if color_order in {"rgb", "rgba"}:
        channels = [0, 1, 2]
    elif color_order in {"bgr", "bgra"}:
        channels = [2, 1, 0]
    else:
        raise ValueError(f"Unsupported color order: {color_order}")
    return np.asarray(image[:, :, channels], dtype=np.uint8)


def point_cloud_from_rgbd(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Intrinsics,
    min_depth_m: float = 0.1,
    max_depth_m: float = 5.0,
    stride: int = 1,
    max_points: int | None = None,
) -> ColoredPointCloud:
    if stride <= 0:
        raise ValueError("stride must be greater than zero")
    if min_depth_m < 0 or max_depth_m <= min_depth_m:
        raise ValueError("depth range must satisfy 0 <= min_depth_m < max_depth_m")
    if depth_m.ndim != 2:
        raise ValueError("depth image must have shape (height, width)")
    if rgb.shape[:2] != depth_m.shape:
        raise ValueError("RGB and depth image dimensions must match")

    sampled_depth = np.asarray(depth_m[::stride, ::stride], dtype=np.float32)
    sampled_rgb = np.asarray(rgb[::stride, ::stride, :3], dtype=np.uint8)
    height, width = sampled_depth.shape
    y, x = np.indices((height, width), dtype=np.float32)
    u = x * stride
    v = y * stride

    valid = np.isfinite(sampled_depth) & (sampled_depth >= min_depth_m) & (sampled_depth <= max_depth_m)
    z = sampled_depth[valid]
    if z.size == 0:
        raise ValueError("no valid depth samples in the requested range")

    x_m = (u[valid] - intrinsics.cx) * z / intrinsics.fx
    y_m = (v[valid] - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack((x_m, y_m, z)).astype(np.float32)
    colors = sampled_rgb[valid].reshape(-1, 3)

    if max_points is not None and max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
        colors = colors[indices]
    return ColoredPointCloud(points=points, colors=colors)


def write_point_cloud(path: Path, cloud: ColoredPointCloud) -> None:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        np.savez_compressed(path, points=cloud.points, colors=cloud.colors)
    elif suffix == ".ply":
        write_ascii_ply(path, cloud)
    else:
        raise ValueError("output must end with .ply or .npz")


def write_ascii_ply(path: Path, cloud: ColoredPointCloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(cloud.points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(cloud.points, cloud.colors, strict=True):
            r, g, b = (int(value) for value in color)
            file.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n")


if __name__ == "__main__":
    raise SystemExit(main())
