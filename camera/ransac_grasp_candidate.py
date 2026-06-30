"""Live tabletop object candidate detection from a ZED point cloud."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from arm.xarm_controller import DEFAULT_ENV_FILE, XArmCommandError, XArmController
from camera.aruco_pose import invert_transform, load_transform_matrix
from camera.zed_config import ZedRuntimeConfig, add_zed_runtime_args, config_from_args, left_view_value, open_zed_camera
from camera.zed_pointcloud import Intrinsics, rgb_from_image, zed_left_intrinsics


DEFAULT_BASE_TO_CAMERA = Path("calibration/base_to_camera.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_zed_runtime_args(parser)
    parser.add_argument("--base-to-camera", type=Path, default=DEFAULT_BASE_TO_CAMERA)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument("--ransac-iterations", type=int, default=160)
    parser.add_argument("--plane-threshold-m", type=float, default=0.008)
    parser.add_argument("--object-min-height-m", type=float, default=0.01)
    parser.add_argument("--object-max-height-m", type=float, default=0.30)
    parser.add_argument("--cluster-cell-m", type=float, default=0.025)
    parser.add_argument("--min-cluster-points", type=int, default=80)
    parser.add_argument("--max-cluster-width-m", type=float, default=0.5)
    parser.add_argument("--max-cluster-depth-m", type=float, default=0.5)
    parser.add_argument("--max-cluster-area-m2", type=float, default=0.15)
    parser.add_argument("--min-cluster-density", type=float, default=0.05)
    parser.add_argument("--z-mode", choices=("top", "centroid"), default="top")
    parser.add_argument("--print-interval-s", type=float, default=1.0)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--debug-mask", action="store_true", help="show a simple XY cluster debug image")
    parser.add_argument("--move", action="store_true", help="move xArm to a safe approach pose above the object")
    parser.add_argument("--approach-height-m", type=float, default=0.20, help="height above object to approach to in meters")
    parser.add_argument("--speed", type=float, default=30.0, help="xArm Cartesian speed in mm/s")
    parser.add_argument("--max-speed", type=float, default=50.0)
    parser.add_argument("--workspace-x", type=float, nargs=2, default=[0.10, 0.50], metavar=("MIN", "MAX"))
    parser.add_argument("--workspace-y", type=float, nargs=2, default=[-0.25, 0.25], metavar=("MIN", "MAX"))
    parser.add_argument("--workspace-z", type=float, nargs=2, default=[0.02, 0.40], metavar=("MIN", "MAX"))
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--ip", help="override XARM_IP from .env")
    parser.add_argument("--no-wait", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
        t_base_cam = load_transform_matrix(args.base_to_camera, preferred_key="transform_matrix")
        if args.speed > args.max_speed:
            raise ValueError(f"speed {args.speed} exceeds max-speed {args.max_speed}")
        candidate = run_live_detection(args, config, t_base_cam)
        if args.move:
            if candidate is None:
                raise RuntimeError("no grasp candidate available for motion")
            move_to_candidate(args, candidate)
        return 0
    except (
        FileNotFoundError,
        ImportError,
        RuntimeError,
        ValueError,
        XArmCommandError,
        OSError,
    ) as exc:
        print(f"ransac-grasp-candidate: {exc}", file=sys.stderr)
        return 1


def run_live_detection(
    args: argparse.Namespace,
    zed_config: ZedRuntimeConfig,
    t_base_cam: np.ndarray,
) -> dict[str, Any] | None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for visualization") from exc

    zed, sl = open_zed_camera(zed_config)
    image = sl.Mat()
    depth = sl.Mat()
    last_print = 0.0
    latest: dict[str, Any] | None = None

    try:
        intrinsics = zed_left_intrinsics(zed)
        while True:
            result = zed.grab()
            if result != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"ZED grab failed: {result}")
            result = zed.retrieve_image(image, left_view_value(sl, zed_config))
            if result != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"ZED image retrieval failed: {result}")
            result = zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            if result != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"ZED depth retrieval failed: {result}")

            raw_image = np.asarray(image.get_data())
            depth_m = np.asarray(depth.get_data(), dtype=np.float32)
            frame = bgr_from_zed_image(cv2, raw_image)
            latest = process_frame(
                depth_m=depth_m,
                intrinsics=intrinsics,
                t_base_cam=t_base_cam,
                args=args,
            )

            if latest is not None:
                draw_candidate(cv2, frame, latest, intrinsics)
                now = time.monotonic()
                if now - last_print >= args.print_interval_s:
                    print_candidate(latest)
                    last_print = now

            if not args.no_window:
                cv2.imshow("RANSAC grasp candidate", frame)
                if args.debug_mask and latest is not None:
                    cv2.imshow("Object cluster debug", cluster_debug_image(latest))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            elif latest is not None:
                break
        return latest
    finally:
        zed.close()
        if not args.no_window:
            cv2.destroyAllWindows()


def process_frame(
    *,
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
    t_base_cam: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    points_cam, pixels = point_cloud_from_depth(
        depth_m,
        intrinsics,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        stride=args.stride,
        max_points=args.max_points,
    )
    points_base = transform_points(t_base_cam, points_cam)
    plane = fit_plane_ransac(
        points_base,
        iterations=args.ransac_iterations,
        threshold_m=args.plane_threshold_m,
    )
    if plane is None:
        return None

    heights = signed_plane_distances(points_base, plane["normal"], plane["d"])
    object_mask = (heights > args.object_min_height_m) & (heights < args.object_max_height_m)
    object_points_base = points_base[object_mask]
    object_points_cam = points_cam[object_mask]
    object_pixels = pixels[object_mask]
    workspace_mask = points_within_workspace(object_points_base, args)
    object_points_base = object_points_base[workspace_mask]
    object_points_cam = object_points_cam[workspace_mask]
    object_pixels = object_pixels[workspace_mask]
    if len(object_points_base) < args.min_cluster_points:
        return None

    cluster_indices = largest_grid_cluster(
        object_points_base[:, :2],
        cell_size=args.cluster_cell_m,
        min_points=args.min_cluster_points,
        max_width_m=args.max_cluster_width_m,
        max_depth_m=args.max_cluster_depth_m,
        max_area_m2=args.max_cluster_area_m2,
        min_density=args.min_cluster_density,
    )
    if cluster_indices.size == 0:
        return None

    cluster_base = object_points_base[cluster_indices]
    cluster_cam = object_points_cam[cluster_indices]
    cluster_pixels = object_pixels[cluster_indices]
    center_base = object_center(cluster_base, z_mode=args.z_mode)
    if not within_workspace(center_base, args):
        return None
    center_cam = transform_points(invert_transform(t_base_cam), center_base.reshape(1, 3))[0]
    return {
        "plane": plane,
        "center_base": center_base,
        "center_cam": center_cam,
        "cluster_size": int(len(cluster_base)),
        "cluster_base": cluster_base,
        "cluster_pixels": cluster_pixels,
        "pixel_center": project_point(center_cam, intrinsics),
    }


def point_cloud_from_depth(
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
    *,
    min_depth_m: float,
    max_depth_m: float,
    stride: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if stride <= 0:
        raise ValueError("stride must be greater than zero")
    sampled_depth = np.asarray(depth_m[::stride, ::stride], dtype=np.float32)
    height, width = sampled_depth.shape
    y, x = np.indices((height, width), dtype=np.float32)
    u = x * stride
    v = y * stride
    valid = np.isfinite(sampled_depth) & (sampled_depth >= min_depth_m) & (sampled_depth <= max_depth_m)
    z = sampled_depth[valid]
    if z.size == 0:
        raise ValueError("no valid depth samples in requested range")
    x_m = (u[valid] - intrinsics.cx) * z / intrinsics.fx
    y_m = (v[valid] - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack((x_m, y_m, z)).astype(np.float64)
    pixels = np.column_stack((u[valid], v[valid])).astype(np.float64)
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
        pixels = pixels[indices]
    return points, pixels


def fit_plane_ransac(
    points: np.ndarray,
    *,
    iterations: int,
    threshold_m: float,
    rng: np.random.Generator | None = None,
) -> dict[str, Any] | None:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return None
    rng = rng or np.random.default_rng()
    best_inliers: np.ndarray | None = None
    best_normal: np.ndarray | None = None
    best_d = 0.0
    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        if normal[2] < 0:
            normal = -normal
        d = -float(normal @ sample[0])
        distances = np.abs(signed_plane_distances(points, normal, d))
        inliers = np.flatnonzero(distances < threshold_m)
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = normal
            best_d = d
    if best_inliers is None or best_normal is None:
        return None
    refined_normal, refined_d = refine_plane(points[best_inliers])
    if refined_normal[2] < 0:
        refined_normal = -refined_normal
        refined_d = -refined_d
    return {"normal": refined_normal, "d": float(refined_d), "inliers": best_inliers}


def refine_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _s, vh = np.linalg.svd(points - centroid)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    d = -float(normal @ centroid)
    return normal, d


def signed_plane_distances(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ np.asarray(normal, dtype=np.float64) + float(d)


def largest_grid_cluster(
    points_xy: np.ndarray,
    *,
    cell_size: float,
    min_points: int,
    max_width_m: float | None = None,
    max_depth_m: float | None = None,
    max_area_m2: float | None = None,
    min_density: float = 0.0,
) -> np.ndarray:
    if len(points_xy) == 0:
        return np.array([], dtype=int)
    if cell_size <= 0:
        raise ValueError("cluster-cell-m must be greater than zero")
    cells = np.floor(points_xy / cell_size).astype(np.int64)
    cell_to_indices: dict[tuple[int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        cell_to_indices.setdefault((int(cell[0]), int(cell[1])), []).append(index)

    visited: set[tuple[int, int]] = set()
    best: list[int] = []
    for start in cell_to_indices:
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        component: list[int] = []
        while queue:
            cell = queue.popleft()
            component.extend(cell_to_indices[cell])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = (cell[0] + dx, cell[1] + dy)
                    if neighbor in cell_to_indices and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        if not cluster_is_valid(
            points_xy[np.asarray(component, dtype=int)],
            cell_size=cell_size,
            min_points=min_points,
            max_width_m=max_width_m,
            max_depth_m=max_depth_m,
            max_area_m2=max_area_m2,
            min_density=min_density,
        ):
            continue
        if len(component) > len(best):
            best = component
    return np.asarray(best, dtype=int)


def cluster_is_valid(
    points_xy: np.ndarray,
    *,
    cell_size: float,
    min_points: int,
    max_width_m: float | None,
    max_depth_m: float | None,
    max_area_m2: float | None,
    min_density: float,
) -> bool:
    if len(points_xy) < min_points:
        return False
    extent = cluster_extent(points_xy)
    width, depth = float(extent[0]), float(extent[1])
    if max_width_m is not None and max_width_m > 0 and width > max_width_m:
        return False
    if max_depth_m is not None and max_depth_m > 0 and depth > max_depth_m:
        return False
    area = max(width, cell_size) * max(depth, cell_size)
    if max_area_m2 is not None and max_area_m2 > 0 and area > max_area_m2:
        return False
    if min_density > 0:
        occupied_cells = len({tuple(cell) for cell in np.floor(points_xy / cell_size).astype(np.int64)})
        bbox_cells = max(1, int(np.ceil(max(width, cell_size) / cell_size))) * max(
            1,
            int(np.ceil(max(depth, cell_size) / cell_size)),
        )
        if occupied_cells / bbox_cells < min_density:
            return False
    return True


def cluster_extent(points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if len(points_xy) == 0:
        return np.zeros(2, dtype=np.float64)
    return points_xy.max(axis=0) - points_xy.min(axis=0)


def object_center(points_base: np.ndarray, *, z_mode: str) -> np.ndarray:
    center = points_base.mean(axis=0)
    if z_mode == "top":
        center[2] = float(np.max(points_base[:, 2]))
    elif z_mode != "centroid":
        raise ValueError(f"unsupported z-mode: {z_mode}")
    return center


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def project_point(point_cam: np.ndarray, intrinsics: Intrinsics) -> tuple[int, int] | None:
    x, y, z = [float(value) for value in point_cam]
    if z <= 0:
        return None
    u = int(round(intrinsics.fx * x / z + intrinsics.cx))
    v = int(round(intrinsics.fy * y / z + intrinsics.cy))
    return u, v


def bgr_from_zed_image(cv2: Any, image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    return rgb_from_image(image, "bgra")


def draw_candidate(cv2: Any, frame: np.ndarray, candidate: dict[str, Any], intrinsics: Intrinsics) -> None:
    pixels = candidate["cluster_pixels"].astype(int)
    height, width = frame.shape[:2]
    pixels = pixels[(pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)]
    if len(pixels):
        frame[pixels[:, 1], pixels[:, 0]] = (0, 255, 0)
    center = candidate.get("pixel_center")
    if center is not None:
        cv2.drawMarker(frame, center, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(frame, "object center", (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def cluster_debug_image(candidate: dict[str, Any], size: int = 420) -> np.ndarray:
    import cv2

    points = candidate["cluster_base"][:, :2]
    image = np.zeros((size, size, 3), dtype=np.uint8)
    if len(points) == 0:
        return image
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    uv = ((points - mins) / span * (size - 20) + 10).astype(int)
    image[uv[:, 1], uv[:, 0]] = (0, 255, 0)
    center = candidate["center_base"][:2]
    center_uv = ((center - mins) / span * (size - 20) + 10).astype(int)
    cv2.drawMarker(image, tuple(center_uv), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    return image


def print_candidate(candidate: dict[str, Any]) -> None:
    normal = candidate["plane"]["normal"]
    d = candidate["plane"]["d"]
    center_cam = candidate["center_cam"]
    center_base = candidate["center_base"]
    print(
        "table plane base: "
        f"{normal[0]:+.5f}x {normal[1]:+.5f}y {normal[2]:+.5f}z {d:+.5f}=0 | "
        f"object_cam_m={center_cam.tolist()} | "
        f"object_base_m={center_base.tolist()} | "
        f"cluster_size={candidate['cluster_size']}"
    )


def within_workspace(point: np.ndarray, args: argparse.Namespace) -> bool:
    return (
        args.workspace_x[0] <= point[0] <= args.workspace_x[1]
        and args.workspace_y[0] <= point[1] <= args.workspace_y[1]
        and args.workspace_z[0] <= point[2] <= args.workspace_z[1]
    )


def points_within_workspace(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (
        (points[:, 0] >= args.workspace_x[0])
        & (points[:, 0] <= args.workspace_x[1])
        & (points[:, 1] >= args.workspace_y[0])
        & (points[:, 1] <= args.workspace_y[1])
        & (points[:, 2] >= args.workspace_z[0])
        & (points[:, 2] <= args.workspace_z[1])
    )


def move_to_candidate(args: argparse.Namespace, candidate: dict[str, Any]) -> None:
    target = np.array(candidate["center_base"], dtype=np.float64)
    target[2] += args.approach_height_m
    if not within_workspace(target, args):
        raise ValueError(f"approach target outside workspace limits: {target.tolist()}")
    if args.speed > args.max_speed:
        raise ValueError(f"speed {args.speed} exceeds max-speed {args.max_speed}")

    controller = XArmController(ip=args.ip, env_file=args.env_file)
    try:
        controller.connect()
        current_pose = controller.get_tcp_pose()
        roll, pitch, yaw = current_pose[3:6]
        print(f"Approach target in base frame (m): {target.tolist()}")
        controller.move_tcp_pose(
            x=float(target[0] * 1000.0),
            y=float(target[1] * 1000.0),
            z=float(target[2] * 1000.0),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=args.speed,
            wait=not args.no_wait,
        )
    finally:
        controller.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
