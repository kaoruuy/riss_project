"""Shared ZED runtime configuration for calibration and manipulation scripts.

Camera frame convention:
+X = image right
+Y = image down
+Z = forward from camera
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from camera.aruco_pose import required_value


DEFAULT_INTRINSICS_FILE = Path("calibration/zed_intrinsics.yaml")
DEFAULT_RESOLUTION = "HD720"
DEFAULT_FPS = 30
DEFAULT_DEPTH_MODE = "NEURAL"
DEFAULT_COORDINATE_UNITS = "METER"
DEFAULT_COORDINATE_SYSTEM = "IMAGE"
DEFAULT_LEFT_VIEW = "LEFT"
DEFAULT_EYE = "left"
DEFAULT_OPEN_TIMEOUT = 30.0


@dataclass(frozen=True)
class ZedRuntimeConfig:
    resolution: str = DEFAULT_RESOLUTION
    fps: int = DEFAULT_FPS
    depth_mode: str = DEFAULT_DEPTH_MODE
    coordinate_units: str = DEFAULT_COORDINATE_UNITS
    coordinate_system: str = DEFAULT_COORDINATE_SYSTEM
    left_view: str = DEFAULT_LEFT_VIEW
    intrinsics_file: Path = DEFAULT_INTRINSICS_FILE
    eye: str = DEFAULT_EYE
    open_timeout: float = DEFAULT_OPEN_TIMEOUT

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["intrinsics_file"] = str(self.intrinsics_file)
        data["left_image_stream_type"] = self.left_view
        data["left_image_rectification"] = left_view_rectification(self.left_view)
        return data

    def metadata_with_sdk(self, sl: Any) -> dict[str, Any]:
        data = self.metadata()
        get_sdk_version = getattr(getattr(sl, "Camera", None), "get_sdk_version", None)
        if callable(get_sdk_version):
            data["sdk_version"] = str(get_sdk_version())
        return data


def add_zed_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    include_depth_mode: bool = True,
) -> None:
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    if include_depth_mode:
        parser.add_argument(
            "--depth-mode",
            default=DEFAULT_DEPTH_MODE,
            choices=["NONE", "PERFORMANCE", "QUALITY", "ULTRA", "NEURAL"],
        )
    parser.add_argument("--coordinate-units", default=DEFAULT_COORDINATE_UNITS)
    parser.add_argument("--coordinate-system", default=DEFAULT_COORDINATE_SYSTEM)
    parser.add_argument("--left-view", default=DEFAULT_LEFT_VIEW)
    parser.add_argument("--intrinsics-file", type=Path, default=DEFAULT_INTRINSICS_FILE)
    parser.add_argument("--eye", choices=("left", "right"), default=DEFAULT_EYE)
    parser.add_argument("--open-timeout", type=float, default=DEFAULT_OPEN_TIMEOUT)


def config_from_args(args: argparse.Namespace, *, depth_mode: str | None = None) -> ZedRuntimeConfig:
    return ZedRuntimeConfig(
        resolution=args.resolution,
        fps=args.fps,
        depth_mode=depth_mode if depth_mode is not None else getattr(args, "depth_mode", DEFAULT_DEPTH_MODE),
        coordinate_units=args.coordinate_units,
        coordinate_system=args.coordinate_system,
        left_view=args.left_view,
        intrinsics_file=args.intrinsics_file,
        eye=args.eye,
        open_timeout=args.open_timeout,
    )


def open_zed_camera(config: ZedRuntimeConfig) -> tuple[Any, Any]:
    try:
        import pyzed.sl as sl
    except ImportError as exc:
        raise ImportError("pyzed is required; install the ZED SDK Python API") from exc

    params = make_init_parameters(sl, config)
    zed = sl.Camera()
    result = zed.open(params)
    if result != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError(f"Could not open ZED camera: {result}")
    return zed, sl


def make_init_parameters(sl: Any, config: ZedRuntimeConfig) -> Any:
    resolution = enum_value(sl.RESOLUTION, config.resolution, "ZED resolution")
    depth_mode = enum_value(sl.DEPTH_MODE, config.depth_mode, "ZED depth mode")
    coordinate_units = enum_value(sl.UNIT, config.coordinate_units, "ZED coordinate units")
    coordinate_system = enum_value(
        sl.COORDINATE_SYSTEM,
        config.coordinate_system,
        "ZED coordinate system",
    )

    params = sl.InitParameters()
    params.camera_resolution = resolution
    params.camera_fps = config.fps
    params.depth_mode = depth_mode
    params.coordinate_units = coordinate_units
    params.coordinate_system = coordinate_system
    params.open_timeout_sec = config.open_timeout
    return params


def left_view_value(sl: Any, config: ZedRuntimeConfig) -> Any:
    return enum_value(sl.VIEW, config.left_view, "ZED left image stream type")


def enum_value(namespace: Any, name: str, label: str) -> Any:
    value = getattr(namespace, name, None)
    if value is None:
        raise ValueError(f"Unsupported {label}: {name}")
    return value


def left_view_rectification(left_view: str) -> str:
    if "UNRECTIFIED" in left_view.upper():
        return "raw_unrectified"
    return "rectified"


def camera_parameters_from_config(
    config: dict[str, Any],
    *,
    resolution: str = DEFAULT_RESOLUTION,
    eye: str = DEFAULT_EYE,
    ignore_distortion: bool = False,
) -> dict[str, np.ndarray]:
    camera = config.get("camera")
    if isinstance(camera, dict) and all(key in camera for key in ("fx", "fy", "cx", "cy")):
        return camera_parameters_from_mapping(camera, ignore_distortion=ignore_distortion)

    resolutions = config.get("resolutions")
    if isinstance(resolutions, dict):
        resolution_data = resolutions.get(resolution)
        if not isinstance(resolution_data, dict):
            raise ValueError(f"config does not contain resolution {resolution}")
        eye_data = resolution_data.get(eye)
        if not isinstance(eye_data, dict):
            raise ValueError(f"config does not contain {resolution}.{eye} intrinsics")
        return camera_parameters_from_mapping(eye_data, ignore_distortion=ignore_distortion)

    raise ValueError("config must be aruco_config.yaml or zed_intrinsics.yaml format")


def camera_parameters_from_mapping(
    camera: dict[str, Any],
    *,
    ignore_distortion: bool,
) -> dict[str, np.ndarray]:
    fx = float(required_value(camera.get("fx"), "camera.fx"))
    fy = float(required_value(camera.get("fy"), "camera.fy"))
    cx = float(required_value(camera.get("cx"), "camera.cx"))
    cy = float(required_value(camera.get("cy"), "camera.cy"))
    camera_matrix = np.asarray(
        camera.get(
            "camera_matrix",
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
        ),
        dtype=np.float64,
    ).reshape(3, 3)
    if ignore_distortion:
        distortion = np.zeros((4, 1), dtype=np.float64)
    else:
        distortion = np.asarray(
            required_value(camera.get("distortion_vector"), "camera.distortion_vector"),
            dtype=np.float64,
        ).reshape(-1, 1)
    return {"camera_matrix": camera_matrix, "distortion": distortion}
