"""Live ArUco marker XYZ frame viewer for ZED or webcam video."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from camera.aruco_pose import (
    DEFAULT_CONFIG,
    estimate_camera_to_marker,
    import_cv2,
    marker_object_points,
    required_value,
)


DEFAULT_WINDOW_NAME = "ArUco Live Viewer"


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None:
        ...

    def close(self) -> None:
        ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("zed", "webcam"), default="zed")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam index for --source webcam")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--marker-length-m", type=float, help="marker black-square side length")
    parser.add_argument("--axis-length-m", type=float, default=0.03)
    parser.add_argument(
        "--ignore-distortion",
        action="store_true",
        help="use zero distortion coefficients, useful for rectified ZED SDK images",
    )
    parser.add_argument("--resolution", default="HD720")
    parser.add_argument("--eye", choices=("left", "right"), default="left")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--open-timeout", type=float, default=30.0)
    parser.add_argument("--print-interval-s", type=float, default=2.0)
    parser.add_argument("--window-name", default=DEFAULT_WINDOW_NAME)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source: FrameSource | None = None
    try:
        cv2 = import_cv2()
        config = load_config(args.config)
        dictionary_name = args.dictionary or required_value(config.get("dictionary"), "dictionary")
        marker_length = float(
            args.marker_length_m
            if args.marker_length_m is not None
            else required_value(config.get("marker_length_m"), "marker_length_m")
        )
        if marker_length <= 0:
            raise ValueError("marker-length-m must be greater than zero")
        if args.axis_length_m <= 0:
            raise ValueError("axis-length-m must be greater than zero")

        camera = live_camera_parameters(
            config,
            resolution=args.resolution,
            eye=args.eye,
            ignore_distortion=args.ignore_distortion,
        )
        detector = create_detector(cv2, dictionary_name)
        source = open_frame_source(args)
        run_viewer(
            cv2=cv2,
            source=source,
            detector=detector,
            dictionary_name=dictionary_name,
            marker_id=args.marker_id,
            marker_length_m=marker_length,
            axis_length_m=args.axis_length_m,
            camera=camera,
            print_interval_s=args.print_interval_s,
            window_name=args.window_name,
        )
        return 0
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, yaml.YAMLError, OSError) as exc:
        print(f"aruco-live-viewer: {exc}", file=sys.stderr)
        return 1
    finally:
        if source is not None:
            source.close()


def run_viewer(
    *,
    cv2: Any,
    source: FrameSource,
    detector: Any,
    dictionary_name: str,
    marker_id: int | None,
    marker_length_m: float,
    axis_length_m: float,
    camera: dict[str, np.ndarray],
    print_interval_s: float,
    window_name: str,
) -> None:
    last_print = 0.0
    while True:
        frame = source.read()
        if frame is None:
            raise RuntimeError("video stream ended")

        corners, ids = detect_frame_markers(cv2, detector, frame)
        if ids is not None and len(corners) > 0:
            visible_ids = ids.reshape(-1)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for index, detected_id in enumerate(visible_ids):
                detected_id = int(detected_id)
                if marker_id is not None and detected_id != marker_id:
                    continue
                pose = estimate_marker_pose_on_frame(
                    cv2=cv2,
                    frame=frame,
                    corners=corners[index],
                    marker_length_m=marker_length_m,
                    axis_length_m=axis_length_m,
                    camera=camera,
                )
                draw_marker_label(cv2, frame, corners[index], detected_id)
                now = time.monotonic()
                if now - last_print >= print_interval_s:
                    print_transform(
                        marker_id=detected_id,
                        dictionary_name=dictionary_name,
                        marker_length_m=marker_length_m,
                        t_cam_marker=pose["T_cam_marker"],
                    )
                    last_print = now

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyWindow(window_name)


def estimate_marker_pose_on_frame(
    *,
    cv2: Any,
    frame: np.ndarray,
    corners: np.ndarray,
    marker_length_m: float,
    axis_length_m: float,
    camera: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    image_corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    success, rvec, tvec = cv2.solvePnP(
        marker_object_points(marker_length_m),
        image_corners,
        camera["camera_matrix"],
        camera["distortion"],
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        raise RuntimeError("cv2.solvePnP failed")

    cv2.drawFrameAxes(
        frame,
        camera["camera_matrix"],
        camera["distortion"],
        rvec,
        tvec,
        axis_length_m,
    )
    transform = estimate_camera_to_marker(image_corners, marker_length_m, camera)
    return {"T_cam_marker": transform["marker_to_camera"], "rvec": rvec, "tvec": tvec}


def draw_marker_label(cv2: Any, frame: np.ndarray, corners: np.ndarray, marker_id: int) -> None:
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    anchor = tuple(np.round(points[0]).astype(int))
    cv2.putText(
        frame,
        f"ID {marker_id}",
        anchor,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def print_transform(
    *,
    marker_id: int,
    dictionary_name: str,
    marker_length_m: float,
    t_cam_marker: np.ndarray,
) -> None:
    payload = {
        "marker_id": marker_id,
        "dictionary": dictionary_name,
        "marker_length_m": marker_length_m,
        "T_cam_marker": t_cam_marker.tolist(),
    }
    print(json.dumps(payload))


def detect_frame_markers(cv2: Any, detector: Any, frame: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _rejected = detector.detectMarkers(gray)
    else:
        dictionary, parameters = detector
        corners, ids, _rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    return corners, ids


def create_detector(cv2: Any, dictionary_name: str) -> Any:
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    parameters = cv2.aruco.DetectorParameters_create()
    return dictionary, parameters


def live_camera_parameters(
    config: dict[str, Any],
    *,
    resolution: str,
    eye: str,
    ignore_distortion: bool,
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


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def open_frame_source(args: argparse.Namespace) -> FrameSource:
    if args.source == "webcam":
        return WebcamFrameSource(args.camera_index)
    return ZedFrameSource(args.resolution, args.fps, args.open_timeout)


class WebcamFrameSource:
    def __init__(self, camera_index: int) -> None:
        cv2 = import_cv2()
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open webcam index {camera_index}")

    def read(self) -> np.ndarray | None:
        ok, frame = self.capture.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        self.capture.release()


class ZedFrameSource:
    def __init__(self, resolution: str, fps: int, open_timeout: float) -> None:
        try:
            import pyzed.sl as sl
        except ImportError as exc:
            raise ImportError("pyzed is required for --source zed") from exc

        resolution_value = getattr(sl.RESOLUTION, resolution, None)
        if resolution_value is None:
            raise ValueError(f"Unsupported ZED resolution: {resolution}")

        self.sl = sl
        self.zed = sl.Camera()
        params = sl.InitParameters()
        params.camera_resolution = resolution_value
        params.camera_fps = fps
        params.depth_mode = sl.DEPTH_MODE.NONE
        params.open_timeout_sec = open_timeout

        result = self.zed.open(params)
        if result != sl.ERROR_CODE.SUCCESS:
            self.zed.close()
            raise RuntimeError(f"Could not open ZED camera: {result}")
        self.image = sl.Mat()

    def read(self) -> np.ndarray | None:
        result = self.zed.grab()
        if result != self.sl.ERROR_CODE.SUCCESS:
            return None
        result = self.zed.retrieve_image(self.image, self.sl.VIEW.LEFT)
        if result != self.sl.ERROR_CODE.SUCCESS:
            return None
        frame = np.asarray(self.image.get_data())
        if frame.ndim == 3 and frame.shape[2] == 4:
            cv2 = import_cv2()
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def close(self) -> None:
        self.zed.close()


if __name__ == "__main__":
    raise SystemExit(main())
