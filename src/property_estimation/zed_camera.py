"""ZED stereo-camera capture with lightweight depth statistics."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OPEN_TIMEOUT = 30.0


@dataclass(frozen=True)
class VisualObservation:
    left_image: bytes
    right_image: bytes
    depth_summary: str
    mime_type: str = "image/jpeg"


class ZedCamera:
    def __init__(
        self,
        resolution: str = "HD720",
        fps: int = 30,
        open_timeout: float = DEFAULT_OPEN_TIMEOUT,
    ) -> None:
        if open_timeout <= 0:
            raise ValueError("Camera open timeout must be greater than zero")
        self.resolution = resolution
        self.fps = fps
        self.open_timeout = open_timeout
        self._zed = None
        self._sl = None

    def __enter__(self) -> "ZedCamera":
        try:
            import pyzed.sl as sl
        except ImportError as exc:
            raise RuntimeError("pyzed is required; install the ZED SDK Python API") from exc

        resolution = getattr(sl.RESOLUTION, self.resolution, None)
        if resolution is None:
            raise ValueError(f"Unsupported ZED resolution: {self.resolution}")

        zed = sl.Camera()
        params = sl.InitParameters()
        params.camera_resolution = resolution
        params.camera_fps = self.fps
        params.depth_mode = sl.DEPTH_MODE.NEURAL
        params.coordinate_units = sl.UNIT.METER
        params.open_timeout_sec = self.open_timeout

        result = zed.open(params)
        if result != sl.ERROR_CODE.SUCCESS:
            zed.close()
            raise RuntimeError(
                f"Could not open ZED camera within {self.open_timeout:g} seconds: {result}"
            )

        self._sl = sl
        self._zed = zed
        return self

    def __exit__(self, *_args: object) -> None:
        if self._zed is not None:
            self._zed.close()
        self._zed = None
        self._sl = None

    def capture(self) -> VisualObservation:
        if self._zed is None or self._sl is None:
            raise RuntimeError("ZedCamera must be opened with a context manager")

        sl = self._sl
        if self._zed.grab() != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED camera failed to grab a frame")

        left, right, depth = sl.Mat(), sl.Mat(), sl.Mat()
        self._zed.retrieve_image(left, sl.VIEW.LEFT)
        self._zed.retrieve_image(right, sl.VIEW.RIGHT)
        self._zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        return VisualObservation(
            left_image=_mat_to_jpeg(left, sl.ERROR_CODE.SUCCESS),
            right_image=_mat_to_jpeg(right, sl.ERROR_CODE.SUCCESS),
            depth_summary=_summarize_depth(depth.get_data()),
        )


def _mat_to_jpeg(mat: object, success_code: object) -> bytes:
    with tempfile.TemporaryDirectory(prefix="zed-property-") as directory:
        path = Path(directory) / "frame.jpg"
        if mat.write(str(path)) != success_code:
            raise RuntimeError("ZED SDK failed to encode a camera frame")
        return path.read_bytes()


def _summarize_depth(depth: object) -> str:
    try:
        import numpy as np
    except ImportError:
        return "Depth statistics unavailable because NumPy is not installed."

    values = np.asarray(depth, dtype=float)
    height, width = values.shape[:2]
    y0, y1 = height // 4, 3 * height // 4
    x0, x1 = width // 4, 3 * width // 4
    center = values[y0:y1, x0:x1]
    valid = center[np.isfinite(center) & (center > 0)]
    if valid.size == 0:
        return "No valid center-region depth samples."

    p10, median, p90 = np.percentile(valid, [10, 50, 90])
    valid_ratio = valid.size / center.size
    return (
        f"Center-region depth in meters: p10={p10:.3f}, median={median:.3f}, "
        f"p90={p90:.3f}; valid sample ratio={valid_ratio:.2f}. "
        "Use depth only as weak evidence because the object may not fill the center."
    )
