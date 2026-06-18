#!/usr/bin/env python3
"""Fail-fast ZED camera connection diagnostic."""

from __future__ import annotations

import glob
import os
import platform
import subprocess
import sys

import pyzed.sl as sl

CAMERA_OPEN_TIMEOUT = 30.0


def command_result(command: list[str]) -> tuple[int | None, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as exc:
        return None, str(exc)
    except subprocess.TimeoutExpired as exc:
        return None, str(exc)
    return result.returncode, (result.stdout + result.stderr).strip()


def command_output(command: list[str]) -> str:
    code, output = command_result(command)
    if output:
        return output
    return f"exit code {code}"


def cuda_is_available() -> tuple[bool, str]:
    code, output = command_result(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    )
    return code == 0, output or f"exit code {code}"


def print_environment() -> tuple[list[str], bool]:
    video_devices = glob.glob("/dev/video*")
    has_cuda, nvidia = cuda_is_available()

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"OS: {platform.platform()}")
    print(f"pyzed: {sl.__file__}")
    print(f"ZED SDK: {sl.Camera.get_sdk_version()}")
    print(f"Video devices: {', '.join(video_devices) or 'none'}")
    print(f"NVIDIA: {nvidia}")

    if video_devices:
        print("Processes using video devices:")
        print(command_output(["fuser", "-v", *video_devices]))

    return video_devices, has_cuda


def main() -> int:
    video_devices, has_cuda = print_environment()

    devices = sl.Camera.get_device_list()
    print(f"ZED devices reported by SDK: {len(devices)}")
    for device in devices:
        print(
            "  "
            f"id={device.id}, serial={device.serial_number}, "
            f"model={device.camera_model}, state={device.camera_state}"
        )

    if not has_cuda:
        print(
            "\nCannot open a ZED camera because CUDA is not available. "
            "The ZED SDK requires a working NVIDIA driver and CUDA-capable GPU."
        )
        print("Fix: verify the NVIDIA driver is loaded, then rerun `nvidia-smi`.")
        return 2

    if not video_devices:
        print(
            "\nCannot open a ZED camera because no /dev/video* devices are visible. "
            "Check the USB cable/port and camera permissions."
        )
        return 2

    if not devices:
        print(
            "\nCannot open a ZED camera because the ZED SDK does not enumerate one. "
            "Check that the camera is connected over a supported USB port and is not busy."
        )
        return 2

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.sdk_verbose = 1
    init_params.open_timeout_sec = CAMERA_OPEN_TIMEOUT

    print(
        f"\nOpening camera at HD720/30 with depth disabled "
        f"(allowing the SDK up to {CAMERA_OPEN_TIMEOUT:g} seconds)..."
    )
    error = zed.open(init_params)
    print(f"Open result: {error}")
    if error != sl.ERROR_CODE.SUCCESS:
        zed.close()
        return 3

    try:
        image = sl.Mat()
        error = zed.grab()
        print(f"First grab result: {error}")
        if error != sl.ERROR_CODE.SUCCESS:
            return 4

        zed.retrieve_image(image, sl.VIEW.LEFT)
        print(f"Image received: {image.get_width()}x{image.get_height()}")
        return 0
    finally:
        zed.close()


if __name__ == "__main__":
    sys.exit(main())
