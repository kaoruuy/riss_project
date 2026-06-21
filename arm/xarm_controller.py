"""Safe UFACTORY xArm control wrapper using the official xarm-python-sdk."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


DEFAULT_ENV_FILE = Path(".env")


class XArmCommandError(RuntimeError):
    """Raised when the xArm SDK returns a non-zero status code."""


class XArmController:
    def __init__(self, ip: str | None = None, env_file: Path = DEFAULT_ENV_FILE) -> None:
        self.ip = ip or read_xarm_ip(env_file)
        self.arm: Any | None = None

    def connect(self) -> None:
        from xarm.wrapper import XArmAPI

        self.arm = XArmAPI(self.ip, is_radian=False)
        self._check("clean_warn", self.arm.clean_warn())
        self._check("clean_error", self.arm.clean_error())
        self._check("motion_enable", self.arm.motion_enable(enable=True))
        self._check("set_mode", self.arm.set_mode(0))
        self._check("set_state", self.arm.set_state(0))

    def get_tcp_pose(self) -> list[float]:
        arm = self._require_arm()
        code, pose = unpack_sdk_result(arm.get_position(is_radian=False), "get_position")
        self._check("get_position", code)
        return list(pose)

    def get_joint_angles(self) -> list[float]:
        arm = self._require_arm()
        code, angles = unpack_sdk_result(arm.get_servo_angle(is_radian=False), "get_servo_angle")
        self._check("get_servo_angle", code)
        return list(angles)

    def move_tcp_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        speed: float = 50,
        wait: bool = True,
        confirm: Callable[[str], bool] | None = None,
    ) -> None:
        message = (
            "About to move xArm TCP to "
            f"[{x}, {y}, {z}, {roll}, {pitch}, {yaw}] "
            f"(mm/degrees), speed={speed}, wait={wait}."
        )
        require_motion_confirmation(message, confirm=confirm)
        arm = self._require_arm()
        code = arm.set_position(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=speed,
            wait=wait,
            is_radian=False,
        )
        self._check("set_position", code)

    def move_joint_angles(
        self,
        angles: Sequence[float],
        speed: float = 20,
        wait: bool = True,
        confirm: Callable[[str], bool] | None = None,
    ) -> None:
        angle_list = [float(angle) for angle in angles]
        message = (
            f"About to move xArm joints to {angle_list} degrees, "
            f"speed={speed}, wait={wait}."
        )
        require_motion_confirmation(message, confirm=confirm)
        arm = self._require_arm()
        code = arm.set_servo_angle(
            angle=angle_list,
            speed=speed,
            wait=wait,
            is_radian=False,
        )
        self._check("set_servo_angle", code)

    def disconnect(self) -> None:
        if self.arm is not None:
            self.arm.disconnect()
            self.arm = None

    def _require_arm(self) -> Any:
        if self.arm is None:
            raise RuntimeError("xArm is not connected; call connect() first")
        return self.arm

    @staticmethod
    def _check(operation: str, code: Any) -> None:
        if code != 0:
            raise XArmCommandError(f"{operation} failed with xArm SDK code {code}")


def read_xarm_ip(env_file: Path = DEFAULT_ENV_FILE) -> str:
    values = read_env_file(env_file)
    ip = values.get("XARM_IP") or os.environ.get("XARM_IP")
    if not ip:
        raise ValueError(f"XARM_IP must be set in {env_file} or the environment")
    return ip


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def unpack_sdk_result(result: Any, operation: str) -> tuple[int, Sequence[float]]:
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise XArmCommandError(f"{operation} returned unexpected result: {result!r}")
    code, values = result
    if not isinstance(values, (list, tuple)):
        raise XArmCommandError(f"{operation} returned unexpected values: {values!r}")
    return int(code), values


def require_motion_confirmation(
    warning: str,
    confirm: Callable[[str], bool] | None = None,
) -> None:
    print("WARNING: xArm motion command requested.")
    print(warning)
    print("Check the workspace, payload, tool, hand, cables, and emergency stop before continuing.")

    if confirm is None:
        response = input("Type MOVE to execute: ")
        confirmed = response == "MOVE"
    else:
        confirmed = confirm(warning)

    if not confirmed:
        raise RuntimeError("motion cancelled by user")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--ip", help="override XARM_IP from .env")
    parser.add_argument(
        "--move",
        action="store_true",
        help="execute an explicit movement after reading status; never moves unless this is set",
    )
    movement = parser.add_mutually_exclusive_group()
    movement.add_argument(
        "--tcp-pose",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="target TCP pose in mm/degrees for use with --move",
    )
    movement.add_argument(
        "--joint-angles",
        type=float,
        nargs="+",
        help="target joint angles in degrees for use with --move",
    )
    parser.add_argument("--speed", type=float, help="movement speed; default depends on motion type")
    parser.add_argument("--no-wait", action="store_true", help="do not wait for motion completion")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    controller = XArmController(ip=args.ip, env_file=args.env_file)

    try:
        controller.connect()
        tcp_pose = controller.get_tcp_pose()
        joint_angles = controller.get_joint_angles()
        print(f"Current TCP pose [x, y, z, roll, pitch, yaw] (mm/degrees): {tcp_pose}")
        print(f"Current joint angles (degrees): {joint_angles}")

        if args.move:
            wait = not args.no_wait
            if args.tcp_pose:
                controller.move_tcp_pose(
                    *args.tcp_pose,
                    speed=args.speed if args.speed is not None else 50,
                    wait=wait,
                )
            elif args.joint_angles:
                controller.move_joint_angles(
                    args.joint_angles,
                    speed=args.speed if args.speed is not None else 20,
                    wait=wait,
                )
            else:
                print("--move was set, but no --tcp-pose or --joint-angles target was provided.")
                return 2
        else:
            print("No motion executed. Pass --move with an explicit target to command the robot.")
        return 0
    except (ImportError, RuntimeError, ValueError, XArmCommandError) as exc:
        print(f"xarm-controller: {exc}", file=sys.stderr)
        return 1
    finally:
        controller.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
