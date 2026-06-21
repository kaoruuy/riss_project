from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path

from arm.xarm_controller import (
    XArmCommandError,
    XArmController,
    main,
    read_xarm_ip,
)


class FakeXArmAPI:
    instances: list[FakeXArmAPI] = []

    def __init__(self, ip: str, is_radian: bool = True) -> None:
        self.ip = ip
        self.is_radian = is_radian
        self.calls: list[tuple[str, object]] = []
        self.position = [1, 2, 3, 4, 5, 6]
        self.angles = [10, 20, 30, 40, 50, 60]
        self.fail_next_code = 0
        FakeXArmAPI.instances.append(self)

    def clean_warn(self) -> int:
        self.calls.append(("clean_warn", None))
        return 0

    def clean_error(self) -> int:
        self.calls.append(("clean_error", None))
        return 0

    def motion_enable(self, enable: bool) -> int:
        self.calls.append(("motion_enable", enable))
        return 0

    def set_mode(self, mode: int) -> int:
        self.calls.append(("set_mode", mode))
        return 0

    def set_state(self, state: int) -> int:
        self.calls.append(("set_state", state))
        return 0

    def get_position(self, is_radian: bool = True) -> list[object]:
        self.calls.append(("get_position", is_radian))
        return [0, self.position]

    def get_servo_angle(self, is_radian: bool = True) -> list[object]:
        self.calls.append(("get_servo_angle", is_radian))
        return [0, self.angles]

    def set_position(self, **kwargs: object) -> int:
        self.calls.append(("set_position", kwargs))
        return self.fail_next_code

    def set_servo_angle(self, **kwargs: object) -> int:
        self.calls.append(("set_servo_angle", kwargs))
        return self.fail_next_code

    def disconnect(self) -> None:
        self.calls.append(("disconnect", None))


class XArmControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeXArmAPI.instances = []
        xarm_module = types.ModuleType("xarm")
        wrapper_module = types.ModuleType("xarm.wrapper")
        wrapper_module.XArmAPI = FakeXArmAPI
        xarm_module.wrapper = wrapper_module
        self.original_modules = {
            "xarm": sys.modules.get("xarm"),
            "xarm.wrapper": sys.modules.get("xarm.wrapper"),
        }
        sys.modules["xarm"] = xarm_module
        sys.modules["xarm.wrapper"] = wrapper_module

    def tearDown(self) -> None:
        for name, module in self.original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_reads_xarm_ip_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("XARM_IP=192.168.1.222\n", encoding="utf-8")

            self.assertEqual(read_xarm_ip(env_file), "192.168.1.222")

    def test_connect_configures_robot_safely(self) -> None:
        controller = XArmController(ip="192.168.1.222")

        controller.connect()

        fake = FakeXArmAPI.instances[-1]
        self.assertEqual(fake.ip, "192.168.1.222")
        self.assertFalse(fake.is_radian)
        self.assertEqual(
            fake.calls[:5],
            [
                ("clean_warn", None),
                ("clean_error", None),
                ("motion_enable", True),
                ("set_mode", 0),
                ("set_state", 0),
            ],
        )

    def test_getters_use_degrees(self) -> None:
        controller = XArmController(ip="192.168.1.222")
        controller.connect()

        self.assertEqual(controller.get_tcp_pose(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(controller.get_joint_angles(), [10, 20, 30, 40, 50, 60])

        fake = FakeXArmAPI.instances[-1]
        self.assertIn(("get_position", False), fake.calls)
        self.assertIn(("get_servo_angle", False), fake.calls)

    def test_move_tcp_pose_requires_confirmation_and_uses_degrees(self) -> None:
        controller = XArmController(ip="192.168.1.222")
        controller.connect()

        with contextlib.redirect_stdout(io.StringIO()):
            controller.move_tcp_pose(1, 2, 3, 4, 5, 6, confirm=lambda _warning: True)

        fake = FakeXArmAPI.instances[-1]
        call_name, kwargs = fake.calls[-1]
        self.assertEqual(call_name, "set_position")
        self.assertEqual(kwargs["is_radian"], False)
        self.assertEqual(kwargs["speed"], 50)

    def test_motion_return_code_raises(self) -> None:
        controller = XArmController(ip="192.168.1.222")
        controller.connect()
        FakeXArmAPI.instances[-1].fail_next_code = 9

        with self.assertRaises(XArmCommandError), contextlib.redirect_stdout(io.StringIO()):
            controller.move_joint_angles([0, 1, 2], confirm=lambda _warning: True)

    def test_main_does_not_move_without_move_flag(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(["--ip", "192.168.1.222"])

        self.assertEqual(result, 0)
        fake = FakeXArmAPI.instances[-1]
        self.assertNotIn("set_position", [name for name, _args in fake.calls])
        self.assertNotIn("set_servo_angle", [name for name, _args in fake.calls])


if __name__ == "__main__":
    unittest.main()
