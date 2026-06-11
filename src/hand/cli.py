from __future__ import annotations

import argparse
import socket
import sys
import time
from collections.abc import Sequence

from hand.client import (
    ANGLE_ACT,
    ANGLE_SET,
    DOF_NAMES,
    FORCE_ACT,
    POS_ACT,
    POS_SET,
    SPEED_SET,
    InspireHand,
    ModbusError,
    decode_int16,
)


DEFAULT_HOST = "192.168.11.210"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control an INSPIRE RH56E2-2R-T2 hand over Modbus TCP."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--unit-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=2.0)

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="read actual angles and actuator positions")

    tactile = commands.add_parser(
        "tactile",
        help="visualize live signed force feedback without moving the hand",
    )
    tactile.add_argument(
        "--once",
        action="store_true",
        help="print one sample instead of continuously refreshing",
    )
    tactile.add_argument(
        "--interval",
        type=lambda value: _positive_float(value, "interval"),
        default=0.1,
        help="refresh interval in seconds (default: 0.1)",
    )
    tactile.add_argument(
        "--scale",
        type=lambda value: _positive_int(value, "scale"),
        default=1000,
        help="signed force magnitude represented by a full bar (default: 1000)",
    )

    angle = commands.add_parser(
        "angle",
        help="set recommended normalized finger angles (0..1000; -1 means no move)",
    )
    _add_targets(angle, -1, 1000)
    _add_speed(angle)

    position = commands.add_parser(
        "position",
        help="set raw actuator positions (0..2000; -1 means no move)",
    )
    _add_targets(position, -1, 2000)
    _add_speed(position)
    position.add_argument(
        "--allow-raw-position",
        action="store_true",
        help="acknowledge that the manufacturer discourages raw POS_SET control",
    )
    return parser


def _add_targets(parser: argparse.ArgumentParser, minimum: int, maximum: int) -> None:
    parser.add_argument(
        "targets",
        metavar="TARGET",
        type=lambda value: _bounded_int(value, minimum, maximum),
        nargs=6,
        help="six values: pinky ring middle index thumb_bend thumb_rotation",
    )


def _add_speed(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--speed",
        type=lambda value: _bounded_int(value, 0, 1000),
        help="set the same 0..1000 speed limit for all six DOFs before moving",
    )


def _bounded_int(value: str, minimum: int, maximum: int) -> int:
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise argparse.ArgumentTypeError(
            f"{parsed} is outside the allowed range {minimum}..{maximum}"
        )
    return parsed


def _positive_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be greater than zero")
    return parsed


def _positive_float(value: str, name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{name} must be greater than zero")
    return parsed


def _print_values(label: str, values: Sequence[int]) -> None:
    print(label)
    for name, value in zip(DOF_NAMES, values, strict=True):
        print(f"  {name:15s} {value}")


def _force_bar(value: int, scale: int, width: int = 20) -> str:
    units = min(width, round(abs(value) * width / scale))
    left = "#" * units if value < 0 else ""
    right = "#" * units if value > 0 else ""
    return f"{left:>{width}}|{right:<{width}}"


def _print_tactile(values: Sequence[int], scale: int) -> None:
    print(f"signed force feedback (full bar = +/-{scale}; Ctrl-C to stop)")
    print(f"  {'channel':15s} {'negative':>{20}}|{'positive':<{20}} value")
    for name, value in zip(DOF_NAMES, values, strict=True):
        print(f"  {name:15s} {_force_bar(value, scale)} {value:6d}")


def _visualize_tactile(hand: InspireHand, once: bool, interval: float, scale: int) -> None:
    while True:
        values = [decode_int16(value) for value in hand.read_holding_registers(FORCE_ACT, 6)]
        if not once:
            print("\033[2J\033[H", end="")
        _print_tactile(values, scale)
        if once:
            return
        time.sleep(interval)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "position" and not args.allow_raw_position:
        parser.error("position requires --allow-raw-position")

    try:
        with InspireHand(
            args.host,
            port=args.port,
            unit_id=args.unit_id,
            timeout=args.timeout,
        ) as hand:
            if args.command == "status":
                _print_values(
                    "actual angles (0..1000):",
                    hand.read_holding_registers(ANGLE_ACT, 6),
                )
                _print_values(
                    "actual actuator positions (0..2000):",
                    hand.read_holding_registers(POS_ACT, 6),
                )
                return 0

            if args.command == "tactile":
                _visualize_tactile(hand, args.once, args.interval, args.scale)
                return 0

            if args.speed is not None:
                hand.write_registers(SPEED_SET, [args.speed] * 6)

            address = ANGLE_SET if args.command == "angle" else POS_SET
            hand.write_registers(address, args.targets)
            _print_values(f"{args.command} target sent:", args.targets)
            return 0
    except KeyboardInterrupt:
        return 0
    except (ConnectionError, ModbusError, OSError, socket.timeout) as error:
        print(f"inspire-hand: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
