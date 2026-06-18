"""Command-line runner for ZED visual physical-property estimation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .fusion import EstimateFusion
from .openai_vlm import OpenAIVisionEstimator
from .zed_camera import ZedCamera


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--resolution", default="HD720")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--camera-open-timeout",
        type=float,
        default=30.0,
        help="allow the ZED SDK this many seconds to open the camera (default: 30)",
    )
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--observations", type=int, default=1, help="0 runs until Ctrl-C")
    parser.add_argument("--output", type=Path, help="Append fused estimates as JSON Lines")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        estimator = OpenAIVisionEstimator(model=args.model)
        fusion = EstimateFusion()
        with ZedCamera(args.resolution, args.fps, args.camera_open_timeout) as camera:
            index = 0
            while args.observations == 0 or index < args.observations:
                estimate = estimator.estimate(camera.capture())
                fused = fusion.add(estimate)
                record = {
                    "timestamp": time.time(),
                    "observation_number": index + 1,
                    "estimate": fused.to_dict(),
                }
                line = json.dumps(record, indent=2)
                print(line, flush=True)
                if args.output:
                    with args.output.open("a", encoding="utf-8") as output:
                        output.write(json.dumps(record) + "\n")
                index += 1
                if args.observations == 0 or index < args.observations:
                    time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
