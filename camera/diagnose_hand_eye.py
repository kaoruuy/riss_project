"""Print per-sample hand-eye residuals and outliers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from camera.fit_hand_eye import (
    DEFAULT_SAMPLES,
    fit_hand_eye,
    invert_transform,
    load_samples,
    rotation_angle_deg,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--method", choices=("shah", "li"), default="shah")
    parser.add_argument("--top", type=int, default=10, help="number of largest outliers to print")
    parser.add_argument("--translation-threshold-m", type=float, default=0.05)
    parser.add_argument("--rotation-threshold-deg", type=float, default=5.0)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        raw_document = load_raw_document(args.samples)
        samples = load_samples(args.samples)
        result = fit_hand_eye(samples, method=args.method)
        diagnostics = diagnose_samples(samples, result, raw_document)
        if args.json:
            print(json.dumps(diagnostics_payload(result, diagnostics), indent=2))
        else:
            print_report(
                result,
                diagnostics,
                top=args.top,
                translation_threshold_m=args.translation_threshold_m,
                rotation_threshold_deg=args.rotation_threshold_deg,
            )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(f"diagnose-hand-eye: {exc}", file=sys.stderr)
        return 1


def load_raw_document(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        if path.suffix.lower() == ".json":
            data = json.load(file)
        else:
            data = yaml.safe_load(file)
    if not isinstance(data, dict):
        return {"samples": data}
    return data


def diagnose_samples(
    samples: list[dict[str, np.ndarray]],
    result: dict[str, Any],
    raw_document: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_samples = raw_document.get("samples", [])
    rows = []
    for index, sample in enumerate(samples):
        left = result["T_base_cam"] @ sample["T_cam_marker"]
        right = sample["T_base_ee"] @ result["T_ee_marker"]
        delta = invert_transform(left) @ right
        raw_sample = raw_samples[index] if index < len(raw_samples) and isinstance(raw_samples[index], dict) else {}
        rows.append(
            {
                "index": index + 1,
                "id": raw_sample.get("id", f"sample_{index + 1:03d}"),
                "image": raw_sample.get("image"),
                "base_ee_file": raw_sample.get("base_ee_file"),
                "marker_file": raw_sample.get("marker_file"),
                "translation_error_m": float(np.linalg.norm(delta[:3, 3])),
                "rotation_error_deg": float(rotation_angle_deg(delta[:3, :3])),
            }
        )
    rows.sort(key=lambda row: (row["translation_error_m"], row["rotation_error_deg"]), reverse=True)
    return rows


def diagnostics_payload(result: dict[str, Any], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "selected_convention": result["convention"],
        "residuals": result["residuals"],
        "samples_by_error": diagnostics,
    }


def print_report(
    result: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    top: int,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> None:
    print("Selected convention:")
    for key, value in result["convention"].items():
        print(f"  {key}: {value}")
    print()
    print("Overall residuals:")
    print(
        "  translation RMS/max: "
        f"{result['residuals']['translation_m']['rms']:.4f} / "
        f"{result['residuals']['translation_m']['max']:.4f} m"
    )
    print(
        "  rotation RMS/max: "
        f"{result['residuals']['rotation_deg']['rms']:.2f} / "
        f"{result['residuals']['rotation_deg']['max']:.2f} deg"
    )
    print()

    flagged = [
        row
        for row in diagnostics
        if row["translation_error_m"] >= translation_threshold_m
        or row["rotation_error_deg"] >= rotation_threshold_deg
    ]
    print(
        f"Outlier thresholds: translation >= {translation_threshold_m:g} m, "
        f"rotation >= {rotation_threshold_deg:g} deg"
    )
    print(f"Flagged samples: {len(flagged)} / {len(diagnostics)}")
    print()
    print(f"Top {min(top, len(diagnostics))} samples by translation error:")
    print(f"{'rank':>4}  {'id':<10} {'trans_m':>10} {'rot_deg':>10}  files")
    for rank, row in enumerate(diagnostics[:top], start=1):
        files = ", ".join(
            value
            for value in (row.get("base_ee_file"), row.get("marker_file"))
            if value
        )
        print(
            f"{rank:>4}  {row['id']:<10} "
            f"{row['translation_error_m']:>10.4f} "
            f"{row['rotation_error_deg']:>10.2f}  {files}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
