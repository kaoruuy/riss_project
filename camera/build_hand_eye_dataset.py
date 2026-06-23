"""Build hand-eye calibration sample YAML from captured pose_XXX files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_INPUT_DIR = Path("calibration/samples")
DEFAULT_OUTPUT = Path("calibration/hand_eye_samples.yaml")
BASE_EE_RE = re.compile(r"^(pose_\d{3})_base_ee\.yaml$")
MARKER_RE = re.compile(r"^(pose_\d{3})_marker\.yaml$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--marker-id",
        type=int,
        default=0,
        help="required ArUco marker id for hand-eye samples; defaults to hand marker 0",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="write an empty dataset instead of failing when no pairs are found",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        samples = build_dataset(args.input_dir, marker_id=args.marker_id)
        if not samples and not args.allow_empty:
            raise ValueError(f"no paired samples found in {args.input_dir}")
        write_dataset(args.output, samples, args.input_dir)
        print(f"Wrote {len(samples)} paired samples to {args.output}")
        return 0
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"build-hand-eye-dataset: {exc}", file=sys.stderr)
        return 1


def build_dataset(input_dir: Path, marker_id: int | None = 0) -> list[dict[str, Any]]:
    pairs = find_sample_pairs(input_dir)
    samples = []
    for stem, files in pairs:
        base_ee = load_yaml(files["base_ee"])
        marker = load_yaml(files["marker"])
        detected_marker_id = marker.get("marker_id")
        if marker_id is not None and detected_marker_id != marker_id:
            raise ValueError(
                f"{files['marker']} marker_id is {detected_marker_id}; expected hand marker {marker_id}"
            )
        samples.append(
            {
                "id": stem,
                "image": marker.get("image"),
                "marker_id": detected_marker_id,
                "base_ee_file": str(files["base_ee"]),
                "marker_file": str(files["marker"]),
                "T_base_ee": as_transform_matrix(base_ee.get("T_base_ee"), f"{files['base_ee']} T_base_ee").tolist(),
                "T_cam_marker": as_transform_matrix(marker.get("T_cam_marker"), f"{files['marker']} T_cam_marker").tolist(),
            }
        )
    return samples


def find_sample_pairs(input_dir: Path) -> list[tuple[str, dict[str, Path]]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"sample directory does not exist: {input_dir}")

    base_files: dict[str, Path] = {}
    marker_files: dict[str, Path] = {}
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        base_match = BASE_EE_RE.match(path.name)
        if base_match:
            base_files[base_match.group(1)] = path
            continue
        marker_match = MARKER_RE.match(path.name)
        if marker_match:
            marker_files[marker_match.group(1)] = path

    stems = sorted(set(base_files) & set(marker_files))
    return [
        (stem, {"base_ee": base_files[stem], "marker": marker_files[stem]})
        for stem in stems
    ]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def as_transform_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite numbers")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")
    return matrix


def write_dataset(output: Path, samples: list[dict[str, Any]], input_dir: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "source_dir": str(input_dir),
        "sample_count": len(samples),
        "samples": samples,
    }
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(document, file, sort_keys=False)


if __name__ == "__main__":
    raise SystemExit(main())
