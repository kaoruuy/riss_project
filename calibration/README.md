# Calibration Artifacts

This folder stores camera intrinsics, ArUco defaults, hand-eye samples, and
fitted transform outputs.

## Canonical Files

- `zed_intrinsics.yaml`: ZED Mini intrinsics for serial `14778242`, copied from
  `/usr/local/zed/settings/SN14778242.conf`.
- `aruco_config.yaml`: default ArUco dictionary, marker size, and camera
  parameters for PnP.
- `base_to_camera.yaml`: fitted `T_base_cam` from multi-sample hand-eye
  calibration.
- `ee_marker_estimated.yaml`: fitted `T_ee_marker` from multi-sample hand-eye
  calibration.
- `base_ee.yaml`: current/live xArm `T_base_ee`; this changes when the robot
  moves and is not a fitted calibration result.
- `hand_eye_samples.yaml`: dataset built from paired sample YAML files.

## Sample Data

`samples/` contains synchronized hand-eye captures:

```text
pose_001.png
pose_001_annotated.png
pose_001_marker.yaml
pose_001_base_ee.yaml
```

The marker YAML stores `T_cam_marker`; the base-EE YAML stores the matching
`T_base_ee`. The annotated PNG shows detected marker corners plus the selected
marker axes for inspection; keep the raw `pose_001.png` as the calibration
image. New captures also include `zed_settings`, and the dataset builder
rejects mismatched ZED settings before fitting.

## Notes

`transforms.yaml` is intentionally not the canonical calibration output. It can
represent a stale single-observation bundle. Use `base_to_camera.yaml` and
`ee_marker_estimated.yaml` for fitted transforms.

See [../camera/README.md](../camera/README.md) for the capture, fitting,
diagnostic, and table-marker recovery commands.
