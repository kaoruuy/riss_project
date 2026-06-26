# Camera, ZED, And Calibration Tools

This folder contains ZED capture utilities, ArUco pose estimation, hand-eye
calibration, table-marker camera recovery, and point-cloud export tools.

Run commands from the repository root.

## Shared ZED Settings

All ZED scripts use the shared settings in `camera.zed_config` by default:
`HD720`, `30` FPS, `NEURAL` depth mode, meter units, `IMAGE` coordinate system,
rectified `LEFT` image stream, and `calibration/zed_intrinsics.yaml`.

Camera frame convention:

```text
+X = image right
+Y = image down
+Z = forward from camera
```

This is the OpenCV optical image convention and matches the point cloud
projection in `camera.zed_pointcloud`: `x=(u-cx)z/fx`, `y=(v-cy)z/fy`,
`z=depth`. Keep these settings identical during hand-eye calibration, live
ArUco viewing, point cloud capture, and grasping. Hand-eye sample YAML files
include a `zed_settings` block so mismatched sample captures can be detected
before fitting.

Fitted calibration YAML files also include quick metadata blocks:

```yaml
calibration:
  date: 2026-06-26
  sample_count: 34
zed:
  sdk_version: 5.3.1
  resolution: HD720
  fps: 30
  depth_mode: NEURAL
  coordinate_units: METER
  coordinate_system: IMAGE
results:
  translation_rms_m: 0.0168
  rotation_rms_deg: 3.91
```

Check the installed ZED API:

```bash
python3 camera/test.py
```

## Point Cloud Capture

Capture left RGB plus depth from the ZED and save a colored point cloud:

```bash
python3 -m camera.zed_pointcloud \
  --output zed_pointcloud.ply \
  --stride 2 \
  --max-depth-m 3.0
```

The command projects each valid depth pixel into XYZ using the active ZED left
camera intrinsics and writes RGB colors from the left image. Use `.ply` for a
portable ASCII point cloud or `.npz` for compressed NumPy arrays containing
`points` and `colors`.

## Tabletop Grasp Candidate

Detect the tabletop plane with RANSAC, segment object points as bumps above the
plane, and print the largest object cluster center in both camera and robot base
frames:

```bash
python3 -m camera.ransac_grasp_candidate \
  --base-to-camera calibration/base_to_camera.yaml \
  --object-min-height-m 0.01 \
  --object-max-height-m 0.08
```

The script uses the shared ZED settings, projects depth using the IMAGE camera
convention, transforms points with `T_base_cam`, fits the dominant plane in the
robot base frame, and visualizes the RGB image with the projected object center.
Press `q` or `Esc` to quit.

Robot motion is disabled by default. To move the xArm to a safe approach pose
above the detected center, pass `--move`; the command keeps the current TCP
orientation, raises `z` by `--approach-height-m` (default `0.08` m), checks the
workspace limits, enforces `--max-speed`, and still requires typing `MOVE`:

```bash
python3 -m camera.ransac_grasp_candidate \
  --move \
  --workspace-x -0.30 0.80 \
  --workspace-y -0.80 0.80 \
  --workspace-z 0.02 0.80 \
  --speed 30
```

## Calibration Files

Calibration files live in `calibration/`:

- `zed_intrinsics.yaml` stores the current ZED Mini intrinsics for serial
  `14778242`, copied from `/usr/local/zed/settings/SN14778242.conf`.
- `aruco_config.yaml` stores the default ArUco/PnP camera parameters.
- `base_to_camera.yaml` stores the fitted robot-base-to-camera transform after
  multi-sample hand-eye calibration.

## ArUco Marker Tools

Generate a printable ArUco marker:

```bash
python3 -m camera.aruco_generator 0 \
  --dictionary DICT_4X4_50 \
  --output aruco_0.png \
  --size-px 800 \
  --border-px 100
```

Estimate the transform between the ZED left camera and an ArUco marker in an
image:

```bash
python3 -m camera.aruco_pose aruco_markers/aruco_0.png \
  --dictionary DICT_4X4_50 \
  --marker-length-m 0.05 \
  --pretty
```

If the image came from the rectified ZED SDK stream, ignore the raw distortion
coefficients:

```bash
python3 -m camera.aruco_pose aruco_markers/aruco_0.png \
  --dictionary DICT_4X4_50 \
  --marker-length-m 0.05 \
  --ignore-distortion \
  --pretty
```

The output includes both `marker_to_camera_matrix` from OpenCV PnP and the
inverted `camera_to_marker_matrix`.

View the marker XYZ frame live from the ZED camera:

```bash
python3 -m camera.aruco_live_viewer \
  --dictionary DICT_4X4_50 \
  --marker-id 0 \
  --marker-length-m 0.05 \
  --axis-length-m 0.03 \
  --ignore-distortion
```

The live viewer draws detected marker corners, marker ID text, and XYZ axes
using OpenCV `drawFrameAxes`. Press `q` or `Esc` to quit. Use
`--source webcam --camera-index 0` to view a standard webcam instead of the
ZED stream. It prints `T_cam_marker` periodically so the terminal remains
readable while the video window updates in real time.

Solve one detected marker pose directly in robot base coordinates to verify the
calibration transform:

```bash
python3 -m camera.solve_marker_base_pose \
  calibration/table_marker_references.png \
  --marker-id 1 \
  --pretty
```

The script computes:

```text
T_base_marker = T_base_cam @ T_cam_marker
```

Use the same distortion setting that was used when the reference image was
captured; changing `--ignore-distortion` changes the PnP result.

## Hand-Eye Calibration

`calibration/transforms.yaml` is not a canonical calibration file anymore.
Single-observation transform bundles can become stale and conflict with the
multi-sample hand-eye result. Keep the final robot-to-camera calibration in
`calibration/base_to_camera.yaml`, and keep the estimated marker mount in
`calibration/ee_marker_estimated.yaml`.

Collect multiple synchronized samples and fit all of them together:

```bash
python3 -m camera.capture_aruco_sample
```

By default this command only accepts ArUco marker ID `0`, which should be the
marker mounted on the hand/end-effector. Use table markers with IDs `1`-`4`
only for recovery references, not for the hand-eye calibration dataset.

Each run captures a fresh ZED left image and freezes the current xArm pose with
the same monotonically increasing index:

```text
calibration/samples/pose_001.png
calibration/samples/pose_001_annotated.png
calibration/samples/pose_001_marker.yaml
calibration/samples/pose_001_base_ee.yaml
```

The marker YAML contains `T_cam_marker`; the base-EE YAML contains the matching
`T_base_ee`. The annotated PNG draws all detected marker corners plus the
selected hand marker ID and XYZ axes, while the raw `pose_001.png` remains
unchanged for calibration. This prevents accidentally reusing an old image with
a new arm pose.

After collecting several poses, build the dataset consumed by the fitter:

```bash
python3 -m camera.build_hand_eye_dataset \
  --input-dir calibration/samples \
  --output calibration/hand_eye_samples.yaml
```

The dataset builder also defaults to `--marker-id 0` and rejects captured
samples from any other marker ID.

```bash
python3 -m camera.fit_hand_eye \
  --samples calibration/hand_eye_samples.yaml \
  --base-to-camera-output calibration/base_to_camera.yaml \
  --ee-marker-output calibration/ee_marker_estimated.yaml \
  --pretty
```

The sample file must contain at least three paired observations. Each
`T_base_ee` must be a frozen xArm pose snapshot captured at the same time as
the corresponding image-derived `T_cam_marker`.

The fitter uses OpenCV `calibrateRobotWorldHandEye`, treating the ArUco marker
as the world/calibration target. It estimates both `T_base_cam` and
`T_ee_marker`, then saves the final transforms to the output YAML files.

Inspect per-sample residuals to find bad captures before refitting:

```bash
python3 -m camera.diagnose_hand_eye \
  --samples calibration/hand_eye_samples.yaml \
  --top 12
```

The diagnostic refits the dataset, uses the selected transform convention, and
sorts samples by translation error while also showing rotation error. Remove or
recapture high-error pose pairs, rebuild `hand_eye_samples.yaml`, and refit.

## Camera Recovery From Table Markers

After hand-eye calibration succeeds and while the camera is still in the
calibrated pose, uncover the fixed table markers. Use marker ID `0` only on the
hand, and table marker IDs `1`, `2`, `3`, and `4` on the table.

Capture one table-marker reference image and save the fixed marker poses in the
robot base frame:

```bash
python3 -m camera.capture_table_references \
  --base-to-camera calibration/base_to_camera.yaml \
  --output calibration/table_marker_references.yaml
```

This detects IDs `1`-`4`, computes `T_base_table_1` through
`T_base_table_4` as:

```text
T_base_table_i = T_base_cam @ T_cam_table_i
```

and stores them in `calibration/table_marker_references.yaml`.

If the camera moves later, recover a fresh `T_base_cam` without redoing
hand-eye calibration:

```bash
python3 -m camera.recover_base_to_camera \
  --references calibration/table_marker_references.yaml \
  --output calibration/base_to_camera_recovered.yaml
```

The recovery script detects the visible table markers and computes each camera
estimate as:

```text
T_base_cam_i = T_base_table_i @ inverse(T_cam_table_i)
```

When more than one referenced marker is visible, it averages the estimates and
reports per-marker residuals in the output YAML.
