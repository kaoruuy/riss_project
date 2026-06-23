# RISS Project

This repository contains tools for:

- controlling an INSPIRE RH56E2-2R-T2 robotic hand over Modbus TCP
- estimating an object's physical properties from ZED stereo-camera observations

Run commands from the repository root.

## INSPIRE Hand Control

The hand CLI connects to `192.168.11.210:6000` by default.

### Configure the Network

The computer must have an Ethernet address on the hand's subnet. Replace
`<interface>` with the interface connected to the hand:

```bash
sudo ip link set <interface> up
sudo ip addr add 192.168.11.50/24 dev <interface>
ping -c 3 192.168.11.210
```

### Read Hand Status

Read the current angles and actuator positions without moving the hand:

```bash
python3 -m hand.cli status
```

Override the default connection settings when needed:

```bash
python3 -m hand.cli \
  --host 192.168.11.210 --port 6000 --unit-id 1 status
```

### Control Finger Angles

The recommended angle command accepts six normalized targets in this order:

```text
pinky ring middle index thumb_bend thumb_rotation
```

Each target ranges from `0` to `1000`. Use `-1` to leave a target unchanged.

```bash
# Open the hand at a conservative speed
python3 -m hand.cli angle --speed 200 \
  1000 1000 1000 1000 1000 0

# Close the fingers, leaving thumb rotation unchanged
python3 -m hand.cli angle --speed 200 \
  0 0 0 0 0 -1
```

### Read Force Feedback

Visualize the six signed `FORCE_ACT` feedback channels without moving the hand:

```bash
# Continuously refresh terminal bars; stop with Ctrl-C
python3 -m hand.cli tactile

# Print one sample
python3 -m hand.cli tactile --once

# Adjust the displayed full-scale magnitude and refresh interval
python3 -m hand.cli tactile --scale 500 --interval 0.05
```

These are raw signed controller values aligned with the six command channels.
They are not calibrated Newton measurements or a spatial taxel map.

### Raw Actuator Positions

> **Warning:** The manufacturer discourages direct actuator-position control.
> Prefer normalized angle commands unless raw positions are specifically needed.

Raw positions range from `0` (minimum actuator stroke/open) to `2000` (maximum
stroke/bent). The command requires explicit acknowledgement:

```bash
python3 -m hand.cli position \
  --allow-raw-position --speed 100 \
  500 500 500 500 500 500
```

## UFACTORY xArm Control

Install the official xArm Python SDK and add the robot IP address to `.env`:

```bash
python3 -m pip install xarm-python-sdk
```

Read the current TCP pose and joint angles without moving the robot:

```bash
python3 -m arm.xarm_controller
```

The wrapper connects with `is_radian=False`, so TCP positions are in
millimeters and angles are in degrees. On connection it clears warnings and
errors, enables motion, sets mode `0`, and sets state `0`.

Motion never runs by default. To command a Cartesian TCP pose, pass `--move`
and an explicit target. The program prints a warning and requires typing
`MOVE` before executing:

```bash
python3 -m arm.xarm_controller \
  --move \
  --tcp-pose 300 50 300 180 0 0 \
  --speed 10
```

Command joint angles in degrees the same way:

```bash
python3 -m arm.xarm_controller \
  --move \
  --joint-angles 0 -30 45 0 60 0 \
  --speed 10
```

Use the current TCP pose from `get_tcp_pose()` as the xArm-side
end-effector pose for hand-eye calibration, together with the camera/ArUco
transform and the INSPIRE hand setup.

Continuously write the current xArm TCP pose as `T_base_ee` for calibration:

```bash
python3 -m arm.base_ee_tracker \
  --output calibration/base_ee.yaml
```

This tracker uses `arm.xarm_controller`, so it reads `XARM_IP` from `.env`,
connects to the xArm, calls `get_tcp_pose()`, and writes:

- `translation_m`, converted from xArm millimeters to meters
- `rotation_quaternion_xyzw`, converted from xArm roll/pitch/yaw degrees
- `T_base_ee`, the 4x4 base-to-end-effector transform
- `xarm_tcp_pose_mm_deg`, the raw `[x, y, z, roll, pitch, yaw]` reading

Use `--once` to write one sample and exit:

```bash
python3 -m arm.base_ee_tracker \
  --output calibration/base_ee.yaml \
  --once
```

Keep `calibration/base_ee.yaml` separate from the fitted calibration outputs.
`base_ee.yaml` is the current xArm base-to-end-effector pose and may change
every time the arm moves. The fitted, stable calibration outputs are
`calibration/base_to_camera.yaml` and `calibration/ee_marker_estimated.yaml`.

## ZED Physical-Property Estimator

This prototype captures left/right images and depth statistics from a ZED
camera, uses a vision-language model to estimate latent physical properties,
and fuses repeated observations into a more stable estimate.

The output includes:

- material candidates and an uncertain mass range
- softness, rigidity, roughness, slipperiness, deformability, and fragility
- confidence and visible-evidence explanations for every property
- conservative grasp-force and manipulation recommendations

Visual predictions are priors, not measurements. Keep robot force and torque
limits active, and refine estimates with tactile feedback.

### Requirements

- Python 3.10+
- ZED camera and ZED SDK with Python API (`pyzed`)
- NumPy
- OpenCV with ArUco support for marker pose estimation (`opencv-contrib-python`)
- an OpenAI API key

Check the installed ZED API:

```bash
python3 camera/test.py
```

### Run the Estimator

Load environment variables from `.env`, then collect three observations:

```bash
set -a
source .env
set +a

python3 -m property_estimation.cli \
  --observations 3 --interval 5 --output estimates.jsonl
```

The application allows the ZED SDK up to 30 seconds to open and retry the
camera before reporting failure. Change this with `--camera-open-timeout`.

Useful options:

- `--observations 0` runs until stopped with Ctrl-C.
- `--model <model>` overrides the default model, `gpt-5.4-mini`.
- `--output <path>` appends fused estimates as JSON Lines.

For best results, place one object near the center of the image. Include a
known-size reference when mass matters, and capture observations from different
viewpoints and lighting conditions.

### Capture a Colored Point Cloud

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

## Camera Calibration

Calibration files live in `calibration/`:

- `zed_intrinsics.yaml` stores the current ZED Mini intrinsics for serial
  `14778242`, copied from `/usr/local/zed/settings/SN14778242.conf`.
- `aruco_config.yaml` stores the default ArUco/PnP camera parameters.
- `base_to_camera.yaml` is reserved for a robot-base-to-camera transform. It is
  still marked `calibrated: false` because the ZED SDK settings do not contain
  that robot extrinsic.

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

`calibration/transforms.yaml` is not a canonical calibration file anymore.
Single-observation transform bundles can become stale and conflict with the
multi-sample hand-eye result. Keep the final robot-to-camera calibration in
`calibration/base_to_camera.yaml`, and keep the estimated marker mount in
`calibration/ee_marker_estimated.yaml`.

For a stronger hand-eye calibration, collect multiple synchronized samples and
fit all of them together:

```bash
python3 -m camera.capture_aruco_sample
```

Each run captures a fresh ZED left image and freezes the current xArm pose with
the same monotonically increasing index:

```text
calibration/samples/pose_001.png
calibration/samples/pose_001_marker.yaml
calibration/samples/pose_001_base_ee.yaml
```

The marker YAML contains `T_cam_marker`; the base-EE YAML contains the matching
`T_base_ee`. This prevents accidentally reusing an old image with a new arm
pose.

After collecting several poses, build the dataset consumed by the fitter:

```bash
python3 -m camera.build_hand_eye_dataset \
  --input-dir calibration/samples \
  --output calibration/hand_eye_samples.yaml
```

```bash
python3 -m camera.fit_hand_eye \
  --samples calibration/hand_eye_samples.yaml \
  --base-to-camera-output calibration/base_to_camera.yaml \
  --ee-marker-output calibration/ee_marker_estimated.yaml \
  --pretty
```

The sample file must contain at least three paired observations. Each
`T_base_ee` must be a frozen xArm pose snapshot captured at the same time as
the corresponding image-derived `T_cam_marker`:

```yaml
samples:
  - T_base_ee:
      - [1.0, 0.0, 0.0, 0.30]
      - [0.0, 1.0, 0.0, 0.00]
      - [0.0, 0.0, 1.0, 0.20]
      - [0.0, 0.0, 0.0, 1.0]
    T_cam_marker:
      - [1.0, 0.0, 0.0, 0.02]
      - [0.0, 1.0, 0.0, 0.00]
      - [0.0, 0.0, 1.0, 0.45]
      - [0.0, 0.0, 0.0, 1.0]
```

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

## Tests

The tests do not require a camera or API key. Run them from the repository root:

```bash
make test
```
