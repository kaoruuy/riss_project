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

## Camera Calibration

Calibration files live in `calibration/`:

- `zed_intrinsics.yaml` stores the current ZED Mini intrinsics for serial
  `14778242`, copied from `/usr/local/zed/settings/SN14778242.conf`.
- `aruco_config.yaml` stores the default ArUco/PnP camera parameters.
- `base_to_camera.yaml` is reserved for a robot-base-to-camera transform. It is
  still marked `calibrated: false` because the ZED SDK settings do not contain
  that robot extrinsic.

Estimate the transform between the ZED left camera and an ArUco marker in an
image:

```bash
python3 -m camera.aruco_pose image.png \
  --dictionary DICT_6X6_250 \
  --marker-length-m 0.05 \
  --pretty
```

If the image came from the rectified ZED SDK stream, ignore the raw distortion
coefficients:

```bash
python3 -m camera.aruco_pose image.png \
  --dictionary DICT_6X6_250 \
  --marker-length-m 0.05 \
  --ignore-distortion \
  --pretty
```

The output includes both `marker_to_camera_matrix` from OpenCV PnP and the
inverted `camera_to_marker_matrix`.

## Tests

The tests do not require a camera or API key. Run them from the repository root:

```bash
make test
```
