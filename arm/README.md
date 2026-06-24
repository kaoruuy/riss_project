# UFACTORY xArm Control

Install the official xArm Python SDK and add the robot IP address to `.env` as
`XARM_IP`.

Run commands from the repository root.

```bash
python3 -m pip install xarm-python-sdk
```

## Read Current State

Read the current TCP pose and joint angles without moving the robot:

```bash
python3 -m arm.xarm_controller
```

The wrapper connects with `is_radian=False`, so TCP positions are in
millimeters and angles are in degrees. On connection it clears warnings and
errors, enables motion, sets mode `0`, and sets state `0`.

## Move The Robot

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

## Track `T_base_ee`

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
