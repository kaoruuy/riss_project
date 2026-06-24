# RISS Project

This repository contains tools for robot manipulation experiments:

- INSPIRE RH56E2-2R-T2 hand control over Modbus TCP
- UFACTORY xArm TCP/joint control and base-to-end-effector tracking
- ZED camera capture, ArUco pose estimation, hand-eye calibration, and camera recovery
- ZED-based physical-property estimation for manipulation planning

Run commands from the repository root.

## Folder Guides

- [hand/README.md](hand/README.md): INSPIRE hand network setup, status, angle control, tactile feedback, and raw actuator commands.
- [arm/README.md](arm/README.md): safe xArm wrapper, TCP/joint motion, and `T_base_ee` tracking.
- [camera/README.md](camera/README.md): shared ZED settings, point clouds, ArUco tools, hand-eye calibration, and table-marker recovery.
- [calibration/README.md](calibration/README.md): calibration files, fitted outputs, and sample dataset notes.
- [property_estimation/README.md](property_estimation/README.md): ZED physical-property estimator workflow.

## Shared Requirements

- Python 3.10+
- ZED camera and ZED SDK with Python API (`pyzed`)
- NumPy
- OpenCV with ArUco support (`opencv-contrib-python`)
- `xarm-python-sdk` for xArm control
- an OpenAI API key for `property_estimation`

Install project/runtime dependencies as needed in your virtual environment.
For xArm support:

```bash
python3 -m pip install xarm-python-sdk
```

For OpenCV ArUco support:

```bash
python3 -m pip install opencv-contrib-python
```

## Common Checks

Check the installed ZED API:

```bash
python3 camera/test.py
```

Read xArm state without moving the robot:

```bash
python3 -m arm.xarm_controller
```

Read INSPIRE hand status without moving the hand:

```bash
python3 -m hand.cli status
```

## Tests

The tests do not require a camera or API key. Run them from the repository root:

```bash
make test
```
