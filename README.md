## INSPIRE RH56E2-2R-T2 command-line control
```bash
cd src
```

The hand CLI uses Modbus TCP at the existing hand address
`192.168.11.210:6000`. Read status without moving the hand:

```bash
python3 -m hand.cli status
```

The computer must have an Ethernet address on the hand's subnet. For example,
replace `<interface>` with the interface connected to the hand:

```bash
sudo ip link set <interface> up
sudo ip addr add 192.168.11.50/24 dev <interface>
ping -c 3 192.168.11.210
```

Send the recommended normalized angle targets in this six-DOF order:
`pinky ring middle index thumb_bend thumb_rotation`.

```bash
# Open the hand at a conservative speed
python3 -m hand.cli angle --speed 200 1000 1000 1000 1000 1000 0

# Close the fingers, leaving thumb rotation unchanged
python3 -m hand.cli angle --speed 200 0 0 0 0 0 -1
```

The manufacturer discourages direct actuator-position control. It is available
only with explicit acknowledgement; values are `0..2000`, where `0` is minimum
actuator stroke/open and `2000` is maximum stroke/bent:

```bash
python3 -m hand.cli position --allow-raw-position --speed 100 \
  500 500 500 500 500 500
```

Override the connection settings before the command when needed:

```bash
python3 -m hand.cli --host 192.168.11.210 --port 6000 --unit-id 1 status
```

Visualize the hand's six signed force-feedback channels without moving it:

```bash
# Continuously refreshing terminal bars; stop with Ctrl-C
python3 -m hand.cli tactile

# Print one sample or adjust the displayed full-scale magnitude
python3 -m hand.cli tactile --once
python3 -m hand.cli tactile --scale 500 --interval 0.05
```

These channels are the hand controller's `FORCE_ACT` feedback, aligned with the
six command channels. They are raw signed controller values, not calibrated
Newtons or a spatial taxel map.

## ZED Visual Physical-Property Estimator

This prototype captures left/right images and depth statistics from a ZED camera,
uses a vision-language model to estimate latent physical properties, and fuses
repeated observations into a more stable estimate.

The output includes:

- material candidates and an uncertain mass range
- softness, rigidity, roughness, slipperiness, deformability, and fragility
- a confidence and visible-evidence explanation for every property
- conservative grasp-force and manipulation recommendations

Visual predictions are priors, not measurements. A robot controller should keep
force/torque limits active and refine these estimates with tactile feedback.

### Requirements

- ZED camera and ZED SDK with Python API (`pyzed`)
- Python 3.10+
- NumPy
- an OpenAI API key

The installed ZED API can be checked with:

```bash
python3 camera/test.py
```

### Run

```bash
set -a
source .env
set +a
python3 -m physical_properties.cli --observations 3 --interval 5 \
  --output estimates.jsonl
```

The application allows the ZED SDK up to 30 seconds to open and internally
retry the camera before reporting failure. Override this with
`--camera-open-timeout` when needed.

Use `--observations 0` to run until Ctrl-C. The default model is
`gpt-5.4-mini`; override it with `--model`.

For best results, put one object near the center of the image, include a known
size reference when mass matters, and capture observations from different
viewpoints and lighting conditions.

### Test

```bash
python3 -m unittest discover -s tests -v
```
The tests do not require a camera or API key.
