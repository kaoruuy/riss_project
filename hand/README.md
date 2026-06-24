# INSPIRE Hand Control

The hand CLI connects to `192.168.11.210:6000` by default.

Run commands from the repository root.

## Configure The Network

The computer must have an Ethernet address on the hand's subnet. Replace
`<interface>` with the interface connected to the hand:

```bash
sudo ip link set <interface> up
sudo ip addr add 192.168.11.50/24 dev <interface>
ping -c 3 192.168.11.210
```

## Read Hand Status

Read the current angles and actuator positions without moving the hand:

```bash
python3 -m hand.cli status
```

Override the default connection settings when needed:

```bash
python3 -m hand.cli \
  --host 192.168.11.210 --port 6000 --unit-id 1 status
```

## Control Finger Angles

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

## Read Force Feedback

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

## Raw Actuator Positions

> **Warning:** The manufacturer discourages direct actuator-position control.
> Prefer normalized angle commands unless raw positions are specifically needed.

Raw positions range from `0` (minimum actuator stroke/open) to `2000` (maximum
stroke/bent). The command requires explicit acknowledgement:

```bash
python3 -m hand.cli position \
  --allow-raw-position --speed 100 \
  500 500 500 500 500 500
```
