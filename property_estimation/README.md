# ZED Physical-Property Estimator

This prototype captures left/right images and depth statistics from a ZED
camera, uses a vision-language model to estimate latent physical properties,
and fuses repeated observations into a more stable estimate.

Run commands from the repository root.

The output includes:

- material candidates and an uncertain mass range
- softness, rigidity, roughness, slipperiness, deformability, and fragility
- confidence and visible-evidence explanations for every property
- conservative grasp-force and manipulation recommendations

Visual predictions are priors, not measurements. Keep robot force and torque
limits active, and refine estimates with tactile feedback.

## Requirements

- Python 3.10+
- ZED camera and ZED SDK with Python API (`pyzed`)
- NumPy
- OpenCV with ArUco support for marker pose estimation (`opencv-contrib-python`)
- an OpenAI API key

Check the installed ZED API:

```bash
python3 camera/test.py
```

## Run The Estimator

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
