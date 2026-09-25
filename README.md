# Edge Vision ML Integration

A compact reconstruction of an edge-AI image-processing pipeline developed as part of an autonomous-systems engineering project.

## Public demonstration

This repository contains a small public reconstruction of the image-processing pipeline developed during the original engineering project.

The demonstration is intentionally limited to the core integration concept rather than reproducing the complete project source.

The pipeline combines:

1. YOLO object detection
2. OpenCV-based ArUco detection
3. Project-specific gauge interpretation
4. Annotated image/video output
5. Structured JSON results suitable for use by another subsystem

## Object classes

The original object-detection model used six labelled classes:

- `Center`
- `Closed`
- `Gauge`
- `Marker`
- `Open`
- `Tip`

The demonstration accepts an image, video file or webcam stream and passes each frame through the model.

Detected objects are converted into structured results while conventional computer-vision processing is used where it is better suited than machine learning.

## Image-processing pipeline

```text
Image / video / camera
          |
          v
      YOLO model
          |
          +----> object class
          +----> confidence
          +----> bounding box
          |
          v
 OpenCV post-processing
          |
          +----> ArUco detection
          +----> gauge interpretation
          +----> visual annotation
          |
          v
   Structured JSON output
```

## Gauge interpretation

For analogue gauge processing, model-detected `Center` and `Tip` objects are used to estimate the gauge-needle angle.

A project-specific calibration converts the angle to pressure in bar:

```text
pressure = 0.0344 * angle - 1.2
```

The calculated pressure is then evaluated against the original system threshold of approximately 2.07 bar.

This repository does not reproduce the external OpenCV analog-gauge-reader implementation used during the original project.

## ArUco marker detection

ArUco marker detection is performed separately using OpenCV rather than the ML model.

This reflects an engineering decision from the original project: marker detection was better handled using an established deterministic computer-vision algorithm rather than relying on the trained object detector.

## Installation

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Example usage

Run inference on an image:

```bash
python src/inference_pipeline.py \
    --model path/to/model.pt \
    --source examples/test_image.jpg \
    --save outputs/result.jpg
```

Run on a webcam:

```bash
python src/inference_pipeline.py \
    --model path/to/model.pt \
    --source 0
```

Press `q` to stop video or webcam inference.

## Example structured output

```json
{
  "detections": [
    {
      "class": "open",
      "confidence": 0.91,
      "xyxy": [120.4, 84.2, 281.8, 330.1],
      "center": [201.1, 207.2]
    }
  ],
  "aruco_ids": [4],
  "gauge": {
    "angle_deg": 95.1,
    "pressure_bar": 2.07,
    "below_2_07_bar": false
  }
}
```

## Repository scope

This repository is a public reconstruction of the original image-processing architecture rather than a copy of the complete engineering project.

The original trained model weights are not included.

The implementation is designed to operate with compatible YOLO weights trained using the documented six-class dataset.

The original project also included embedded hardware, networking, web visualisation and other subsystems that are outside the scope of this small demonstration.

## Code provenance

The original engineering project used a combination of project-specific Python code, open-source examples and vendor SDKs.

This repository does not present vendor or third-party examples as original work.

External resources used during the original project are documented in `references/SOURCES.md`.

No original project reports, credentials, proprietary documentation or complete project source code are included.
