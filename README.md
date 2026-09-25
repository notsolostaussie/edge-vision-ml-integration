# Edge Vision ML Integration

A compact reconstruction of an edge-AI image-processing pipeline developed
as part of an autonomous-systems engineering project.

The original subsystem was designed to process live UAV imagery to identify
open/closed lever valves, interpret analogue pressure gauges, detect ArUco
markers and provide processed visual information to downstream system
components.

## Public demonstration

This repository contains a small public reconstruction of the image-processing
pipeline developed during the original engineering project.

The demonstration is intentionally limited to the core integration concept
rather than reproducing the complete project source.

The pipeline combines:

1. YOLO object detection
2. OpenCV-based ArUco detection
3. Project-specific gauge interpretation
4. Annotated image/video output
5. Structured JSON results suitable for use by another subsystem

## Original project context

The original image-processing subsystem formed part of a UAV payload system
using a Raspberry Pi and OAK-D Lite camera.

The subsystem was required to:

- acquire and process live imagery;
- detect and classify target objects;
- distinguish open and closed valve states;
- interpret analogue pressure gauges;
- detect ArUco markers;
- provide annotated video and target information to other system components.

The original software architecture used separate Python functions and scripts
running on Raspberry Pi OS, with the image-processing subsystem integrated
with other project subsystems through defined data interfaces.

## Object classes

The original object-detection model used six labelled classes:

- `Center`
- `Closed`
- `Gauge`
- `Marker`
- `Open`
- `Tip`

The demonstration accepts an image, video file or webcam stream and passes
each frame through the model.

Detected objects are converted into structured results while conventional
computer-vision processing is used where it is better suited than machine
learning.

## Dataset and model development

The original YOLOv5 model was developed using images captured with the
OAK-D Lite camera under conditions representative of the intended test
environment.

The dataset was labelled for valve state, gauge components and marker
detection, then iteratively reviewed after early testing identified uneven
representation between some classes.

Additional training images were introduced to improve representation of the
`Open` class. ArUco marker detection was subsequently handled using OpenCV
rather than relying on the ML `Marker` class, providing a more appropriate
deterministic solution for that task.

The complete training dataset and original model weights are not included in
this repository.

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
The original system followed the same general processing flow: live OAK-D
video was passed through the object-detection model, followed by OpenCV
processing for ArUco detection, gauge-value conversion and visual annotation
before results were made available to downstream interfaces.

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

## Testing and validation

The original subsystem was tested at both unit and integration level.

Testing included:
- YOLOv5 model training and validation;
- physical testing against real target imagery;
- valve open/closed classification;
- analogue gauge interpretation;
- ArUco marker detection;
- live video transmission;
- integration with downstream project functions.

Individual tests demonstrated working object detection, valve-state
classification and gauge processing. Live video transmission and Raspberry Pi
operation were also demonstrated during integration testing.

Not all planned end-to-end functionality was successfully demonstrated during
the final acceptance test.

## Engineering lessons

A key limitation identified during final integration was reliance on an
external Roboflow authentication service.
Although individual image-processing functions had previously been tested,
failure of this external dependency prevented part of the object-detection
pipeline from operating during final system integration.

This highlighted several practical engineering lessons:
- avoid unnecessary single points of failure;
- distinguish successful model testing from successful system integration;
- validate deployment dependencies before final acceptance testing;
- provide diagnostic and fallback mechanisms for critical inference services;
- prefer local processing where system reliability requires it.

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
