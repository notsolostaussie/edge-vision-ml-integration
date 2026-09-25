"""Edge-vision ML integration demonstration."""

import argparse
import json
import math
from pathlib import Path

import cv2
from ultralytics import YOLO

VALID_CLASSES = {"center", "closed", "gauge", "marker", "open", "tip"}
GAUGE_THRESHOLD_BAR = 2.07


def box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def gauge_angle(center, tip):
    dx = tip[0] - center[0]
    dy = center[1] - tip[1]
    angle_from_x = math.degrees(math.atan2(dy, dx)) % 360.0
    return (270.0 - angle_from_x) % 360.0


def gauge_angle_to_bar(theta):
    return 0.0344 * theta - 1.2


def detect_aruco(frame, annotated):
    if not hasattr(cv2, "aruco"):
        return []

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if hasattr(aruco, "ArucoDetector"):
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        params = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(
            gray, dictionary, parameters=params
        )

    if ids is None:
        return []

    aruco.drawDetectedMarkers(annotated, corners, ids)
    return ids.flatten().astype(int).tolist()


def parse_detections(result):
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        label = str(result.names[class_id]).lower()

        if label not in VALID_CLASSES:
            continue

        xyxy = box.xyxy[0].cpu().tolist()
        centre = box_center(xyxy)

        detections.append({
            "class": label,
            "confidence": round(float(box.conf[0]), 3),
            "xyxy": [round(v, 1) for v in xyxy],
            "center": [round(v, 1) for v in centre],
        })

    return detections


def calculate_gauge(detections):
    centres = [d for d in detections if d["class"] == "center"]
    tips = [d for d in detections if d["class"] == "tip"]

    if not centres or not tips:
        return None

    centre = max(centres, key=lambda d: d["confidence"])["center"]
    tip = max(tips, key=lambda d: d["confidence"])["center"]

    theta = gauge_angle(centre, tip)
    pressure = gauge_angle_to_bar(theta)

    return {
        "angle_deg": round(theta, 1),
        "pressure_bar": round(pressure, 2),
        "below_2_07_bar": pressure < GAUGE_THRESHOLD_BAR,
    }


def process_frame(model, frame):
    result = model.predict(frame, verbose=False)[0]
    detections = parse_detections(result)
    annotated = result.plot()

    marker_ids = detect_aruco(frame, annotated)
    gauge = calculate_gauge(detections)

    if gauge is not None:
        cv2.putText(
            annotated,
            f"Gauge: {gauge['pressure_bar']:.2f} bar",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated, {
        "detections": detections,
        "aruco_ids": marker_ids,
        "gauge": gauge,
    }


def open_source(source):
    if source.isdigit():
        return cv2.VideoCapture(int(source)), None

    path = Path(source)
    image = cv2.imread(str(path))

    if image is not None:
        return None, image

    return cv2.VideoCapture(str(path)), None


def main():
    parser = argparse.ArgumentParser(
        description="Edge-vision ML integration demonstration"
    )
    parser.add_argument("--model", required=True, help="YOLO .pt weights")
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index, image path or video path",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional annotated image output",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    capture, image = open_source(args.source)

    if image is not None:
        annotated, output = process_frame(model, image)
        print(json.dumps(output, indent=2))

        if args.save:
            output_path = Path(args.save)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)

        cv2.imshow("Edge Vision ML Integration", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    if capture is None or not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        annotated, output = process_frame(model, frame)
        print(json.dumps(output))
        cv2.imshow("Edge Vision ML Integration", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
