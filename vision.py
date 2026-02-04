"""
Object Detection Module
Uses YOLOv8 for real-time object detection with CPU optimization
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, Dict


class ObjectDetector:
    """Real-time object detection using YOLOv8"""

    def __init__(self, model_name: str = "yolov8n", confidence_threshold: float = 0.5):
        """
        Initialize YOLO detector

        Args:
            model_name: YOLOv8 model size (nano, small, medium, large)
            confidence_threshold: Minimum confidence score for detections
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = "cpu"

        print(f"Loading {model_name} model on {self.device}...")
        self.model = YOLO(f"{model_name}.pt")
        self.model.to(self.device)
        print("✓ Model loaded successfully")

        self.class_names = self.model.names
        print(f"✓ Detected {len(self.class_names)} object classes")

    def detect(self, frame: np.ndarray, iou_threshold: float = 0.4) -> Dict:
        """
        Detect objects in a frame

        Args:
            frame: Input image/frame (BGR)
            iou_threshold: IOU threshold for NMS

        Returns:
            Dictionary containing detections
        """
        try:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )

            detections = {
                "boxes": [],
                "class_names": [],
                "confidences": [],
                "class_ids": []
            }

            if results and len(results) > 0:
                boxes = results[0].boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.class_names[class_id]

                    detections["boxes"].append([x1, y1, x2, y2])
                    detections["class_names"].append(class_name)
                    detections["confidences"].append(confidence)
                    detections["class_ids"].append(class_id)

            return detections

        except Exception as e:
            print(f"Error during detection: {e}")
            return {
                "boxes": [],
                "class_names": [],
                "confidences": [],
                "class_ids": []
            }

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: Dict,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.6,
        show_confidence: bool = True
    ) -> np.ndarray:
        """Draw detection boxes and labels on frame"""
        output_frame = frame.copy()

        for box, class_name, confidence in zip(
            detections["boxes"], detections["class_names"], detections["confidences"]
        ):
            x1, y1, x2, y2 = box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), box_color, thickness)

            if show_confidence:
                label = f"{class_name} {confidence:.2f}"
            else:
                label = class_name

            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, font_scale, 1)[0]

            label_y = y1 - 5 if y1 > 25 else y2 + 20
            cv2.rectangle(
                output_frame,
                (x1, label_y - text_size[1] - 5),
                (x1 + text_size[0] + 5, label_y + 5),
                box_color,
                -1
            )

            cv2.putText(
                output_frame,
                label,
                (x1 + 2, label_y - 2),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return output_frame

    def get_detection_summary(self, detections: Dict) -> str:
        """Get human-readable summary of detections"""
        if not detections["class_names"]:
            return "No objects detected"

        unique_classes = {}
        for class_name in detections["class_names"]:
            unique_classes[class_name] = unique_classes.get(class_name, 0) + 1

        class_counts = ", ".join(
            [f"{count} {name}{'s' if count > 1 else ''}"
             for name, count in sorted(unique_classes.items())]
        )

        total = len(detections["class_names"])
        return f"Detected {total} object{'s' if total != 1 else ''}: {class_counts}"
