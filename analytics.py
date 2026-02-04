"""
Analytics and Logging Module
Tracks detections and generates statistics
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict


class AnalyticsLogger:
    """Log and analyze detection events"""

    def __init__(self, log_file: str = "logs/detections.log", stats_file: str = "logs/statistics.json"):
        self.log_file = log_file
        self.stats_file = stats_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.frame_count = 0
        self.total_detections = 0
        self.detections_per_class = defaultdict(int)
        self.confidence_scores = defaultdict(list)
        self.processing_times = []
        self.fps_history = []

        print("✓ Analytics logger initialized")
        print(f"  Log file: {log_file}")
        print(f"  Stats file: {stats_file}")

    def log_detection(self, frame_id: int, detections: Dict, timestamp: Optional[datetime] = None) -> None:
        if timestamp is None:
            timestamp = datetime.now()

        if not detections["class_names"]:
            return

        try:
            detection_summary = ", ".join(detections["class_names"])
            log_entry = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Frame {frame_id}: {detection_summary}\n"

            with open(self.log_file, 'a') as f:
                f.write(log_entry)

            self.frame_count += 1
            self.total_detections += len(detections["class_names"])

            for class_name, confidence in zip(detections["class_names"], detections["confidences"]):
                self.detections_per_class[class_name] += 1
                self.confidence_scores[class_name].append(confidence)

        except Exception as e:
            print(f"Error logging detection: {e}")

    def log_processing_time(self, processing_time: float) -> None:
        self.processing_times.append(processing_time)

    def log_fps(self, fps: float) -> None:
        self.fps_history.append(fps)

    def get_statistics(self) -> Dict:
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0
        )

        avg_fps = (
            sum(self.fps_history) / len(self.fps_history)
            if self.fps_history else 0
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "frame_count": self.frame_count,
            "total_detections": self.total_detections,
            "detections_per_class": dict(self.detections_per_class),
            "average_confidence_per_class": {
                class_name: sum(scores) / len(scores)
                for class_name, scores in self.confidence_scores.items()
                if scores
            },
            "average_processing_time_ms": avg_processing_time * 1000,
            "average_fps": avg_fps,
            "unique_classes": len(self.detections_per_class)
        }

    def save_statistics(self) -> None:
        try:
            stats = self.get_statistics()
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)

            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

            print(f"✓ Statistics saved to {self.stats_file}")

        except Exception as e:
            print(f"Error saving statistics: {e}")

    def print_statistics(self) -> None:
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("SESSION STATISTICS")
        print("="*60)
        print(f"Total Frames Processed: {stats['frame_count']}")
        print(f"Total Detections: {stats['total_detections']}")
        print(f"Unique Classes: {stats['unique_classes']}")
        print(f"Average Processing Time: {stats['average_processing_time_ms']:.2f} ms")
        print(f"Average FPS: {stats['average_fps']:.2f}")

        if stats['detections_per_class']:
            print("\nDetections Per Class:")
            for class_name, count in sorted(
                stats['detections_per_class'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                avg_conf = stats['average_confidence_per_class'].get(class_name, 0)
                print(f"  {class_name}: {count} detections (avg confidence: {avg_conf:.2f})")

        print("="*60 + "\n")

    def reset(self) -> None:
        self.frame_count = 0
        self.total_detections = 0
        self.detections_per_class.clear()
        self.confidence_scores.clear()
        self.processing_times.clear()
        self.fps_history.clear()
        print("✓ Analytics reset")

    def clear_logs(self) -> None:
        try:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
                print(f"✓ Log file cleared: {self.log_file}")
        except Exception as e:
            print(f"Error clearing log: {e}")
