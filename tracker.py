"""
Object Tracking Module
Tracks objects across frames using simple centroid tracking
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import math


class ObjectTracker:
    """Simple centroid-based object tracker"""

    def __init__(self, max_distance: float = 50.0, max_disappeared: int = 30):
        """Initialize tracker"""
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.next_id = 0
        self.tracks = {}
        self.track_history = defaultdict(list)
        self.class_counts = defaultdict(int)
        self.total_detections = 0

    def update(self, detections: Dict, frame_shape: Tuple[int, int]) -> Dict:
        """Update tracks with new detections"""
        new_centroids = []
        new_classes = []

        for box, class_name in zip(detections["boxes"], detections["class_names"]):
            x1, y1, x2, y2 = box
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            new_centroids.append((centroid_x, centroid_y))
            new_classes.append(class_name)

        if len(self.tracks) == 0:
            for centroid, class_name in zip(new_centroids, new_classes):
                self._add_track(centroid, class_name)
        else:
            self._match_detections(new_centroids, new_classes)

        self._remove_disappeared_tracks()
        self._update_statistics(new_classes)

        return self.get_tracked_objects()

    def _add_track(self, centroid: Tuple[int, int], class_name: str) -> None:
        self.tracks[self.next_id] = {
            "centroid": centroid,
            "class": class_name,
            "disappeared": 0,
            "added_frame": self.total_detections
        }
        self.track_history[self.next_id].append(centroid)
        self.next_id += 1

    def _match_detections(self, centroids: List, classes: List) -> None:
        matched_detection_indices = set()
        matched_track_ids = set()

        for i, (centroid, class_name) in enumerate(zip(centroids, classes)):
            closest_track_id = None
            closest_distance = self.max_distance

            for track_id, track_data in self.tracks.items():
                if track_id in matched_track_ids:
                    continue

                distance = self._euclidean_distance(centroid, track_data["centroid"])

                if distance < closest_distance:
                    closest_distance = distance
                    closest_track_id = track_id

            if closest_track_id is not None:
                self.tracks[closest_track_id]["centroid"] = centroid
                self.tracks[closest_track_id]["class"] = class_name
                self.tracks[closest_track_id]["disappeared"] = 0
                self.track_history[closest_track_id].append(centroid)
                matched_track_ids.add(closest_track_id)
                matched_detection_indices.add(i)

        for track_id in self.tracks:
            if track_id not in matched_track_ids:
                self.tracks[track_id]["disappeared"] += 1

        for i, (centroid, class_name) in enumerate(zip(centroids, classes)):
            if i not in matched_detection_indices:
                self._add_track(centroid, class_name)

    def _remove_disappeared_tracks(self) -> None:
        track_ids_to_remove = [
            track_id for track_id, track_data in self.tracks.items()
            if track_data["disappeared"] > self.max_disappeared
        ]

        for track_id in track_ids_to_remove:
            del self.tracks[track_id]

    def _update_statistics(self, classes: List) -> None:
        self.total_detections += 1
        for class_name in classes:
            self.class_counts[class_name] += 1

    @staticmethod
    def _euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def get_tracked_objects(self) -> Dict:
        tracked_objects = {}
        for track_id, track_data in self.tracks.items():
            if track_data["disappeared"] == 0:
                tracked_objects[track_id] = track_data
        return tracked_objects

    def get_track_history(self, track_id: int) -> List[Tuple[int, int]]:
        return self.track_history.get(track_id, [])

    def get_statistics(self) -> Dict:
        return {
            "total_tracks": len(self.tracks),
            "active_tracks": len(self.get_tracked_objects()),
            "total_detections": self.total_detections,
            "class_counts": dict(self.class_counts),
            "unique_classes": len(self.class_counts)
        }

    def reset(self) -> None:
        self.tracks.clear()
        self.track_history.clear()
        self.next_id = 0
