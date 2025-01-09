import datetime
import random
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from running_models.detector.base.detector import (
    DetectionResult,
    YOLODetector,
    create_detection_result,
)


class ObjectCounter(YOLODetector):
    def __init__(
        self, classes_of_interest: List[str] = ["person"], log_file: str = "detection_log.txt", *args, **kwargs
    ):
        """
        Extends YOLODetector to count objects of interest.
        :param args: Arguments for YOLODetector.
        :param kwargs: Keyword arguments for YOLODetector.
        """
        super().__init__(*args, **kwargs)
        self.classes_of_interest: List[str] = classes_of_interest  # Default classes of interest
        self.cumulative_counts: Dict[str, int] = {}  # Cumulative count until each frame
        self.callback: Callable[[Dict[str, int]], None] = None  # Optional callback
        self.restricted_areas: List[Polygon] = []
        self.log_file: str = log_file
        self.counted_track_id = set()
        self.alert_history = set()
        self.reset_cumulative_counts()

    def _frame_to_timestamp(self, frame_index: int) -> str:
        """
        Converts a frame index to a timestamp.
        :param frame_index: Frame index.
        :return: Timestamp in "YYYY-MM-DD HH:MM:SS" format.
        """
        base_time = datetime.datetime(2024, 1, 1, 0, 0, 0)
        seconds = frame_index / 24  # 24 FPS
        timestamp = base_time + datetime.timedelta(seconds=seconds)
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    def log_event(self, frame_index: int, event_type: str, details: str) -> None:
        """
        Logs an event to the log file.
        :param frame_index: Frame index where the event occurred.
        :param event_type: Type of event (e.g., "Detection", "Entry").
        :param details: Details of the event.
        """
        timestamp = self._frame_to_timestamp(frame_index)
        log_message = f"{timestamp}, {event_type}: {details}\n"
        with open(self.log_file, "a") as log_file:
            log_file.write(log_message)

    def alert_callback(self, frame_index: int):
        print(f"ALERT: Frame {frame_index}, something entered the restricted area.")

    def add_restricted_area(self, vertices: List[Tuple[float, float]]) -> None:
        """
        Adds a restricted area defined by a list of vertices.
        :param vertices: List of (x, y) tuples defining the polygon vertices.
        """
        self.restricted_areas.append(Polygon(vertices))

    def check_restricted_area(self, detection_result: DetectionResult) -> None:
        """
        Checks if any detected object is inside the restricted areas.
        :param detection_result: DetectionResult object for the frame.
        """
        if not self.restricted_areas:
            return

        entered_ids = []  # Store IDs of objects inside restricted areas
        for box, name, track_id in zip(detection_result.boxes, detection_result.names, detection_result.track_ids):
            # Convert bounding box center to a Point
            x, y, w, h = box
            center = Point(x, y)

            # Check if the center is within any restricted area
            if any(area.contains(center) for area in self.restricted_areas) and track_id not in self.alert_history:
                self.alert_history.add(track_id)
                entered_ids.append(track_id)
                self.log_event(
                    detection_result.frame_index,
                    "ALERT",
                    f"'{name}' (ID: {track_id}) entering restricted area at position ({x:.2f}, {y:.2f})",
                )

        # Trigger the entry callback if any object entered the restricted area
        if entered_ids:
            self.alert_callback(detection_result.frame_index)

    def reset_cumulative_counts(self, new_classes_of_interest: List[str] = None) -> None:
        """
        Resets the cumulative counts to start fresh.
        """

        if new_classes_of_interest is not None:
            self.classes_of_interest = new_classes_of_interest
        self.frame_index = 0
        self.cumulative_counts = {cls_: 0 for cls_ in self.classes_of_interest}

    def count_objects_in_frame(self, detection_result: DetectionResult) -> None:
        """
        Updates the cumulative count of classes of interest for a single frame.
        :param detection_result: DetectionResult object for the frame.
        """

        for box, name, track_id in zip(detection_result.boxes, detection_result.names, detection_result.track_ids):
            if track_id not in self.counted_track_id:
                self.counted_track_id.add(track_id)
                if name in self.classes_of_interest:
                    self.cumulative_counts[name] += 1  
        self.log_event(
            detection_result.frame_index,
            "INFO",
            f"counting results by now: {self.cumulative_counts}",
        )
        self.check_restricted_area(detection_result)

    def online_predict(self, frame: np.ndarray, conf_thre = 0.4, iou_thre = 0.5, imgsz = 640) -> tuple:
        """
        Performs online object detection and updates the count in real-time for each frame.
        :param video_path: Path to the video file.
        :return: A frame with bounding boxes drawn around detected objects.
        """

        detection_result = create_detection_result(self.frame_index, self._predict_one_frame(frame, conf_thre, iou_thre, imgsz))
        self.count_objects_in_frame(detection_result)  # Update cumulative count
        self.frame_index += 1

        return (detection_result.cls, detection_result.conf, detection_result.boxes, self.cumulative_counts, detection_result.track_ids)

    def get_count_in_range(self, start_frame: int, end_frame: int) -> Dict[str, int]:
        """
        Retrieves the count of objects of interest in a specified frame range.
        :param start_frame: The starting frame index (inclusive).
        :param end_frame: The ending frame index (inclusive).
        :return: A dictionary with the count of objects of interest in the specified range.
        """

        raise NotImplementedError("This method is not support yet.")
        if start_frame > end_frame:
            raise ValueError("Start frame index must be less than or equal to end frame index.")

        if start_frame > len(self.cumulative_counts) or end_frame > len(self.cumulative_counts):
            raise IndexError("Frame index out of bounds. Make sure to run online_predict first.")

        start_count = (
            self.cumulative_counts[start_frame - 1] if start_frame > 0 else {k: 0 for k in self.classes_of_interest}
        )
        end_count = self.cumulative_counts[end_frame]
        return {k: end_count[k] - start_count.get(k, 0) for k in self.classes_of_interest}

    def truncate_after_frame(self, frame_index: int) -> None:
        """
        Truncates detection results and cumulative counts after a given frame index.
        :param frame_index: The frame index after which all results will be removed.
        """

        raise NotImplementedError("This method is not support yet.")
        if frame_index < 0 or frame_index >= len(self.cumulative_counts):
            raise IndexError("Frame index out of bounds.")

        # Truncate cumulative counts and detection results
        self.cumulative_counts = self.cumulative_counts[frame_index:]
        print(f"Truncated after frame {frame_index}. Remaining frames: {len(self.cumulative_counts)}.")