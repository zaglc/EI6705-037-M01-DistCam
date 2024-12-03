import cv2
import datetime
import numpy as np
import random
from shapely.geometry import Polygon, Point
from typing import Callable, Dict, List, Tuple

from detector.base.detector import YOLODetector, DetectionResult, create_detection_result


class ObjectCounter(YOLODetector):
    def __init__(self, 
            classes_of_interest: List[str] = ["person"],
            log_file: str = "detection_log.txt",
            *args, 
            **kwargs
        ):
        """
        Extends YOLODetector to count objects of interest.
        :param args: Arguments for YOLODetector.
        :param kwargs: Keyword arguments for YOLODetector.
        """
        super().__init__(*args, **kwargs)
        self.classes_of_interest: List[str] = classes_of_interest  # Default classes of interest
        self.cumulative_counts: List[Dict[str, int]] = []  # Cumulative count until each frame
        self.callback: Callable[[Dict[str, int]], None] = None  # Optional callback
        self.restricted_areas: List[Polygon] = []
        self.log_file: str = log_file

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
    
    def alert_callback(frame_index: int):
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
            if any(area.contains(center) for area in self.restricted_areas):
                entered_ids.append(track_id)
                self.log_event(
                    detection_result.frame_index,
                    "Entry",
                    f"Detected '{name}' (ID: {track_id}) entering restricted area at position ({x:.2f}, {y:.2f})"
                )

        # Trigger the entry callback if any object entered the restricted area
        if entered_ids:
            self.alert_callback(detection_result.frame_index)

    def reset_cumulative_counts(self) -> None:
        """
        Resets the cumulative counts to start fresh.
        """
        self.cumulative_counts = [{cls_: 0} for cls_ in self.classes_of_interest]

    def count_objects_in_frame(self, detection_result: DetectionResult) -> None:
        """
        Updates the cumulative count of classes of interest for a single frame.
        :param detection_result: DetectionResult object for the frame.
        """
        current_count = self.cumulative_counts[-1].copy() if self.cumulative_counts else {k: 0 for k in self.classes_of_interest}
        for box, name in zip(detection_result.boxes, detection_result.names):
            if name in self.classes_of_interest:
                current_count[name] += 1
                x, y, w, h = box
                self.log_event(
                    detection_result.frame_index,
                    "Detection",
                    f"Detected '{name}' at position ({x:.2f}, {y:.2f})"
                )
        self.cumulative_counts.append(current_count)
        self.check_restricted_area(detection_result)


    def online_predict(self, frame: np.ndarray) -> np.ndarray:
        """
        Performs online object detection and updates the count in real-time for each frame.
        :param video_path: Path to the video file.
        :return: A frame with bounding boxes drawn around detected objects.
        """
        detection_result = create_detection_result(self.frame_index, self._predict_one_frame(frame))
        self.count_objects_in_frame(detection_result)  # Update cumulative count
        self.frame_index += 1
        for name, conf, *xywh in zip(detection_result.names, detection_result.conf, detection_result.boxes):
            tl = round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1  # line/font thickness
            c1 = tuple(map(int, (xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2)))
            c2 = tuple(map(int, (xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2)))
            color = color or [random.randint(0, 255) for _ in range(3)]
            cv2.rectangle(frame, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            if name and conf > 0.7:
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(name, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(frame, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(frame, name, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        return frame


    def get_count_in_range(self, start_frame: int, end_frame: int) -> Dict[str, int]:
        """
        Retrieves the count of objects of interest in a specified frame range.
        :param start_frame: The starting frame index (inclusive).
        :param end_frame: The ending frame index (inclusive).
        :return: A dictionary with the count of objects of interest in the specified range.
        """
        if start_frame > end_frame:
            raise ValueError("Start frame index must be less than or equal to end frame index.")

        if start_frame > len(self.cumulative_counts) or end_frame > len(self.cumulative_counts):
            raise IndexError("Frame index out of bounds. Make sure to run online_predict first.")

        start_count = self.cumulative_counts[start_frame - 1] if start_frame > 0 else {k: 0 for k in self.classes_of_interest}
        end_count = self.cumulative_counts[end_frame]
        return {k: end_count[k] - start_count.get(k, 0) for k in self.classes_of_interest}

    def truncate_after_frame(self, frame_index: int) -> None:
        """
        Truncates detection results and cumulative counts after a given frame index.
        :param frame_index: The frame index after which all results will be removed.
        """
        if frame_index < 0 or frame_index >= len(self.cumulative_counts):
            raise IndexError("Frame index out of bounds.")

        # Truncate cumulative counts and detection results
        self.cumulative_counts = self.cumulative_counts[frame_index:]
        print(f"Truncated after frame {frame_index}. Remaining frames: {len(self.cumulative_counts)}.")
