from typing import List, Union

import cv2
import numpy as np
from ultralytics import YOLO

from src.running_models.detector.base.detect_result import DetectionResult, create_detection_result


class YOLODetector:
    """
    A class to perform object detection and tracking using the YOLO model.
    """

    def __init__(self, device: str = "cuda:0", weights: str = "yolo11n.pt"):
        """
        Initializes the YOLODetector with the specified device and model weights.

        :param device: The device to run the model on (e.g., "cuda:0" or "cpu").
        :param weights: Path to the YOLO model weights file.
        """
        self.device = device
        self.weights = weights
        self.model = YOLO(self.weights).to(self.device)
        self.frame_index = 0

    def _predict_one_frame(self, frame: np.ndarray, conf_thre: float, iou_thre: float, imgsz: int) -> List[DetectionResult]:
        """
        Performs object detection and tracking on a single frame.

        :param frame: A numpy array representing the input frame (shape: [3, 1920, 1080]).
        :return: Detection results for the input frame.
        :raises TypeError: If the input is not a numpy array.
        :raises ValueError: If the input frame does not have the required shape.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("Input frame must be a numpy array.")

        results = self.model.track(frame, persist=True, conf=conf_thre, iou=iou_thre, imgsz=imgsz, verbose=False)
        return results[0]

    def offline_predict(self, video_path: str) -> List[DetectionResult]:
        """
        Performs offline object detection on a video file.

        :param video_path: Path to the video file.
        :return: A list of DetectionResult objects, one for each frame in the video.
        """
        detection_results = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = self._predict_one_frame(frame)
            detection_result = create_detection_result(self.frame_index, results)
            self.frame_index += 1
            detection_results.append(detection_result)

        cap.release()
        return detection_results