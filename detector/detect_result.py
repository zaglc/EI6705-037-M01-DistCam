from typing import List, Union

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    """
    Pydantic model to store detection results for a single video frame.
    """


    frame_index: int = Field(..., description="The index of the current frame.")
    names: List[str] = Field(..., description="List of detected class names.")
    cls: List[float] = Field(..., description="List of detected class indices.")
    conf: List[float] = Field(..., description="List of confidence scores for detections.")
    boxes: List[List[float]] = Field(..., description="List of bounding boxes in xywh format.")
    track_ids: List[int] = Field(..., description="List of unique tracking IDs for detected objects.")

    class Config:
        schema_extra = {
            "example": {
                "frame_index": 1,
                "names": ["person", "car"],
                "cls": [0.0, 2.0],
                "conf": [0.95, 0.89],
                "boxes": [[100.0, 200.0, 50.0, 60.0], [300.0, 400.0, 80.0, 120.0]],
                "track_ids": [101, 202],
            }
        }


# Helper function to create a DetectionResult instance
def create_detection_result(frame_index: int, results) -> DetectionResult:
    """
    Creates a DetectionResult object from the detection model's output.


    :param frame_index: The index of the current frame.
    :param results: The detection model's output.
    :return: A DetectionResult instance.
    """
    names = results[0].names
    cls = results[0].boxes.cls.cpu().tolist()
    conf = results[0].boxes.conf.cpu().tolist()
    boxes = results[0].boxes.xywh.cpu().tolist()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    return DetectionResult(frame_index=frame_index, names=names, cls=cls, conf=conf, boxes=boxes, track_ids=track_ids)
