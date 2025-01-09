

import os, sys
sys.path.append(os.getcwd())
from collections import defaultdict
import cv2
import datetime
import numpy as np
from typing import List
from running_models.detector.base.detector import DetectionResult, YOLODetector, create_detection_result
from running_models.detector.advanced.counter import ObjectCounter
from shapely.geometry import Point, Polygon
import cv2
import numpy as np
from shapely.geometry import Polygon


if __name__ == "__main__":
    # Initialize the YOLODetector
    # counter = ObjectCounter(
    #     classes_of_interest=["car", "truck", "bus"],
    # )
    # polygon_coords = [(800, 500), (800, 1000), (1200, 1000), (1200, 500)]
    counter = ObjectCounter(
        device="cuda:0",
        classes_of_interest=["car"],
        weights="yolo11n.pt",
        log_file=f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt",
    )
    polygon_coords = [(1000, 100), (1300, 100), (1300, 500), (1000, 500)]
    polygon = Polygon(polygon_coords)
    # counter.add_restricted_area(polygon_coords)
    # 获取多边形的外部坐标并转换为OpenCV需要的格式
    exterior_coords = np.array(list(polygon.exterior.coords), dtype=np.int32)
    video_path = "data\src\Jackson-Hole-WY3@06-27_07-05-02.mp4"
    output_video_path = 'counting_demo.mp4'
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧高
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 编码器
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        cls, conf, boxes, cumu_cnt = counter.online_predict(frame)
        mask = np.zeros_like(frame)

        # 设置多边形的颜色和透明度
        color = (0, 255, 0)  # 绿色
        alpha = 0.5          # 半透明度，范围在0到1之间

        # 绘制多边形并填充颜色
        cv2.fillPoly(mask, [exterior_coords], color)

        # 将掩膜与原始图像融合
        # annotated_frame = cv2.addWeighted(annotated_frame, 1, mask, alpha, 0)
        print("Frame Index", counter.frame_index, "Counting results",  counter.cumulative_counts[-1])
        # cv2.imshow("YOLO Tracking Frame", annotated_frame)
        out.write(annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()