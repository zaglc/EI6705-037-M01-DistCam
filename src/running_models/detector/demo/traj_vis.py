import os, sys
sys.path.append(os.getcwd())
from collections import defaultdict
import cv2
import numpy as np
from typing import List
from src.running_models.detector.base.detector import DetectionResult, YOLODetector, create_detection_result
import datetime

if __name__ == "__main__":
    # Initialize the YOLODetector
    detector = YOLODetector(
        device="cuda:0",
        weights="yolo11n.pt",
    )
    video_path = "data\src\Jackson-Hole-WY3@06-27_07-05-02.mp4"
    output_video_path = 'traj_demo.mp4'
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧高
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 编码器
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    track_history = defaultdict(list)
    frame_count = 0

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        result = detector._predict_one_frame(frame)
        # annotated_frame = result.plot()
        detection_result = create_detection_result(frame_count, result)
        for track_id, name, conf, xywh in zip(detection_result.track_ids, detection_result.names, detection_result.conf, detection_result.boxes):
            if name and conf > 0.5:
                tl = round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1  # line/font thickness
                c1 = tuple(map(int, (xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2)))
                c2 = tuple(map(int, (xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2)))
                color = [255, 0, 0]
                
                cv2.rectangle(frame, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(name, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(frame, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    frame, name, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA
                )
                x, y, w, h = xywh
                track_history[track_id].append((float(x), float(y)))
                if len(track_history[track_id]) > 300:
                    track_history[track_id] = track_history[track_id][-300:]
                track = track_history[track_id]
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors[track_id % len(colors)], thickness=2)  # Use a more visible color and thinner line


        cv2.imshow("YOLO Tracking Frame", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("视频已保存到:", output_video_path)