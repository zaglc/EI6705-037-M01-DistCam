import random
from typing import List

import numpy as np
import torch

from running_models.yolov3.models import Darknet
from running_models.yolov3_utils import (
    letterbox,
    load_classes,
    non_max_suppression,
    plot_one_box,
    scale_coords,
)

IMG_SIZE = 512

# TODO: so many todo...
def initialize_model(
    local_num_cam: int,
    type: str = "yolov3",
    weights: str = "./running_models/yolov3/weights/110-0.447.pt",
    # data_class: str = "./running_models/yolov3/data/coco.names",
    data_class: str = "./configs/names.txt",
    seedid: int = 1024,
) -> list:
    assert type in ["yolov3"]

    random.seed(seedid)
    classes = load_classes(data_class)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    if type == "yolov3":
        cfg = "./running_models/yolov3/cfg/yolov3-t.cfg"
        img_size = IMG_SIZE
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = Darknet(cfg, img_size)
        model.load_state_dict(torch.load(weights, map_location=device)["model"])
        model.to(device).eval()

        chunk_tsr = torch.zeros((local_num_cam, 3, int(img_size * 9 / 16), img_size)).float().to(device)
        model(chunk_tsr, augment=False)

    return [model, device, classes, colors]


def preprocess_img(ori_img: np.ndarray, img_size: int = IMG_SIZE, device: str = "cuda:0"):
    # print(ori_img.dtype)
    img = letterbox(ori_img.copy(), new_shape=(img_size, int(img_size * 9 / 16)))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0

    return img


def process_result(ori_img: np.ndarray, det, classes, colors):
    inf_shape = (int(IMG_SIZE * 9 / 16), IMG_SIZE)

    # process detections
    if det is not None and len(det):
        # if i == 0: print(f"DETECT: total {len(det)} boxes")
        det[:, :4] = scale_coords(inf_shape, det[:, :4], ori_img.shape).round()

        for *xyxy, conf, cls in reversed(det):
            label = "%s %.2f" % (classes[int(cls)], conf)
            plot_one_box(xyxy, ori_img, label=label, color=colors[int(cls)])

    return ori_img
