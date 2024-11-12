import torch
import numpy as np
import random
from typing import List

from running_models.yolov3.models import Darknet
from running_models.yolov3_utils import (
    load_classes, letterbox, non_max_suppression,
    scale_coords, plot_one_box,
)


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
        img_size = 512
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = Darknet(cfg, img_size)
        model.load_state_dict(torch.load(weights, map_location=device)["model"])
        model.to(device).eval()

        chunk_tsr = torch.zeros((local_num_cam, 3, int(img_size*9/16), img_size)).float().to(device)
        model(chunk_tsr, augment=False)

    return [model, img_size, device, classes, colors, chunk_tsr]


def preprocess_img(chunk_tsr: torch.Tensor, ori_imgs: List[np.ndarray], img_size: int, device: str, active_id: int):
    # print(ori_img.dtype)
    imgs = [letterbox(img, new_shape=(img_size, int(img_size*9/16))) for img in ori_imgs]
    imgs = np.stack(imgs, 0)
    imgs = imgs[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
    imgs = np.ascontiguousarray(imgs)

    imgs = torch.from_numpy(imgs).to(device)
    imgs = imgs.float()/255.0

    chunk_tsr.data[:active_id] = imgs.data


def process_result(ori_imgs: List[np.ndarray], img_size: int, results, classes, colors):
    # apply NMS
    results = non_max_suppression(results, conf_thres=0.3, multi_label=False)
    inf_shape = (int(img_size*9/16), img_size)

    # process detections
    for i, det in enumerate(results):
        
        if det is not None and len(det):
            # if i == 0: print(f"DETECT: total {len(det)} boxes")
            det[:, :4] = scale_coords(inf_shape, det[:, :4], ori_imgs[i].shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box(xyxy, ori_imgs[i], label=label, color=colors[int(cls)])