import json
import os

import cv2
import imutils
import numpy as np
import torch
import torchvision.transforms.functional as tvf
from PIL import Image

from models.rapid import RAPiD
from tracker.deep_sort import DeepSort
from utils import utils

weights_path = "weights/pL1_HBCP608_Apr14_6000.ckpt"
model = RAPiD(backbone='dark53')
model.load_state_dict(torch.load(weights_path)['model_state_dict'])
print(f'Successfully loaded weights: {weights_path}')
model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    device = "cuda"

image_dir = "datasets"
file_name = "75-279_LinhDam_HN-1"
ds = DeepSort("weights/osnet_x1_0.onnx")
vs = cv2.VideoCapture(f"/mnt/sdb1/Data/record/{file_name}.mkv")
input_size = 608
conf_thres = 0.3
nms_thres = 0.45
top_k = 100
DEBUG = False
frame_num = 0
interval = 10

ann_json = {
    "annotations": [],
    "images": [],
    "categories": [
    {
        "id": 1,
        "name": "person",
        "supercategory": "person"
    }]
}

while True:
    image = vs.read()[1]
    if image is None:
        continue
    
    image = image.copy()

    pil_img = Image.fromarray(image)
    # pad to square
    input_img, _, pad_info = utils.rect_to_square(pil_img, None, input_size, 0)
    input_ori = tvf.to_tensor(input_img)
    input_ = input_ori.unsqueeze(0)
    input_ = input_.to(device=device)

    with torch.no_grad():
        dts = model(input_).cpu()

    dts = dts.squeeze()
    # post-processing
    dts = dts[dts[:,5] >= conf_thres]
    if len(dts) > top_k:
        _, idx = torch.topk(dts[:,5], k=top_k)
        dts = dts[idx, :]

    dts = utils.nms(dts, is_degree=True, nms_thres=nms_thres, img_size=input_size)
    dts = utils.detection2original(dts, pad_info.squeeze())

    boxes = []
    infos= []
    for bb in dts:
        x,y,w,h,a,conf = bb
        radian = a*np.pi/180
        C, S = np.cos(radian), np.sin(radian)
        R = np.asarray([[-C, -S], [S, -C]])
        pts = np.asarray([[-w / 2, -h / 2], [w / 2, -h / 2], 
                            [w / 2, h / 2], [-w / 2, h / 2]])
        points = np.asarray([((x, y) + pt @ R).astype(int) for pt in pts])

        # cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), 10)
        # cv2.polylines(image, [points], True, (0, 0, 255), 5)
        boxes.append(points)
        infos.append([x,y,w,h,a])

    tracked_boxes = ds.update(boxes, infos, image)

    if frame_num % interval == 0:
        with open(f'{file_name}.json', 'w') as f:
            img_id = f"{file_name}_{frame_num}"
            cv2.imwrite(os.path.join(image_dir, f"{img_id}.jpg"), image)

            for trk in tracked_boxes:
                x,y,w,h,a,trk_id = trk

                obj_ann = {
                    "area": w*h,
                    "bbox": [x,y,w,h,a],
                    "category_id": 1,
                    "image_id": img_id,
                    "iscrowd": 0,
                    "segmentation": [],
                    "person_id": trk_id,
                }

                img_info = {
                    "file_name": f"{img_id}.jpg",
                    "id": img_id,
                    "width": image.shape[0],
                    "height": image.shape[1]
                }

                ann_json["images"].append(img_info)
                ann_json["annotations"].append(obj_ann)

                if DEBUG:
                    a = a*np.pi/180
                    C, S = np.cos(a), np.sin(a)
                    R = np.asarray([[-C, -S], [S, -C]])
                    pts = np.asarray([[-w / 2, -h / 2], [w / 2, -h / 2], 
                                        [w / 2, h / 2], [-w / 2, h / 2]])
                    points = np.asarray([((x, y) + pt @ R).astype(int) for pt in pts])

                    cv2.putText(image, str(trk_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                    (0,255,0), 4, cv2.LINE_AA)
                    cv2.polylines(image, [points], True, (0, 0, 255), 5)

            json.dump(ann_json, f, indent=4)

    frame_num+=1

    if DEBUG:
        cv2.imshow("image", imutils.resize(image, width=720))
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

cv2.destroyAllWindows()
