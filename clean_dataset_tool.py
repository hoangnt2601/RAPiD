import os
import cv2
import numpy as np
import json
import imutils
import shutil


data_dir = "/mnt/sdb1/Data/Phamacity"
filename = "timecity1"
image_dir = os.path.join(data_dir, filename+"_clean")
clean_image_dir = os.path.join(data_dir, filename+"_clean")
if not os.path.exists(clean_image_dir):
    os.mkdir(clean_image_dir)
ann_file = os.path.join(data_dir, "annotations", f"{filename}_clean_new.json")
ann_file_clean = os.path.join(data_dir, "annotations", f"{filename}_clean2.json")

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

with open(ann_file) as f:
    anns = json.load(f)
    anns = anns["annotations"]
    for ann in anns:
        image_id = ann["image_id"]
        bbox = list(map(int, ann["bbox"]))
        person_id = int(ann["person_id"])

        img_path = os.path.join(image_dir, f"{image_id}.jpg")
        img_clean_path = os.path.join(clean_image_dir, f"{image_id}.jpg")
        img = cv2.imread(img_path)
        if img is None: continue

        x,y,w,h,a = bbox
        radian = a*np.pi/180
        C, S = np.cos(radian), np.sin(radian)
        R = np.asarray([[-C, -S], [S, -C]])
        pts = np.asarray([[-w / 2, -h / 2], [w / 2, -h / 2], 
                            [w / 2, h / 2], [-w / 2, h / 2]])
        points = np.asarray([((x, y) + pt @ R).astype(int) for pt in pts])

        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), 10)
        cv2.polylines(img, [points], True, (0, 0, 255), 5)
        cv2.putText(img, str(person_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                    (0,255,0), 4, cv2.LINE_AA)


        cv2.imshow("image", imutils.resize(img, width=1024))
        key = cv2.waitKey(0) & 0xff
        if key == ord('c'):
            print(f"ignore {person_id}")
            continue
        if key == ord('q'):
            json.dump(ann_json, f, indent=4)
            break

        with open(ann_file_clean, 'w') as f:
            obj_ann = {
                "area": w*h,
                "bbox": [x,y,w,h,a],
                "category_id": 1,
                "image_id": image_id,
                "iscrowd": 0,
                "segmentation": [],
                "person_id": person_id,
            }

            img_info = {
                "file_name": f"{image_id}.jpg",
                "id": image_id,
                "width": img.shape[0],
                "height": img.shape[1]
            }

            ann_json["images"].append(img_info)
            ann_json["annotations"].append(obj_ann)

            json.dump(ann_json, f, indent=4)

            if not os.path.exists(img_clean_path):
                shutil.copyfile(img_path, img_clean_path)

cv2.destroyAllWindows()

