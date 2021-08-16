import os
import cv2
import numpy as np
import json

import imutils
from collections import defaultdict


data_dir = "/mnt/sdb1/Data/Phamacity_datasets/train"
filename = "linhdam2_final"
image_dir = os.path.join(data_dir, filename)
ann_file = os.path.join(data_dir, f"{filename}.json")
print(ann_file)

imgid2path = dict()
imgid2anns = defaultdict(list)

with open(ann_file) as f:
	data_json = json.load(f)
	annotations = data_json["annotations"]
	for ann in annotations:
		image_id = ann["image_id"]
		imgid2anns[image_id].append(ann)
		imgid2path[image_id] = os.path.join(image_dir, f"{image_id}.jpg")

	for image_id in imgid2anns.keys():
		anns = imgid2anns[image_id]
		img = cv2.imread(imgid2path[image_id])
		if img is None:
			print(f"Error {image_id}")
			continue

		print(image_id)
		
		for ann in anns:
			bbox = list(map(int, ann["bbox"]))
			person_id = int(ann["person_id"])

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
		key = cv2.waitKey(1) & 0xff
		if key == ord('q'):
			break


cv2.destroyAllWindows()