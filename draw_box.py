import os
import cv2
import numpy as np
import json
import sympy
import imutils
from collections import defaultdict

shape = None
scaleX = None
scaley = None
resize = (1024, 1024)

img_c = None
coords = []
draw_box = []

def get_centroid(vertexes):
	_x_list = [vertex [0] for vertex in vertexes]
	_y_list = [vertex [1] for vertex in vertexes]
	_len = len(vertexes)
	_x = sum(_x_list) / _len
	_y = sum(_y_list) / _len
	
	return int(_x), int(_y)


def draw_polygon(event, x, y, flags, param):
	global coords
	if event == cv2.EVENT_LBUTTONDOWN:
		font = cv2.FONT_HERSHEY_TRIPLEX
		coords.append((x, y))
		cv2.circle(img_c, (x, y), 2, (255, 0, 0), 10)
		if len(coords) == 4:
			pts = np.array(coords).astype(np.int32)
			cv2.polylines(img_c, [pts], 
					  True, (0, 255, 0), 5)

			x1 = int(coords[0][0] * scaleX)
			y1 = int(coords[0][1] * scaleY)

			x2 = int(coords[1][0] * scaleX)
			y2 = int(coords[1][1] * scaleY)

			x3 = int(coords[2][0] * scaleX)
			y3 = int(coords[2][1] * scaleY)
			
			x4 = int(coords[3][0] * scaleX)
			y4 = int(coords[3][1] * scaleY)

			# x1 = int(coords[0][0])
			# y1 = int(coords[0][1])

			# x2 = int(coords[1][0])
			# y2 = int(coords[1][1])

			# x3 = int(coords[2][0])
			# y3 = int(coords[2][1])
			
			# x4 = int(coords[3][0])
			# y4 = int(coords[3][1])


			person_vector = np.asarray([x4 - x1, y4 - y1])
			axis_vector = np.asarray([1, 0])
			cos_alpha = np.dot(person_vector, axis_vector) / (np.linalg.norm(person_vector) * np.linalg.norm(axis_vector))
			if cos_alpha < 0:
				cos_alpha = abs(cos_alpha)
			else:
				cos_alpha = -cos_alpha

			cx, cy = get_centroid([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
			angle = int(cos_alpha/(np.pi/180))

			W, H = sympy.symbols('W, H')
			eq1 = sympy.Eq(H*np.sin(cos_alpha) - W*np.cos(cos_alpha), 2*(x1-cx))
			eq2 = sympy.Eq(-H*np.cos(cos_alpha) - W*np.sin(cos_alpha), 2*(y1-cy))
			output = sympy.solve((eq1,eq2), (W, H))
			W, H = int(output[W]), int(output[H])

			# cv2.rectangle(img_c, (cx-W//2, cy-H//2), (cx+W//2, cy+H//2), color=(0,0,0), thickness=5)

			# radian = angle*np.pi/180
			# C, S = np.cos(radian), np.sin(radian)
			# R = np.asarray([[-C, -S], [S, -C]])
			# pts = np.asarray([[-W / 2, -W / 2], [W / 2, -W / 2], 
			# 					[W / 2, W / 2], [-W / 2, W / 2]])
			# p = np.asarray([((cx, cy) + pt @ R).astype(int) for pt in pts])

			# cv2.circle(img_c, (cx, cy), 2, (0, 255, 0), 10)
			# cv2.polylines(img_c, [p], True, (255, 0, 0), 5)

			draw_box.append([cx, cy, W, H, angle])

			if len(coords) == 4:
				print(draw_box)
				coords = []
				
		# cv2.imwrite("img.jpg", img)
		# print("Saved")	

		cv2.imshow("image", imutils.resize(img_c, width=1024))
		key = cv2.waitKey(0)
		if key == ord("q"):
			cv2.destroyAllWindows()
			exit()


data_dir = "/mnt/sdb1/Data/Phamacity"
filename = "linhdam2_clean"
image_dir = os.path.join(data_dir, filename)
ann_file = os.path.join(data_dir, "annotations", f"{filename}.json")
ann_file_new = os.path.join(data_dir, "annotations", f"{filename}_new.json")

imgid2path = dict()
imgid2anns = defaultdict(list)

with open(ann_file) as f:
	data_json = json.load(f)
	annotations = data_json["annotations"]
	total = len(annotations)
	count = 0
	for ann in annotations:
		image_id = ann["image_id"]
		imgid2anns[image_id].append(ann)
		imgid2path[image_id] = os.path.join(image_dir, f"{image_id}.jpg")

	for image_id in imgid2anns.keys():
		anns = imgid2anns[image_id]
		img = cv2.imread(imgid2path[image_id])
		if img is None: continue
		shape = img.shape[:2]
		scaleX = shape[0] / resize[0]
		scaleY = shape[0] / resize[1]
		
		img_c = imutils.resize(img, width=1024)

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
		cv2.setMouseCallback('image', draw_polygon)
		key = cv2.waitKey(0) & 0xff
		if key == ord('q'):
			break
		if key == ord('c'):
			os.remove(imgid2path[image_id])
			continue
		
		if draw_box:
			with open(ann_file_new, 'w') as f:
				for bb in draw_box:
					x,y,w,h,a = bb
					if w < 0 or h < 0:
						continue
					ann = {
						"area": w*h,
						"bbox": [x,y,w,h,a],
						"category_id": 1,
						"image_id": image_id,
						"iscrowd": 0,
						"segmentation": [],
						"person_id": 0,
					}

					data_json["annotations"].append(ann)

					radian = a*np.pi/180
					C, S = np.cos(radian), np.sin(radian)
					R = np.asarray([[-C, -S], [S, -C]])
					pts = np.asarray([[-w / 2, -h / 2], [w / 2, -h / 2], 
										[w / 2, h / 2], [-w / 2, h / 2]])
					points = np.asarray([((x, y) + pt @ R).astype(int) for pt in pts])

					# cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), 10)
					cv2.polylines(img, [points], True, (0, 0, 255), 5)
					cv2.putText(img, str(person_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,
												(0,255,0), 4, cv2.LINE_AA)
				
				json.dump(data_json, f, indent=4)

			# cv2.imwrite("img.jpg", img)
			
			draw_box = []

		count += 1
		print(f"Processing...{count}/{total}")


cv2.destroyAllWindows()

