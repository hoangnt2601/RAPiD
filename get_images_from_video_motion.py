import numpy as np
import imutils
import cv2
from imutils.video import FileVideoStream, VideoStream
import os
from pathlib import Path

class SingleMotionDetector:
    def __init__(self, accum_weight=0.5):
        self.accum_weight = accum_weight
        self.bg = None

    def update(self, image):
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        cv2.accumulateWeighted(image, self.bg, self.accum_weight)

    def detect(self, image, thresh_val=25, min_area=150):
        motion = True
        delta = cv2.absdiff(self.bg.astype("uint8"), image)
        
        thresh = cv2.threshold(delta, thresh_val, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=5)
        thresh = cv2.dilate(thresh, None, iterations=5)
        cv2.imshow("thresh", imutils.resize(thresh, width=720))
        cv2.imshow("image", imutils.resize(image, width=720))
        cv2.waitKey(1)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            if cv2.contourArea(c) > min_area:
                return motion

        motion = False
        return motion


# sub_rect = [0.0, 0.45, 0.7, 1.0]
# video_path = "/mnt/sdb1/datasets/fish-eye/511/greet/176.mp4"
video_path = "rtsp://root:Hnpmc511@192.168.40.130:564/live1s1.sdp"
output_dir = "/mnt/sdb1/datasets/fish-eye/lyquocsu/lyquocsu_quay"
data_name = "lyquocsu_quay"
frame_count = 25
total = 0

vs = VideoStream(video_path).start()
smd = SingleMotionDetector()

while True:
    frame = vs.read()
    h, w = frame.shape[:2]
    # x1 = int(w * sub_rect[0])
    # y1 = int(h * sub_rect[1])
    # x2 = int(w * sub_rect[2])
    # y2 = int(h * sub_rect[3])
    # frame = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    try:
        motion = smd.detect(gray)
    except Exception as e:
        print(e)
    
    smd.update(gray)
    total += 1
    if total < frame_count:
        continue
    
    if motion and total % 15 == 0:
        idx = len(list(Path(output_dir).glob("*.jpg")))
        img_name = f"{data_name}_{idx:06}"
        cv2.imwrite(f"{output_dir}/{img_name}.jpg", frame)
        
vs.stop()