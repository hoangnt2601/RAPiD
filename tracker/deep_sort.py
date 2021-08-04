import numpy as np
import torch
import cv2

from .onnx_extractor import Extractor
# from .pytorch_extractor import Extractor
from .nn_matching import NearestNeighborDistanceMetric
from .detection import Detection
from .tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, boxes, infos, ori_img):
        bbox_tlwh = []
        for p in boxes:
            xmin = np.array([p[0][0], p[1][0], p[2][0], p[3][0]]).min()
            ymin = np.array([p[0][1], p[1][1], p[2][1], p[3][1]]).min()
            xmax = np.array([p[0][0], p[1][0], p[2][0], p[3][0]]).max()
            ymax = np.array([p[0][1], p[1][1], p[2][1], p[3][1]]).max()
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            w = xmax - xmin
            h = ymax - ymin
            bbox_tlwh.append([xmin, ymin, w, h])
        bbox_tlwh = np.asarray(bbox_tlwh)
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_tlwh, ori_img)
        detections = [Detection(bbox_tlwh[i], features[i], infos[i]) for i in range(bbox_tlwh.shape[0])]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # box = track.to_tlwh()
            # x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            # x, y, w, h = box
            x,y,w,h,a = track.info
            track_id = track.track_id
            outputs.append([x,y,w,h,a,track_id])
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs


    """
        TODO:
            Convert bbox from xc_yc_w_h to xtl_ytl_w_h
        Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    def _get_features(self, bbox_tlwh, ori_img):
        im_crops = []
        for bb in bbox_tlwh:
            x,y,w,h = bb
            croped = ori_img[y:y+h, x:x+w].copy()
            im_crops.append(croped)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


