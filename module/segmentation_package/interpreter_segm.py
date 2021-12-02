import numpy as np
import cv2
import time
import logging

from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg


class SegmentationInference:
    def __init__(self, model_path, device='cpu', config_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
        start_time = time.time()
        self.model_path = model_path
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(config_path))
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.WEIGHTS = self.model_path
        self.model = DefaultPredictor(self.cfg)
        self.approximate_value = 0.0015
        logging.info(
            "Initialization SegmentationInference was finished in {} [s]".format(round(time.time() - start_time, 2)))

    def __approximate(self, output):
        polygons = []
        logging.info("Aproxiame: {}".format(self.approximate_value))
        for z in range(len(output['instances'])):
            masks = np.array(output['instances'].get('pred_masks')[z].to('cpu'))
            imgUMat = np.array(masks * 255, dtype=np.uint8)
            cnts, hierarchy = cv2.findContours(imgUMat, 1, 2)
            cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
            cnts_s = cnts[len(cnts) - 1]
            epsilon = self.approximate_value * cv2.arcLength(cnts_s, True)
            approx = cv2.approxPolyDP(cnts_s, epsilon, True)
            polygons_dict = {}
            for i in range(len(approx)):
                polygons_dict.update({
                    "x{}".format(i + 1): int(approx[i][0][0]),
                    "y{}".format(i + 1): int(approx[i][0][1])
                })
            polygons.append(polygons_dict)
        return polygons

    def __get_mask(self, image, pts):
        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        croped = image[y:y + h, x:x + w].copy()

        ## (2) make mask
        pts = pts - pts.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)
        return dst

    def __convert_to_polygon(self, json):
        polygon = []

        for i in range(int(len(json) / 2)):
            polygon.append([json["x{}".format(i + 1)], json["y{}".format(i + 1)]])
        return polygon

    def simple_inference(self, img):
        start_time = time.time()
        outputs = self.model(img)
        array = self.__approximate(outputs)
        masks = []
        for single in array:
            polygon = np.array(self.__convert_to_polygon(single))
            masks.append(self.__get_mask(img, polygon))
        logging.info("Inference time by Mask RCNN models has taken {} [s]".format(round(time.time() - start_time, 2)))
        return array, masks

    def re_init_model(self, threshold):
        try:
            new_threshold = max(0.01, min(0.9, threshold))
            if new_threshold != self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = new_threshold
                self.model = DefaultPredictor(self.cfg)
                logging.info(
                    "Was chaged new defoult parameter: {}".format(self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST))
        except Exception as e:
            logging.warning("Error re_init_model: {}".format(e))