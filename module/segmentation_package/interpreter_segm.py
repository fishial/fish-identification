import numpy as np
import cv2
import time
import logging
from scipy.interpolate import splprep, splev
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg
from shapely.geometry import Polygon


def get_iou_poly(polygon1, polygon2):
    intersect = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersect / union
    return iou


def get_area_intersection(poly_a, poly_b, threshold=0.9):
    if poly_a.intersection(poly_b).area / poly_a.area > threshold:
        return True
    return False


class SegmentationInference:
    def __init__(self, model_path, device='cpu', config_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                 class_num=1, labels_list=['Fish']):
        start_time = time.time()
        self.labels_list = labels_list
        self.model_path = model_path
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(config_path))
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = class_num
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.WEIGHTS = self.model_path
        self.cfg.freeze()
        self.model = DefaultPredictor(self.cfg)
        self.approximate_value = 0.0012
        logging.info(
            "Initialization SegmentationInference was finished in {} [s]".format(round(time.time() - start_time, 2)))

    def __NMS(self, polygons, poly):
        for poly_id_a in range(len(polygons) - 1, -1, -1):
            iou = get_iou_poly(polygons[poly_id_a], poly)
            if iou > 0.5:
                return True
        return False

    def __approximate(self, output):
        meta_data = {
            'polygons': [],
            'scores': [],
            'areas': [],
            'labels_list': []
        }
        poly_instances = []

        logging.info("Aproxiame: {}".format(self.approximate_value))

        polygons_tmp = []

        for z in range(len(output['instances'])):
            masks = np.array(output['instances'].get('pred_masks')[z].to('cpu'))
            imgUMat = np.array(masks * 255, dtype=np.uint8)
            cnts, hierarchy = cv2.findContours(imgUMat, 1, 2)
            cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))

            if len(cnts) == 0:
                continue
            cnts_s = cnts[len(cnts) - 1]
            epsilon = self.approximate_value * cv2.arcLength(cnts_s, True)
            approx = cv2.approxPolyDP(cnts_s, epsilon, True)
            polygons_dict = {}
            if len(approx) < 10:
                continue
            try:
                poly_tmp = Polygon(
                    [(int(approx[point_id][0][0]), int(approx[point_id][0][1])) for point_id in range(len(approx))])
            except Exception as e:
                print("Error! ", e)

            x = []
            y = []

            for i in range(len(approx)):
                x.append(int(approx[i][0][0]))
                y.append(int(approx[i][0][1]))

            x.append(x[0])
            y.append(y[0])
            tck, _ = splprep([x, y], s=0, per=True)
            xx, yy = splev(np.linspace(0, 1, int(len(x) * 1.5)), tck, der=0)
            polygons_dict = {}
            for i in range(len(xx)):
                polygons_dict.update({
                    "x{}".format(i + 1): int(xx[i]),
                    "y{}".format(i + 1): int(yy[i])
                })

            meta_data['polygons'].append(polygons_dict)
            meta_data['scores'].append(float(np.array(output['instances'].get('scores')[z].to('cpu'))))
            meta_data['areas'].append(float(cv2.contourArea(cnts_s)))
            meta_data['labels_list'].append(self.labels_list[int(output['instances'][z].pred_classes[0])])
            poly_instances.append([poly_tmp, poly_tmp.area, len(meta_data['polygons']) - 1])

        poly_instances.sort(key=lambda x: x[1])
        index_to_remove = []
        for poly_id_a in range(len(poly_instances) - 1, -1, -1):
            for poly_id_b in range(len(poly_instances)):
                if poly_id_a == poly_id_b: continue
                try:
                    if get_area_intersection(poly_instances[poly_id_a][0], poly_instances[poly_id_b][0]):
                        index_to_remove.append(poly_instances[poly_id_a][2])
                        break
                except Exception as e:
                    index_to_remove.append(poly_instances[poly_id_a][2])
                    break
        index_to_remove.sort(reverse=True)
        for index_ in index_to_remove:
            for key_name in meta_data:
                del meta_data[key_name][index_]
        return meta_data

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

        meta_data = self.__approximate(outputs)
        masks = []

        for iddx, single in enumerate(meta_data['polygons']):
            polygon = np.array(self.__convert_to_polygon(single))
            masks.append(self.__get_mask(img, polygon))
        logging.info("Inference time by Mask RCNN models has taken {} [s]".format(round(time.time() - start_time, 2)))
        return meta_data, masks, outputs

    def re_init_model(self, threshold):
        try:
            new_threshold = max(0.01, min(0.999, threshold))
            if new_threshold != self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                self.cfg.defrost()
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = new_threshold
                self.cfg.freeze()
                self.model = DefaultPredictor(self.cfg)
                logging.info(
                    "Was chaged new defoult parameter: {}".format(self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST))
        except Exception as e:
            logging.warning("Error re_init_model: {}".format(e))