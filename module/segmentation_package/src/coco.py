import os
import cv2
from torchvision.datasets import CocoDetection
from module.segmentation_package.src.copy_paste import copy_paste_class
import numpy as np
min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True

    return False


@copy_paste_class
class CocoDetectionCP(CocoDetection):
    def __init__(
        self,
        root,
        annFile,
        transforms
    ):
        super(CocoDetectionCP, self).__init__(
            root, annFile, None, None, transforms
        )

        # filter images without detection annotations
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)
        self.ids = ids

    def load_example(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        masks = []
        bboxes = []
        for ix, obj in enumerate(target):
            for z in range(int(len(obj['segmentation'][0]) / 2)):
                if  obj['segmentation'][0][z * 2] < 0:
                    obj['segmentation'][0][z * 2] = 0
                if  obj['segmentation'][0][z * 2 + 1] < 0:
                    obj['segmentation'][0][z * 2 + 1] = 0
    
                if  obj['segmentation'][0][z * 2]  > w:
                    obj['segmentation'][0][z * 2]  = w
                if  obj['segmentation'][0][z * 2 + 1] > h:
                    obj['segmentation'][0][z * 2 + 1] = h
                                        
            obj['segmentation'] = [[sd if sd >=0 else 0 for sd in obj['segmentation'][0]]]
            px = []
            py = []
            for z in range(int(len(obj['segmentation'][0]) / 2)):
                px.append(obj['segmentation'][0][z * 2])
                py.append(obj['segmentation'][0][z * 2 + 1])
            bbox_tmp = [np.min(px).tolist(), np.min(py).tolist(), np.max(px).tolist(), np.max(py).tolist()]
            
            obj['bbox'] = [
                bbox_tmp[0],
                bbox_tmp[1],
                bbox_tmp[2] - bbox_tmp[0],
                bbox_tmp[3] - bbox_tmp[1]
            ]
            
            masks.append(self.coco.annToMask(obj))
            b_box = obj['bbox']
            bboxes.append(b_box + [obj['category_id']] + [ix])

        output = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes
        }
        return self.transforms(**output)
