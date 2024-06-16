import logging
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

from data_types.image_content_recogniser_type import IcrType
from src.profiler import Profiler
from src.settings import cfg


class ImageObjectDetectorPostrocessor:
    def postprocess_predictions(
        self,
        img: np.ndarray,
        bboxes: np.ndarray,
        classes: List[str],
        object_detection_type: str,
    ) -> Tuple[np.ndarray, List[str]]:
        cls_with_conf = []
        for i, classI in enumerate(classes):
            cls_with_conf.append([classI, bboxes[i][4]])

        logging.debug(f"cls_with_conf {cls_with_conf}")

        original_height, original_width = img.shape[:2]
        img = np.array(cv2.resize(img, (224, 224)))
        x_scale = 224 / original_width
        y_scale = 224 / original_height

        if len(bboxes):
            scaled_bboxes = np.multiply(
                bboxes, [x_scale, y_scale, x_scale, y_scale, 1, 1]
            )
        else:
            scaled_bboxes = bboxes

        filtered_bboxes = []
        filtered_classes = []

        highest_conf_sun = {"bbox": None, "confidence": -1}

        for original_bbox, scaled_bbox, cls in zip(bboxes, scaled_bboxes, classes):
            if cls in cfg["NUDITY_CLASSES"]:
                if self.filter_nudity(img, scaled_bbox, cls):
                    filtered_bboxes.append(original_bbox)
                    filtered_classes.append(cls)
            elif cls in cfg["SKY_CLASSES"]:
                if self.filter_sky(img, scaled_bbox):
                    filtered_bboxes.append(original_bbox)
                    filtered_classes.append(cls)
            elif cls in cfg["WATER_CLASSES"]:
                if self.filter_water(img, scaled_bbox):
                    filtered_bboxes.append(original_bbox)
                    filtered_classes.append(cls)
            elif cls == "sun":
                if (
                    scaled_bbox[4] > cfg["DEFAULT_CONF_THRESHOLDS"][cls]
                    and scaled_bbox[4] > highest_conf_sun["confidence"]
                ):
                    highest_conf_sun["bbox"] = original_bbox
                    highest_conf_sun["confidence"] = scaled_bbox[4]
            else:
                if self.filter_default(img, scaled_bbox, cls):
                    filtered_bboxes.append(original_bbox)
                    filtered_classes.append(cls)

        if highest_conf_sun["bbox"]:
            filtered_bboxes.append(highest_conf_sun["bbox"])
            filtered_classes.append("sun")

        with Profiler("Filtering useless classes", logging):
            filtered_bboxes, filtered_classes = self.filter_by_object_detection_type(
                object_detection_type, filtered_bboxes, filtered_classes
            )

        logging.debug(f"filtered_classes {filtered_classes}")

        final_bboxes, final_classes = self.filter_bboxes(
            filtered_bboxes, filtered_classes
        )

        logging.debug(f"IOU filtered_classes {final_classes}")

        return final_bboxes, final_classes

    def is_bbox_inside(self, inner_bbox: np.ndarray, outer_bbox: np.ndarray) -> bool:
        ix1, iy1, ix2, iy2 = inner_bbox[:4]
        ox1, oy1, ox2, oy2 = outer_bbox[:4]
        return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2

    def bbox_size_percentage(
        self, bbox: np.ndarray, img_width: int, img_height: int
    ) -> float:
        x1, y1, x2, y2 = bbox[:4]
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        img_area = img_width * img_height
        return (bbox_area / img_area) * 100

    def filter_nudity(self, img: np.ndarray, bbox: np.ndarray, cls: str) -> bool:
        if bbox[4] < cfg["NUDITY_CONF_THRESHOLDS"][cls]:
            logging.debug(f"[nudity filter]: coef {cfg['NUDITY_CONF_THRESHOLDS'][cls]}")
            return False

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Ensure bounding box coordinates are within image dimensions
        x_min = max(0, int(bbox[0]))
        y_min = max(0, int(bbox[1]))
        x_max = min(img.shape[0], int(bbox[2]))
        y_max = min(img.shape[1], int(bbox[3]))

        # Crop the image
        cropped_img = img[y_min:y_max, x_min:x_max]

        logging.debug(f"f nud - cropped_img {cropped_img}")

        skin_pixels = np.logical_and.reduce(
            (
                cropped_img[:, :, 0] >= cfg["SKIN_COLOR_RANGE"][0][0],
                cropped_img[:, :, 0] <= cfg["SKIN_COLOR_RANGE"][1][0],
                cropped_img[:, :, 1] >= cfg["SKIN_COLOR_RANGE"][0][1],
                cropped_img[:, :, 1] <= cfg["SKIN_COLOR_RANGE"][1][1],
                cropped_img[:, :, 2] >= cfg["SKIN_COLOR_RANGE"][0][2],
                cropped_img[:, :, 2] <= cfg["SKIN_COLOR_RANGE"][1][2],
            )
        )

        logging.debug(f"skin_pixels {skin_pixels}")

        skin_percentage = np.mean(skin_pixels)

        logging.debug(f"[nudity filter]: % skin color {skin_percentage}")
        if skin_percentage < 0.4:  # Less than 40% skin color
            logging.debug("[nudity filter]: < 40% skin color")
            return False

        return True

    def filter_water(self, img: np.ndarray, bbox: np.ndarray) -> bool:
        if bbox[4] < cfg["WATER_CONF_THRESHOLD"]:
            return False

        # Convert the cropped image to HSV color space
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Crop the image using bbox
        cropped_img = img[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # Use vectorized operations for Color-based Thresholding in HSV
        grass_pixels = np.logical_and.reduce(
            (
                cropped_img[:, :, 0] >= cfg["GRASS_COLOR_RANGE"][0][0],
                cropped_img[:, :, 0] <= cfg["GRASS_COLOR_RANGE"][1][0],
                cropped_img[:, :, 1] >= cfg["GRASS_COLOR_RANGE"][0][1],
                cropped_img[:, :, 1] <= cfg["GRASS_COLOR_RANGE"][1][1],
                cropped_img[:, :, 2] >= cfg["GRASS_COLOR_RANGE"][0][2],
                cropped_img[:, :, 2] <= cfg["GRASS_COLOR_RANGE"][1][2],
            )
        )

        grass_percentage = np.mean(grass_pixels)

        logging.debug(f"[water filter]: grass percentage: {grass_percentage}")
        if grass_percentage > 0.6:
            logging.debug("[water filter]: > 60% grass color")
            return False

        return True

    def filter_sky(self, img: np.ndarray, bbox: np.ndarray) -> bool:
        if bbox[4] < cfg["SKY_CONF_THRESHOLD"]:
            return False

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Crop the image using bbox
        cropped_img = img[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # Use vectorized operations for Color-based Thresholding
        daytime_skies = np.logical_and.reduce(
            (
                cropped_img[:, :, 2] >= 100,
                cropped_img[:, :, 1] <= 255,
                cropped_img[:, :, 0] <= 255,
            )
        )
        nighttime_skies = np.all(cropped_img <= 150, axis=2)
        overcast_skies = np.logical_and.reduce(
            (
                cropped_img[:, :, 2] >= 100,
                cropped_img[:, :, 1] >= 100,
                cropped_img[:, :, 0] <= 255,
            )
        )

        sky_count = np.sum(daytime_skies | nighttime_skies | overcast_skies)

        w, h, _ = cropped_img.shape
        total_pixels = w * h
        sky_percentage = sky_count / total_pixels

        logging.debug(f"[sky filter]: sky percentage: {sky_percentage}")
        if sky_percentage < 0.4:
            logging.debug(f"[sky filter]: < 60% sky color {sky_percentage}")
            return False

        # Positional Rules
        img_height = img.shape[0]
        logging.debug(f"[sky filter]: top 1/3: {bbox[1]} v {img_height * 0.33}")
        if bbox[1] > img_height * 0.5:  # Top half of the image
            logging.debug("[sky filter]: not in top half")
            return False

        # Size and Aspect Ratio
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        img_area = img.shape[0] * img.shape[1]
        logging.debug(f"[sky filter]: area: {bbox_area / img_area}")
        if bbox_area < img_area * 0.05:  # Less than 5% of the image size
            logging.debug("[sky filter]: Less than 5% of the image size")
            return False

        return True

    def filter_default(self, img: np.ndarray, bbox: np.ndarray, cls: str) -> bool:
        cls = cls.replace(" ", "")

        if cls in cfg["DEFAULT_CONF_THRESHOLDS"]:
            if bbox[4] < cfg["DEFAULT_CONF_THRESHOLDS"][cls]:
                logging.debug(f"Filtered {cls} (conf: {bbox[4]})")
                return False
        elif bbox[4] < cfg["DEFAULT_CONF_THRESHOLDS"]["default"]:
            logging.debug(f"Filtered {cls} (conf: {bbox[4]})")
            return False

        return True

    def filter_by_object_detection_type(
        self, object_detection_type: str, bboxes: List[np.ndarray], classes: List[str]
    ) -> Tuple[List[np.ndarray], List[str]]:
        logging.debug(f"bboxes, classes {bboxes}, {classes}")
        if len(bboxes) == 0 and len(classes) == 0:
            return [], []

        zipped = zip(bboxes, classes)

        match object_detection_type:
            case IcrType.FULL:
                result_classes = cfg["OBJECT_DETECTION_TYPE_FULL_CLASSES"]
            case IcrType.SUN_ONLY:
                result_classes = cfg["OBJECT_DETECTION_TYPE_SUN_CLASSES"]
            case IcrType.VEGETATION_ONLY:
                result_classes = cfg["OBJECT_DETECTION_TYPE_VEGETATION_CLASSES"]

        filtered = filter(lambda zip: zip[1] in result_classes, zipped)
        logging.debug(f"filtered {filtered}")

        unzipped = list(zip(*filtered))
        filtered_bboxes, filtered_classes = unzipped
        return list(filtered_bboxes), list(filtered_classes)

    def filter_bboxes(
        self, bboxes: List[np.ndarray], classes: List[str]
    ) -> Tuple[List[np.ndarray], List[str]]:
        if len(bboxes) == 0 and len(classes) == 0:
            return [], []
        # Sort data by confidence
        zipped_data = sorted(zip(bboxes, classes), key=lambda x: x[0][4], reverse=True)
        sorted_bboxes, sorted_classes = zip(*zipped_data)
        sorted_bboxes_tensor = torch.tensor([bbox[:4] for bbox in sorted_bboxes])

        iou_matrix = box_iou(sorted_bboxes_tensor, sorted_bboxes_tensor)
        logging.debug(f"iou_matrix {iou_matrix}")

        keep_flags = [True] * len(sorted_bboxes)

        for i in range(len(sorted_bboxes)):
            if sorted_classes[i] in cfg["PRIORITY_IOU_PICK"]:
                continue
            for j in range(i + 1, len(sorted_bboxes)):
                if keep_flags[j] and iou_matrix[i][j] > cfg["IOU_PICK_THRESHOLD"]:
                    keep_flags[j] = False

        filtered_bboxes = [
            sorted_bboxes[i] for i, flag in enumerate(keep_flags) if flag
        ]
        filtered_classes = [
            sorted_classes[i] for i, flag in enumerate(keep_flags) if flag
        ]

        return list(filtered_bboxes), list(filtered_classes)
