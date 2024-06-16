import logging
from typing import Dict, Union

import numpy as np
from ultralytics import YOLO

from data_types.data_types import ObjectDetectionDTO
from data_types.image_content_recogniser_type import IcrType
from src.object_detection.image_object_detection_postrocessor import (
    ImageObjectDetectorPostrocessor,
)
from src.profiler import Profiler


class ImageObjectDetector:
    def __init__(self):
        self.postprocessor = ImageObjectDetectorPostrocessor()
        self.predictors: Dict[str, Union[YOLO, str]] = self.setup_predictors()

    def setup_predictors(self) -> Dict[str, Union[YOLO, str]]:
        raise NotImplementedError("Predictors setup method is not implemented")

    async def __call__(
        self, img: np.array, object_detection_type: IcrType
    ) -> ObjectDetectionDTO:
        logging.debug(
            f"Active YOLO object detection predictors are: {self.predictors.keys()}"
        )

        with Profiler("YOLO Object Detecting took", logging):
            output_boxes, output_classes = await self.detect(img, self.predictors)

        with Profiler("Postrocessing of YOLOs' detections", logging):
            (
                post_output_boxes,
                post_output_classes,
            ) = self.postprocessor.postprocess_predictions(
                img, output_boxes, output_classes, object_detection_type
            )

        return {
            "gen_boxes": output_boxes,
            "gen_classes": output_classes,
            "boxes": post_output_boxes,
            "classes": post_output_classes,
        }
