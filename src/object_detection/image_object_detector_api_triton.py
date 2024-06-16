import cv2
import logging
import torch
import numpy as np

from src.settings import cfg
from ultralytics import YOLO
from typing import Dict, List, Tuple, Union
from ultralytics.yolo.utils.ops import non_max_suppression, scale_boxes
from data_types.data_types import Bboxes
from src.object_detection.image_object_detector import ImageObjectDetector
from triton.triton_client import TritonClient


class ImageObjectDetectorAPITriton(ImageObjectDetector, TritonClient):
    def __init__(self):
        ImageObjectDetector.__init__(self)
        TritonClient.__init__(self)

    def setup_predictors(self) -> Dict[str, YOLO]:
        # There should not be any need to setup predictors for Triton
        return {}

    def base_pre_transform(self, images, new_shape=(640, 640)):
        return [self.resize_and_pad_image(x, new_shape=new_shape) for x in images]

    def base_preprocess(self, im):
        _, w, h, _ = im.shape
        input_shape = (w, h)
        resized_shape = (640, 640)
        im = np.stack(self.base_pre_transform(im, new_shape=resized_shape))
        im = im[..., ::-1].transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)
        im = im.astype(np.float32)
        im /= 255
        return im, input_shape, resized_shape

    def resize_and_pad_image(
        self,
        image,
        new_shape=(640, 640),
        auto=False,
        scaleFill=False,
        scaleup=True,
        stride=32,
        center=True,
    ):
        shape = image.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = new_shape

        if center:
            dw /= 2
            dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(
                image.astype(float), new_unpad, interpolation=cv2.INTER_LINEAR
            )
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        res = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return res

    async def detect(
        self, img_batch: Union[str, bytes], active_predictors: Dict[str, YOLO]
    ) -> Tuple[Bboxes, List[str]]:
        output_classes = []
        output_boxes = []

        if img_batch.ndim == 3:
            img_batch = np.expand_dims(img_batch, axis=0)

        preprocessed_img, input_shape, resized_shape = self.base_preprocess(img_batch)

        results = await self.infer_img(
            preprocessed_img,
            "images",
            "FP32",
            [
                "ANATOMICAL_EXPOSURE_DETECTION_OUT",
                "COCO_DETECTION_OUT",
                "FACE_DETECTION_OUT",
                "LICENSE_PLATE_DETECTION_OUT",
                "SCENE-UNDERSTANDING_DETECTION_OUT",
                "SKY_DETECTION_OUT",
                "SUN_DETECTION_OUT",
                "TEXT_DETECTION_OUT",
            ],
            "pipeline",
            "1",
            {
                "input_shape": input_shape,
                "resized_shape": resized_shape,
            },
        )

        for ri, res in enumerate(results):
            if res is not None and len(res) > 0:
                output_boxes.extend(res[0][0])
                output_classes.extend(res[0][1])
            else:
                logging.warning("Result of results in YOLO is None")

        logging.debug(f"Resulting classes: {np.array(output_classes).shape}")
        logging.debug(f"Resulting bboxes: {np.array(output_boxes).shape}")

        return output_boxes, output_classes

    async def postprocess(self, preds, data):
        """Postprocesses predictions and returns a list of Results objects."""
        mapping = cfg[f"ITN_{data['model_name'].lower()[:-4]}"]
        try:
            preds_nms = non_max_suppression(
                [torch.from_numpy(preds.copy())], conf_thres=0.05, iou_thres=0.5
            )

            result = []
            for batch_item in preds_nms:
                batch_item = batch_item.numpy()

                if batch_item is not None:
                    class_predictions = [
                        mapping[str(int(prediction[5]))] for prediction in batch_item
                    ]
                else:
                    class_predictions = []

                batch_item[:, :4] = scale_boxes(
                    data["resized_shape"], batch_item[:, :4], data["input_shape"]
                )

                result.append([batch_item.tolist(), class_predictions])
            return result

        except Exception as exc:
            print(f"Error occurred while non-max suppression: {exc}.")

        return []
