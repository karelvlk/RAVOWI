import io
import logging
from typing import Tuple

import numpy as np

# import pyheif
import torch
from fastapi import File, UploadFile
from PIL import Image

from data_types.data_types import (
    ResponseDTO,
    ResponseSunOnlyDTO,
    ResponseVegetationOnlyDTO,
)
from src.image_checkers.image_checker import ImageChecker
from src.image_content_recogniser import ImageContentRecogniser
from data_types.image_content_recogniser_type import IcrType
from src.profiler import Profiler
from src.settings import cfg
from src.utils import normalize_yolo_bbox


class ApiCallProcessor:
    def __init__(
        self,
        image_content_recogniser: ImageContentRecogniser,
    ) -> None:
        self.image_content_recogniser = image_content_recogniser
        self.image_checker = ImageChecker()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def remap_response(
        self,
        icr_type: IcrType,
        response: ResponseDTO,
        shape: Tuple[int, int],
    ) -> ResponseDTO:
        if icr_type == IcrType.SUN_ONLY:
            normalized_sun_position = [None, None, None, None]
            if "sun" in response["classes"]:
                sun_index = response["classes"].index("sun")
                sun_bbox = response["boxes"][sun_index]
                if sun_bbox:
                    normalized_sun_position = normalize_yolo_bbox(sun_bbox, shape)

            return ResponseSunOnlyDTO(
                x_norm_center=normalized_sun_position[0],
                y_norm_center=normalized_sun_position[1],
                norm_width=normalized_sun_position[2],
                norm_height=normalized_sun_position[3],
                boxes=response["boxes"],
                classes=response["classes"],
            )
        elif icr_type == IcrType.VEGETATION_ONLY:
            return ResponseVegetationOnlyDTO(
                boxes=response["boxes"], classes=response["classes"]
            )
        else:
            return response

    async def detect(
        self, file: UploadFile = File(...), icr_type: IcrType = IcrType.FULL
    ) -> Tuple[
        ResponseDTO,
        io.BytesIO,
        Tuple[float, float],
    ]:
        with Profiler("Image preprocessing", logging):
            (
                np_image,
                io_bytes_image,
                (x_scale, y_scale),
            ) = self.preprocess_image(file)

        origo_shape = (np_image.shape[0] * x_scale, np_image.shape[1] * y_scale)

        response = await self.image_content_recogniser(np_image, origo_shape, icr_type)

        response = self.remap_response(icr_type, response, np_image.shape)

        response.pop("gen_boxes", None)
        response.pop("gen_classes", None)

        return response, io_bytes_image, (x_scale, y_scale)

    def adjust_image_orientation(self, image: Image) -> Image:
        try:
            exif_data = image._getexif()
            orientation_tag = 274  # Value for the 'Orientation' tag in EXIF data

            if exif_data and orientation_tag in exif_data:
                orientation = exif_data[orientation_tag]

                # Adjust orientation based on the value
                match orientation:
                    case 2:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    case 3:
                        image = image.rotate(180)
                    case 4:
                        image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
                    case 5:
                        image = image.rotate(-90, expand=True).transpose(
                            Image.FLIP_LEFT_RIGHT
                        )
                    case 6:
                        image = image.rotate(-90, expand=True)
                    case 7:
                        image = image.rotate(90, expand=True).transpose(
                            Image.FLIP_LEFT_RIGHT
                        )
                    case 8:
                        image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            return image

        return image

    def resize_image(
        self, image: Image, new_longer_side: int = 720
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        original_width, original_height = image.size

        longer_side = max(image.size)

        if longer_side > new_longer_side:
            shorter_side = min(image.size)
            new_shorter_side = int((new_longer_side / longer_side) * shorter_side)

            if image.size[0] > image.size[1]:
                new_size = (new_longer_side, new_shorter_side)
            else:
                new_size = (new_shorter_side, new_longer_side)

            resized_width, resized_height = new_size

            # Resize the image using OpenCV
            resized_image = np.array(image.resize(new_size))
        else:
            resized_image = np.array(image)
            resized_width, resized_height = image.size

        # Compute scale factors
        x_scale = original_width / resized_width
        y_scale = original_height / resized_height

        return resized_image, (x_scale, y_scale)

    def preprocess_image(
        self, file: UploadFile
    ) -> Tuple[np.ndarray, io.BytesIO, Tuple[float, float]]:
        contents = file.file.read()

        image_bytes_io = io.BytesIO(contents)

        self.image_checker.check_image(image_bytes_io)

        image = Image.open(image_bytes_io)

        image = self.adjust_image_orientation(image)

        with Profiler("Resizing image", logging):
            resized_np_img, (x_scale, y_scale) = self.resize_image(
                image, new_longer_side=cfg["RESIZING_LONGER_SIDE"]
            )

        return resized_np_img, image_bytes_io, (x_scale, y_scale)
