import logging
import asyncio
import numpy as np

from typing import Dict, List, Tuple, Union
from src.image_classifiers.categorizer.image_categorizer import ImageCategorizer
from src.image_classifiers.weather_classifier.image_weather_classifier import (
    ImageWeatherClassifier,
)
from data_types.image_content_recogniser_type import IcrType
from src.image_properties_validators.image_properties_validator import (
    ImagePropertiesValidator,
)
from src.object_detection.image_object_detector_api_triton import (
    ImageObjectDetectorAPITriton,
)
from src.object_detection.image_object_detector_local import ImageObjectDetectorLocal
from src.profiler import Profiler
from src.settings import cfg


class ImageContentRecogniser:
    def __init__(
        self,
        validation: bool = True,
        categorization: bool = True,
        weather_classification: bool = True,
        detection: bool = True,
    ) -> None:
        self.ready: bool = False
        self.validation: bool = validation
        self.categorization: bool = categorization
        self.weather_classification: bool = weather_classification
        self.detection: bool = detection

        with Profiler("Initialize ImageContentRecogniser", logging):
            self.initialize()

    def initialize(self) -> None:
        if self.validation:
            self.image_properties_validator = ImagePropertiesValidator()

        if self.detection:
            self.image_object_detector = (
                ImageObjectDetectorAPITriton()
                if cfg["USE_MODEL_API"]
                else ImageObjectDetectorLocal()
            )

        if self.categorization:
            self.image_categorizer = ImageCategorizer()

        if self.weather_classification:
            self.image_weather_classifier = ImageWeatherClassifier()

        self.ready = True

    async def __call__(
        self, np_image: np.ndarray, origo_shape: Tuple[int, int], icr_type: IcrType
    ) -> Dict[str, Union[bool, Dict[str, Union[bool, float, List[str]]]]]:
        with Profiler("Image content recognition", logging):
            response: Dict[
                str, Union[bool, Dict[str, Union[bool, float, List[str]]]]
            ] = await self.run(np_image, origo_shape, icr_type)

        return response

    async def run(
        self, np_image: np.ndarray, origo_shape: Tuple[int, int], icr_type: IcrType
    ) -> Dict[str, Union[bool, Dict[str, Union[bool, float, List[str]]]]]:
        w, h, _ = np_image.shape
        img_size = [w, h]

        is_valid: bool = True

        tasks = []

        if self.validation:
            tasks.append(self.image_properties_validator(np_image))

        if self.detection:
            tasks.append(self.image_object_detector(np_image, icr_type))

        if self.weather_classification:
            tasks.append(self.image_weather_classifier(np_image))

        with Profiler("Image content recognition tasks", logging):
            results = await asyncio.gather(*tasks)

        if self.validation:
            prop_response, _ = results.pop(0)
        else:
            prop_response = {}

        if self.detection:
            object_detection_response = results.pop(0)
        else:
            object_detection_response = {}

        if self.weather_classification:
            weather_response = results.pop(0)
        else:
            weather_response = {}

        if self.categorization:
            with Profiler("Image categorization", logging):
                categorization_response = self.image_categorizer(
                    object_detection_response, img_size
                )
        else:
            categorization_response = {}

        if self.detection:
            is_valid, od_valids = self.get_is_valid(
                prop_response,
                object_detection_response["classes"],
                object_detection_response["boxes"],
                w * h,
            )
        else:
            od_valids = {}
            is_valid = prop_response["no_image"]

        return {
            "is_valid": is_valid,
            **od_valids,
            **prop_response,
            **categorization_response,
            **weather_response,
            **object_detection_response,
        }

    async def process_perf_test_detection(self, image_batch):
        return await self.image_object_detector.detect(image_batch, {})

    def intersection_area(self, bbox1: List[float], bbox2: List[float]) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        return (x2 - x1) * (y2 - y1) if x1 < x2 and y1 < y2 else 0

    def get_is_valid(
        self,
        prop_validation: Dict[str, Union[bool, float]],
        predicted_classes: List[str],
        predicted_bboxes: List[List[float]],
        image_area: int,
    ) -> Tuple[bool, Dict[str, bool]]:
        pred_cls = set(predicted_classes)

        filtered_nudity = set(cfg["FILTERED_NUDITY"])

        # Initialize variables to hold bounding box areas
        person_bbox_area: float = 0
        face_bbox_area: float = 0
        vehicle_bbox_area: float = 0
        text_bbox_area: float = 0
        biggest_text_box_area: float = 0
        license_plate_bbox_area: float = 0
        sun_bbox_area: float = 0
        sun_bbox: Union[List[float], None] = None
        sky_bboxes: List[List[float]] = []

        # Calculate bounding box areas
        for cls, bbox in zip(predicted_classes, predicted_bboxes):
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if cls in ["person", "face"]:
                face_bbox_area += bbox_area
            elif cls in ["license_plate", *cfg["FILTERED_VEHICLE"]]:
                vehicle_bbox_area += bbox_area
            elif cls == "sky":
                sky_bboxes.append(bbox)
            elif cls == "sun":
                sun_bbox = bbox
            elif cls == "text":
                text_bbox_area += bbox_area
                biggest_text_box_area = max(biggest_text_box_area, bbox_area)

        # Check conditions for 'person' and 'face'
        if "face" in pred_cls:
            no_significant_person: bool = False
            logging.debug("Validation failed for 'person' because 'face' was detected.")
        elif any(
            bbox_area / image_area >= 0.025
            for bbox_area in [person_bbox_area, face_bbox_area]
        ):
            no_significant_person = False
            logging.debug(
                "Validation failed for 'person' because one or more bounding boxes exceed 1% of the image area."
            )
        else:
            no_significant_person = (
                person_bbox_area + face_bbox_area
            ) < 0.1 * image_area
            if not no_significant_person:
                logging.debug(
                    "Validation failed for 'person' because total bounding box area exceeds 10% of the image area."
                )

        logging.debug(
            f"Total 'person' and 'face' bounding box area as a percentage of image area: {((person_bbox_area + face_bbox_area) / image_area) * 100:.2f}%"
        )

        # Check conditions for 'vehicle' and 'license_plate'
        if "license_plate" in pred_cls:
            no_significant_vehicle: bool = False
            logging.debug(
                "Validation failed for 'vehicle' because 'license_plate' was detected."
            )
        elif any(
            bbox_area / image_area >= 0.025
            for bbox_area in [vehicle_bbox_area, license_plate_bbox_area]
        ):
            no_significant_vehicle = False
            logging.debug(
                "Validation failed for 'vehicle' because one or more bounding boxes exceed 1% of the image area."
            )
        else:
            no_significant_vehicle = (
                vehicle_bbox_area + license_plate_bbox_area
            ) < 0.1 * image_area
            if not no_significant_vehicle:
                logging.debug(
                    "Validation failed for 'vehicle' because total bounding box area exceeds 10% of the image area."
                )

        logging.debug(
            f"Total 'vehicle' and 'license_plate' bounding box area as a percentage of image area: {((vehicle_bbox_area + license_plate_bbox_area) / image_area) * 100:.2f}%"
        )

        # Check conditions for 'text'
        no_significant_text = (text_bbox_area < 0.05 * image_area) and (
            biggest_text_box_area < 0.01 * image_area
        )
        if not no_significant_text:
            logging.debug(
                "Validation failed for 'text' because total bounding box area exceeds 5% of the image area."
            )

        is_sky: bool = "sky" in predicted_classes or "sun" in predicted_classes
        # Check for overlap between sun and sky bounding boxes
        if sun_bbox:
            sun_sky_overlap: bool = False  # Initialize variable
            sun_bbox_area = (sun_bbox[2] - sun_bbox[0]) * (sun_bbox[3] - sun_bbox[1])

            for sky_bbox in sky_bboxes:
                overlap_area = self.intersection_area(sun_bbox, sky_bbox)
                # Calculate percentage
                overlap_percentage = (overlap_area / sun_bbox_area) * 100
                logging.debug(f"sun/sky overlap_percentage: {overlap_percentage}")
                if overlap_percentage >= 50:  # At least 50% overlap
                    sun_sky_overlap = True
                    break

            if not sun_sky_overlap:
                logging.debug(
                    "Validation failed for 'sky' because less than 50% of the sun overlaps with the sky."
                )
                is_sky = False  # Update is_sky variable

        no_filtered_nudity: bool = pred_cls.isdisjoint(filtered_nudity)

        is_valid: bool = (
            is_sky
            and no_filtered_nudity
            and no_significant_vehicle
            and no_significant_person
        )

        if prop_validation:
            is_valid &= (
                prop_validation["no_image"]
                & prop_validation["vertical_corruption"]
                & prop_validation["partial_download"]
            )

        logging.info(f"Validity by objects: no_filtered_nudity: {no_filtered_nudity}")
        logging.info(
            f"Validity by objects: no_significant_vehicle: {no_significant_vehicle}"
        )
        logging.info(
            f"Validity by objects: no_significant_person: {no_significant_person}"
        )
        logging.info(f"Validity by objects: no_significant_text: {no_significant_text}")

        return is_valid, {
            "vehicle": no_significant_vehicle,
            "person": no_significant_person,
            "nudity": no_filtered_nudity,
            "sky": is_sky,
            "text": no_significant_text,
        }
