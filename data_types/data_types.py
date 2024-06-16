from pydantic import BaseModel
from typing import List, Literal, Optional, Tuple, Union

# xmin, ymin, xmax, ymax, confidence, class
Bboxes = List[Tuple[float, float, float, float, float, Union[int, float]]]


class ObjectDetectionValidsDTO(BaseModel):
    vehicle: Optional[bool]
    person: Optional[bool]
    nudity: Optional[bool]
    text: Optional[bool]
    sky: Optional[bool]


class ClassificationDTO(BaseModel):
    category: Optional[Union[str, List[str]]]
    weather: Optional[Union[str, List[str]]]


class PropertiesValidatorsDTO(BaseModel):
    partial_download: Optional[Literal[0, 1]]
    vertical_corrupt: Optional[Literal[0, 1]]
    no_image: Optional[Literal[0, 1]]


class ResponseSunOnlyDTO(BaseModel):
    x_norm_center: float
    y_norm_center: float
    norm_width: float
    norm_height: float
    boxes: Bboxes
    classes: List[str]


class ResponseVegetationOnlyDTO(BaseModel):
    boxes: Bboxes
    classes: List[str]


class ResponseFullDTO(
    ObjectDetectionValidsDTO, ClassificationDTO, PropertiesValidatorsDTO
):
    boxes: Bboxes
    classes: List[str]


ResponseDTO = Union[ResponseSunOnlyDTO, ResponseVegetationOnlyDTO, ResponseFullDTO]


class ObjectDetectionDTO(BaseModel):
    gen_boxes: Bboxes
    gen_classes: List[str]
    boxes: Bboxes
    classes: List[str]
