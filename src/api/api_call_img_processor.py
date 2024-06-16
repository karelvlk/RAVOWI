import orjson
import io

from PIL import Image, ImageDraw
from fastapi import File, UploadFile
from fastapi.responses import Response
from typing import Any, Dict, Tuple
from src.api.api_call_processor import ApiCallProcessor
from src.image_content_recogniser import ImageContentRecogniser

# Data that should not be send in response
INTERNAL_DATA = [
    "boxes",
    "classes",
    "duration",
    "cls_category",
    "cls_weather",
    "gen_boxes",
    "gen_classes",
]


class ApiCallImgProcessor(ApiCallProcessor):
    def __init__(
        self,
        image_content_recogniser: ImageContentRecogniser,
    ):
        super().__init__(image_content_recogniser)

    async def __call__(self, file: UploadFile = File(...)) -> Response:
        response, image_bytes_io, _ = await self.detect(file)
        output_image_bytes_io, color = self.postprocess_image(image_bytes_io, response)
        response = Response(output_image_bytes_io, media_type="image/jpeg")

        for key, value in response.items():
            if key in INTERNAL_DATA:
                continue

            if not isinstance(value, str):
                value = orjson.dumps(value)
            key = f"-ai-{key}"
            response.headers[key] = value

        for key, value in color.items():
            key = f"-color-detection_{key}"
            response.headers[key] = value

        return response

    def postprocess_image(
        self, image_bytes_io: io.BytesIO, response: Dict[str, Any]
    ) -> Tuple[io.BytesIO, Dict[str, str]]:
        # Load the image
        image_bytes_io.seek(0)
        image = Image.open(image_bytes_io)

        # Draw bounding boxes
        draw = ImageDraw.Draw(image)
        VEHICLES = [
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
        ]
        NUDITY = [
            "EXPOSED_BELLY",
            "EXPOSED_BREAST_F",
            "EXPOSED_BREAST_M",
            "EXPOSED_BUTTOCKS",
            "EXPOSED_GENITALIA_F",
            "EXPOSED_GENITALIA_M",
        ]
        SKY = ["sky"]
        PERSON = ["person"]
        TEXT = ["text"]

        COLOR = {
            "vehicle": "green",
            "nudity": "red",
            "sky": "blue",
            "person": "orange",
            "text": "pink",
        }

        for i, bbox in enumerate(response["boxes"]):
            x_min, y_min, x_max, y_max, _, _ = bbox
            if response["classes"][i] in VEHICLES:
                color = COLOR["vehicle"]
            elif response["classes"][i] in NUDITY:
                color = COLOR["nudity"]
            elif response["classes"][i] in SKY:
                color = COLOR["sky"]
            elif response["classes"][i] in PERSON:
                color = COLOR["person"]
            elif response["classes"][i] in TEXT:
                color = COLOR["text"]
            else:
                continue

            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=4)

        # Save the modified image to a BytesIO object
        output_image_bytes_io = io.BytesIO()
        image.save(output_image_bytes_io, format="JPEG")
        output_image_bytes_io.seek(0)

        return output_image_bytes_io, COLOR
