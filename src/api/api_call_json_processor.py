import logging
import time
import numpy as np

from src.profiler import Profiler
from fastapi import File, UploadFile
from fastapi.responses import ORJSONResponse
from src.api.api_call_processor import ApiCallProcessor
from data_types.data_types import ResponseDTO
from data_types.image_content_recogniser_type import IcrType

class ApiCallJsonProcessor(ApiCallProcessor):
    async def __call__(
        self, file: UploadFile = File(...), icr_type: IcrType = IcrType.FULL
    ) -> ORJSONResponse:
        with Profiler("Whole process", logging):
            x_scale = 0.0
            y_scale = 0.0
            response, _, (x_scale, y_scale) = await self.detect(file, icr_type)
            response = self.rescale_bboxes(response, x_scale, y_scale)
            json_response: ORJSONResponse = ORJSONResponse(content=response)
            return json_response

    def rescale_bboxes(
        self,
        response: ResponseDTO,
        x_scale: float,
        y_scale: float,
    ) -> ResponseDTO:
        if "boxes" in response:
            bboxes: np.ndarray = np.array(response["boxes"])
            if len(bboxes):
                scaled_bboxes: np.ndarray = np.multiply(
                    bboxes, [x_scale, y_scale, x_scale, y_scale, 1, 1]
                )
                response["boxes"] = scaled_bboxes.tolist()
            else:
                response["boxes"] = bboxes

        return response
