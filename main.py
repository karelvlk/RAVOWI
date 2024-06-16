import logging
from typing import Tuple

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response

from src.api.api_call_img_processor import ApiCallImgProcessor
from src.api.api_call_json_processor import ApiCallJsonProcessor
from src.image_content_recogniser import ImageContentRecogniser
from data_types.image_content_recogniser_type import IcrType
from src.settings import cfg


def initialize() -> (
    Tuple[ImageContentRecogniser, ApiCallImgProcessor, ApiCallJsonProcessor]
):
    icr = ImageContentRecogniser()
    icr_img = ApiCallImgProcessor(icr)
    icr_json = ApiCallJsonProcessor(icr)

    logging.info("Image-content-recogniser initialized")

    return icr, icr_img, icr_json


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    logging.getLogger().addHandler(console_handler)


# ----------------


setup_logger()
image_content_recogniser, icr_img, icr_json = initialize()

app = FastAPI(
    docs_url="/validator/routesz",
    openapi_url="/validator/openapi.json",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.info("FastAPI initialized")


@app.post("/validator/detect-img")
async def upload_image_for_detect_img(file: UploadFile = File(...)) -> Response:
    return await icr_img(file)


@app.post("/validator/detect-json")
async def upload_image_for_detect_json(file: UploadFile = File(...)) -> ORJSONResponse:
    return await icr_json(file, IcrType.FULL)


@app.get("/validator/healthz")
@app.get("/healthz")
def healthz() -> Response:
    return Response("OK", status_code=200)
