import logging
from io import BytesIO

from fastapi import HTTPException
from PIL import Image


class ImageChecker:
    def check_image(self, image_bytes_io: BytesIO) -> None:
        logging.debug("Checking if image is valid.")
        try:
            img = Image.open(image_bytes_io)
            img.verify()
            logging.debug("Image is valid.")

        except Exception as e:
            logging.warning(f"Unable to open image. - {e}")
            raise HTTPException(
                status_code=415, detail="Uploaded file must be a valid image file."
            ) from e
