import cv2
import logging
import numpy as np


class NoImageValidator:
    def __init__(
        self, std_dev_threshold: int = 35, edge_threshold: float = 0.12
    ) -> None:
        self.STD_DEV_THRESHOLD = std_dev_threshold
        self.EDGE_THRESHOLD = edge_threshold

    def __call__(self, grey_np_image: np.ndarray) -> bool:
        std_dev = np.std(grey_np_image)
        logging.debug(f"[NoImage Validator]: Standard deviation: {std_dev}")

        edges = cv2.Canny(grey_np_image, threshold1=30, threshold2=100)
        edge_percentage = np.count_nonzero(edges) / (
            grey_np_image.shape[0] * grey_np_image.shape[1]
        )
        logging.debug(f"[NoImage Validator]: Edge percentage: {edge_percentage}")

        if std_dev < self.STD_DEV_THRESHOLD and edge_percentage < self.EDGE_THRESHOLD:
            logging.debug(
                "[NoImage Validator]: This image is likely a single-color background with some text."
            )
            is_valid = False
        else:
            logging.debug("[NoImage Validator]: This image is likely a photo.")
            is_valid = True

        return is_valid
