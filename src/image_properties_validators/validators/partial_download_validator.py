import cv2
import numpy as np
import logging


class PartialDownloadValidator:
    def __init__(
        self,
        threshold: float = 0.005,
    ) -> None:
        self.THRESHOLD: float = threshold

    def __call__(self, gray_image: np.ndarray) -> bool:
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        gradient = np.gradient(np.sum(blurred_image, axis=1))

        significant_gradients = self.find_significant_gradient(gradient)

        if significant_gradients.size == 0:
            logging.debug(
                "[PartialDownload Validation] no significant gradient found",
            )
            return True

        cutoff_index = significant_gradients[-1]
        std_deviation_below_cutoff = np.std(gray_image[cutoff_index:, :])

        normalized_std_deviation = std_deviation_below_cutoff / (
            np.std(gray_image) + 1e-9
        )

        logging.debug(
            f"[PartialDownload Validation] normalized_std_deviation {normalized_std_deviation}",
        )

        return normalized_std_deviation >= self.THRESHOLD

    def find_significant_gradient(
        self, gradients, lower_threshold: float = 0.1, upper_threshold: float = 0.9
    ):
        gradient_differences = np.abs(np.diff(gradients))
        significant_changes = np.where(
            (
                gradient_differences
                > np.percentile(gradient_differences, lower_threshold * 100)
            )
            & (
                gradient_differences
                < np.percentile(gradient_differences, upper_threshold * 100)
            )
        )[0]
        return significant_changes
