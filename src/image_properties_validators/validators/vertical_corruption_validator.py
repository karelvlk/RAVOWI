import os
import joblib
import logging
import numpy as np
from scipy.stats import skew, kurtosis


class VerticalCorruptionValidator:
    def __init__(self) -> None:
        self.classifier = self.init_classifier("../models/svm_classifier83.00.joblib")

    def init_classifier(self, classifier_path: str):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        classifier_path = os.path.join(current_directory, classifier_path)

        classifier = None
        try:
            classifier = joblib.load(classifier_path)
        except Exception as e:
            logging.error(
                f"[VerticalCorruption Validator] initializing classifier failed: {e}"
            )

        return classifier

    def __call__(self, gray_image: np.ndarray) -> bool:
        std_devs = np.std(gray_image, axis=0)
        diff = np.diff(std_devs)
        # Note: Max decrease in std_dev
        max_diff_idx = np.argmax(diff)

        if max_diff_idx > 0:
            corrupted_region = gray_image[:, max_diff_idx:]
            avg_diff = np.mean(np.abs(np.diff(corrupted_region, axis=0)))
        else:
            avg_diff = 0

        # Summarizing std_devs into fixed-size statistics
        std_devs_mean = np.mean(std_devs)
        std_devs_std = np.std(std_devs)
        std_devs_skew = skew(std_devs)
        std_devs_kurtosis = kurtosis(std_devs)

        logging.debug(f"[VerticalCorruption Validator]: Max diff index: {max_diff_idx}")
        logging.debug(f"[VerticalCorruption Validator]: Average diff: {avg_diff}")
        logging.debug(f"[VerticalCorruption Validator]: Std devs mean: {std_devs_mean}")
        logging.debug(f"[VerticalCorruption Validator]: Std devs std: {std_devs_std}")
        logging.debug(f"[VerticalCorruption Validator]: Std devs skew: {std_devs_skew}")
        logging.debug(
            f"[VerticalCorruption Validator]: Std devs kurtosis: {std_devs_kurtosis}"
        )

        pred = self.classifier.predict(
            [
                [
                    std_devs_mean,
                    std_devs_std,
                    std_devs_skew,
                    std_devs_kurtosis,
                    max_diff_idx,
                    avg_diff,
                ]
            ]
        )

        logging.debug(f"[VerticalCorruption Validator]: {bool(pred[0])}")

        return bool(pred[0])
