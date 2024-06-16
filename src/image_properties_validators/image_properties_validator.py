import logging
import numpy as np
from typing import Any, Tuple

from data_types.data_types import PropertiesValidatorsDTO

from src.profiler import Profiler
from src.image_properties_validators.validators.partial_download_validator import (
    PartialDownloadValidator,
)
from src.image_properties_validators.validators.vertical_corruption_validator import (
    VerticalCorruptionValidator,
)
from src.image_properties_validators.validators.no_image_validator import (
    NoImageValidator,
)


class ImagePropertiesValidator:
    def __init__(self) -> None:
        self.vertical_corruption_validator = VerticalCorruptionValidator()
        self.partial_download_validator = PartialDownloadValidator()
        self.no_image_validator = NoImageValidator()

    async def __call__(self, np_image: np.ndarray) -> PropertiesValidatorsDTO:
        # Convert to greyscale formula
        grey_array = (
            0.299 * np_image[:, :, 0]
            + 0.587 * np_image[:, :, 1]
            + 0.114 * np_image[:, :, 2]
        )
        grey_array = grey_array.astype(np.uint8)

        validators = {
            "vertical_corruption": (self.vertical_corruption_validator, (grey_array,)),
            "partial_download": (self.partial_download_validator, (grey_array,)),
            "no_image": (self.no_image_validator, (grey_array,)),
        }

        validator_outputs = {}
        for name, (validator, args) in validators.items():
            with Profiler(f"Validator '{name}'", logging):
                try:
                    result = validator(*args)
                    validator_outputs[name] = int(result)
                except Exception as e:
                    logging.error(f"Validation failed for {name} due to {e}")
                    validator_outputs[name] = 0  # or some default value

        conjunction = all(validator_outputs.values())
        return validator_outputs, conjunction
