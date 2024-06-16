import torch
import numpy as np
import torchvision.transforms.functional as F
from src.image_classifiers.base_image_classifier import BaseImageClassifier
from triton.triton_client import TritonClient


class ImageWeatherClassifier(BaseImageClassifier, TritonClient):
    def __init__(self) -> None:
        BaseImageClassifier.__init__(self)
        TritonClient.__init__(self)

        self.weather_classes = [
            "cloudy",
            "foggy",
            "rainy",
            "snowy",
            "sunny",
            "unknown",
        ]

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = F.resize(img_tensor, [256, 256], antialias=True)
        img_tensor = F.center_crop(img_tensor, [224, 224])
        img_tensor = img_tensor.unsqueeze(0)
        img_numpy = img_tensor.numpy()
        return img_numpy

    async def postprocess(self, raw_output: np.ndarray, data: dict):
        preds = np.argmax(raw_output, axis=1)
        return [self.weather_classes[p] for p in preds]

    async def __call__(self, img: np.ndarray):
        preprocessed_img = self.preprocess_img(img)
        results = await self.infer_img(
            preprocessed_img,
            "images",
            "FP32",
            ["output0"],
            "weather_classification",
            "1",
            {},
        )
        return {"weather": results[0]}
