import logging
import numpy as np

from src.settings import cfg
from typing import Any, List
from src.profiler import Profiler
from tritonclient.grpc import InferInput, InferRequestedOutput
from tritonclient.grpc.aio import InferenceServerClient


class TritonClient:
    def __init__(self):
        logging.info(f"Starting initializing Triton Client on url: {cfg['TRITON_URL']}")
        self.triton_client = InferenceServerClient(cfg["TRITON_URL"])
        logging.info("Triton client initialized")

    async def postprocess(self, raw_output: np.ndarray) -> Any:
        raise NotImplementedError()

    async def infer_img(
        self,
        input_img_batch: np.ndarray,
        input_layer_name: str,
        input_type: str,
        output_names: List[str],
        model_name: str,
        model_version: str,
        postprocess_data: dict,
    ) -> List[Any]:
        b, c, w, h = input_img_batch.shape
        inp = InferInput(input_layer_name, [b, c, w, h], input_type)
        inp.set_data_from_numpy(input_img_batch)

        out = [InferRequestedOutput(name) for name in output_names]

        with Profiler(f"Triton inference of model '{model_name}'", logging):
            raw_response = await self.triton_client.infer(
                model_name, inputs=[inp], outputs=out, model_version=model_version
            )

        with Profiler(f"Triton postprocessing of model '{model_name}'", logging):
            postprocessed = [
                await self.postprocess(
                    raw_response.as_numpy(out_name),
                    {**postprocess_data, "model_name": out_name},
                )
                for out_name in output_names
            ]

        return postprocessed
