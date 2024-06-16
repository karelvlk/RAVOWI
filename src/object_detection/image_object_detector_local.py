import itertools
import logging
import threading
from typing import Dict, List, Tuple, Union

from ultralytics import YOLO
from data_types.data_types import Bboxes
from src.object_detection.image_object_detector import ImageObjectDetector
from src.profiler import Profiler
from src.settings import cfg


class ImageObjectDetectorLocal(ImageObjectDetector):
    def __init__(self):
        logging.info("Starting Local ImageObjectDetector")
        super().__init__()
        self.gpu_lock = threading.Lock()  # Create a global lock for GPU access

    def setup_predictors(self) -> Dict[str, YOLO]:
        logging.info("ImageObjectDetector starts initializing detection predictors...")
        predictors = {
            "sky": YOLO(cfg["SKY_MODEL"]),
            "nud": YOLO(cfg["NUD_MODEL"]),
            "txt": YOLO(cfg["TXT_MODEL"]),
            "bse": YOLO(cfg["BSE_MODEL"]),
            "scu": YOLO(cfg["SCU_MODEL"]),
            "sun": YOLO(cfg["SUN_MODEL"]),
            "fce": YOLO(cfg["FCE_MODEL"]),
            "plt": YOLO(cfg["PLT_MODEL"]),
        }

        logging.info("Object Detection predictors successfully initialized")
        return predictors

    async def detect(
        self, img: Union[str, bytes], active_predictors: Dict[str, YOLO]
    ) -> Tuple[Bboxes, List[str]]:
        output_boxes: Bboxes = []
        output_classes: List[str] = []

        # Create a lock to ensure only one thread can access the predictor at a time
        lock = threading.Lock()

        # Create a thread for each predictor
        threads = []
        results = [None] * len(active_predictors)
        cuda_img = img  # self.predictors[0].preprocess(img)
        for i, (key, predictor) in enumerate(active_predictors.items()):
            thread = threading.Thread(
                target=self.threaded_predict,
                args=(key, predictor, cuda_img, results, i, lock),
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        # Process the results
        for res in results:
            if res is not None:
                output_boxes.append(res[0])
                output_classes.append(res[1])
            logging.warning("Result of results in YOLO is None")

        output_boxes = list(itertools.chain(*output_boxes))
        output_classes = list(itertools.chain(*output_classes))

        return output_boxes, output_classes

    def predict(
        self, name: str, predictor: YOLO, img: Union[str, bytes]
    ) -> Tuple[Bboxes, List[str]]:
        with self.gpu_lock:  # Acquire the GPU lock
            with Profiler(f"YOLO [{name}] interference", logging):
                preds = predictor.predict(img)

        boxes = []
        classes = []
        for p in preds:
            b = p.boxes.data.cpu().numpy().tolist()
            boxes.extend(b)
            classes.extend([p.names[round(b[5])] for b in b])

        return boxes, classes

    def threaded_predict(
        self,
        name: str,
        predictor: YOLO,
        img: Union[str, bytes],
        results: List,
        i: int,
        lock: threading.Lock,
    ) -> None:
        with lock:
            boxes, classes = self.predict(name, predictor, img)
            results[i] = (boxes, classes)
