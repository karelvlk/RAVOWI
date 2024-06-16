from typing import Any, Dict, List, Literal

import numpy as np

from src.settings import cfg


class ImageCategorizationPostrocessor:
    def category_postprocessing(
        self,
        category_classes: List[str],
        detected_response: Dict[str, Any],
        img_area: float,
    ) -> Dict[Literal['weather'], List[str]]:
        detected_objects = detected_response["classes"]
        detected_bboxes = detected_response["boxes"]
        detected_objects_np = np.array(detected_objects)

        filtered_categories = [
            category
            for category in category_classes
            if category not in cfg["CATEGORIZATION_MUST_ORS"]
            or all(
                any(np.isin(must_or, detected_objects_np))
                for must_or in cfg["CATEGORIZATION_MUST_ORS"][category]
            )
        ]

        for obj, bbox in zip(detected_objects, detected_bboxes):
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            match obj:
                case "sky":
                    if bbox_area / img_area > 0.25:
                        if "meteo" not in filtered_categories:
                            filtered_categories.append("meteo")

                case "road" | "sidewalk" | "crosswalk":
                    if bbox_area / img_area > 0.5:
                        if "traffic" not in filtered_categories:
                            filtered_categories.append("traffic")

                case "sea water":
                    if "sea" not in filtered_categories:
                        filtered_categories.append("sea")

        return {"category": list(filtered_categories)}
