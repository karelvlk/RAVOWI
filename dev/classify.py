import io
import logging

import numpy as np
import orjson
from PIL import Image
from tqdm import tqdm

from src.image_content_recogniser import ImageContentRecogniser


def process_image_data(json_file_path, output_file_path) -> None:
    print("START")
    # Initialize the Image_Content_Recogniser
    image_content_recogniser = ImageContentRecogniser()

    print("INITIALIZED")

    # Create the output file and write the start of the JSON array
    with open(output_file_path, "w") as out_file:
        out_file.write("[")

    # Read the input JSON file
    with open(json_file_path, "r") as in_file:
        data = orjson.loads(in_file)

        print("DATA:", len(data))

        for index, item in tqdm(enumerate(data), total=len(data)):
            try:
                # Open the image and convert it to bytes
                with Image.open(item["path"]) as img:
                    img_rescaled = img.resize((640, 640))
                    byte_array = io.BytesIO()
                    img_rescaled.save(byte_array, format="JPEG")

                # Get the response from Image_Content_Recogniser
                response = image_content_recogniser(np.array(img_rescaled), (640, 640))

                # Append the classes to the item
                item = {**item, **response}
                # item["classes"] = response["classes"]
                # item["boxes"] = response["boxes"]

                # Append the processed item to the output file
                with open(output_file_path, "a") as out_file:
                    orjson.dumps(item, out_file, indent=4)
                    # If it's not the last item, add a comma to separate JSON objects
                    if index < len(data) - 1:
                        out_file.write(",\n")

            except Exception as e:
                logging.error(f"Error processing {item['path']}: {e}")

    # Write the end of the JSON array to the output file
    with open(output_file_path, "a") as out_file:
        out_file.write("]")


if __name__ == "__main__":
    input_file_path = "/data/label-studio-niceness-processed-938.json"
    output_file_path = "/data/label-studio/niceness-classes-938.json"
    process_image_data(input_file_path, output_file_path)
