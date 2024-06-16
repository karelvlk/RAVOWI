import json
import os
import toml

cfg = {}
config = toml.load("settings.toml")
for key, value in config.items():
    cfg[key.upper()] = value


def load_jsons_from_subdirs(directory):
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            json_file_path = os.path.join(subdir_path, "index_to_name.json")
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as json_file:
                    data = json.load(json_file)
                    cfg[f"ITN_{subdir}"] = data


load_jsons_from_subdirs("./triton/index-to-name")
