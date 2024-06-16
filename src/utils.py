import contextlib
import glob
import hashlib
import logging
import math
import os
import re
import urllib
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import parse
import requests
import torch
import yaml
from PIL import Image
from torch import Tensor

ROOT = Path(os.getcwd())


def yaml_load(file: str = "data.yaml") -> Any:
    # Single-line safe yaml loading
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)


def normalize_yolo_bbox(
    bbox: List[float], img_shape: Tuple[int, int, int]
) -> List[float]:
    img_height, img_width, _ = img_shape
    x_min, y_min, x_max, y_max = bbox[:4]

    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return [x_center, y_center, width, height]


def cfg2dict(cfg: dict) -> dict:
    """
    Convert a configuration object to a dictionary, whether it is a file path, a string, or a SimpleNamespace object.

    Args:
        cfg (str | Path | SimpleNamespace): Configuration object to be converted to a dictionary.

    Returns:
        cfg (dict): Configuration object in dictionary format.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def prepare_image_for_stream_response(img: Image) -> BytesIO:
    img_io = BytesIO()
    img.save(img_io, format="PNG")
    img_io.seek(0)

    return img_io


def make_divisible(x: Union[int, float], divisor: Union[int, float, Tensor]) -> int:
    # Returns nearest x divisible by divisor
    if isinstance(divisor, Tensor):
        divisor = int(divisor.max())  # convert to int
    return math.ceil(x / divisor) * divisor


def check_img_size_stride(
    img_size: Union[int, List[int], Tuple[int, ...]], stride: int = 32, floor: int = 0
) -> Union[int, List[int]]:
    # Verify image size is a multiple of stride in each dimension
    if isinstance(img_size, int):  # integer i.e., img_size=640
        new_size = max(make_divisible(img_size, int(stride)), floor)
    else:  # list or tuple i.e., img_size=[640, 480]
        img_size = list(img_size)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(stride)), floor) for x in img_size]
    if new_size != img_size:
        logging.warning(
            f"img_size {img_size} must be multiple of max stride {stride}, updating to {new_size}"
        )
    return new_size


def increment_path(
    path: str, exist_ok: bool = False, sep: str = "", mkdir: bool = False
) -> str:
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )

        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def curl_download(url, filename, *, silent: bool = False) -> bool:
    """
    Download a file from a url to a filename using requests.
    """
    try:
        # Send a GET request to the URL
        with requests.get(
            url, stream=True, headers={"Range": "bytes=0-"}, timeout=5
        ) as response:
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Write the content to the file
            with open(filename, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    if not silent:
                        print(".", end="", flush=True)
                if not silent:
                    print()  # Newline after download progress dots

        return True
    except requests.RequestException as e:
        if not silent:
            print(f"Error: {e}")
        return False


def check_suffix(file: str, suffix: Union[str, Tuple[str]], msg: str = "") -> None:
    """Check file(s) for acceptable suffix."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}, not {s}"


def url2file(url: str) -> str:
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name


def clean_url(url: str) -> str:
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = str(Path(url)).replace(":/", "://")  # Pathlib turns :// -> :/
    # '%2F' to '/', split https://url.com/file.txt?auth
    return urllib.parse.unquote(url).split("?")[0]


def check_file(
    file: str, suffix: str = "", download: bool = True, hard: bool = True
) -> str:
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    # exists ('://' check required in Windows Python<3.10)
    if not file or ("://" not in file and Path(file).exists()):
        return file
    elif download and file.lower().startswith(
        ("https://", "http://", "rtsp://", "rtmp://")
    ):  # download
        url = file  # warning: Pathlib turns :// -> :/
        # '%2F' to '/', split https://url.com/file.txt?auth
        file = url2file(file)
        if Path(file).exists():
            # file already exists
            logging.info(f"Found {clean_url(url)} locally at {file}")
        else:
            safe_download(url=url, file=file, unzip=False)
        return file
    else:  # search
        files = []
        for d in (
            "models",
            "datasets",
            "tracker/cfg",
            "yolo/cfg",
        ):  # search directories
            files.extend(
                glob.glob(str(ROOT / d / "**" / file), recursive=True)
            )  # find file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")

        if len(files) > 1 and hard:
            raise FileNotFoundError(
                f"Multiple files match '{file}', specify exact path: {files}"
            )
        return files[0] if files else []  # return file


def safe_download(url: str, file: str, unzip: bool = False) -> None:
    try:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        logging.warning(f"HTTP Error: {errh}")
        return
    except requests.exceptions.ConnectionError as errc:
        logging.warning(f"Error Connecting: {errc}")
        return
    except requests.exceptions.Timeout as errt:
        logging.warning(f"Timeout Error: {errt}")
        return
    except requests.exceptions.RequestException as err:
        logging.warning(f"Something went wrong: {err}")
        return

    with open(file, "wb") as fd:
        for chunk in response.iter_content(chunk_size=1024):
            fd.write(chunk)

    if unzip:
        if zipfile.is_zipfile(file):
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(file))
        else:
            logging.warning(f"{file} is not a zipfile.")


def is_url(url: str, check: bool = True) -> bool:
    """Check if string is URL and check if URL exists."""
    with contextlib.suppress(Exception):
        url = str(url)
        result = parse.urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False

        if check:
            with requests.urlopen(url) as response:
                return response.getcode() == 200  # check if exists online
        return True
    return False


def colorstr(*input: str) -> str:
    """Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')."""

    USE_COLORS = False
    if not USE_COLORS:
        return input[-1]

    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def segments2boxes(segments: List[List[List[int]]]) -> np.ndarray:
    """
    It converts segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)

    Args:
        segments (list): list of segments, each segment is a list of points, each point is a list of x, y coordinates

    Returns:
        (np.ndarray): the xywh coordinates of the bounding boxes.
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box

    Args:
        masks (torch.Tensor): [h, w, n] tensor of masks
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
        (torch.Tensor): The masks are being cropped to the bounding box.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[
        None, None, :
    ]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[
        None, :, None
    ]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def xyxy2xywh(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xyn2xy(
    x: Union[np.ndarray, torch.Tensor],
    w: int = 640,
    h: int = 640,
    padw: int = 0,
    padh: int = 0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert normalized coordinates to pixel coordinates of shape (n,2)

    Args:
        x (np.ndarray | torch.Tensor): The input tensor of normalized bounding box coordinates
        w (int): The width of the image. Defaults to 640
        h (int): The height of the image. Defaults to 640
        padw (int): The width of the padding. Defaults to 0
        padh (int): The height of the padding. Defaults to 0
    Returns:
        y (np.ndarray | torch.Tensor): The x and y coordinates of the top left corner of the bounding box
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y


def xywhn2xyxy(
    x: Union[np.ndarray, torch.Tensor],
    w: int = 640,
    h: int = 640,
    padw: int = 0,
    padh: int = 0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert normalized bounding box coordinates to pixel coordinates.

    Args:
        x (np.ndarray | torch.Tensor): The bounding box coordinates.
        w (int): Width of the image. Defaults to 640
        h (int): Height of the image. Defaults to 640
        clip (bool): If True, the boxes will be clipped to the image boundaries. Defaults to False
        eps (float): The minimum value of the box's width and height. Defaults to 0.0
    Returns:
        y (np.ndarray | torch.Tensor): The coordinates of the bounding box in the format [x1, y1, x2, y2] where
            x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def clip_boxes(boxes: Union[np.ndarray, torch.Tensor], shape: Tuple[int, int]) -> None:
    """
    It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
    shape

    Args:
        boxes (torch.Tensor): the bounding boxes to clip
        shape (tuple): the shape of the image
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def xyxy2xywhn(
    x: Union[np.ndarray, torch.Tensor],
    w: int = 640,
    h: int = 640,
    clip: bool = False,
    eps: float = 0.0,
):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format.
    x, y, width and height are normalized to image dimensions

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): The width of the image. Defaults to 640
        h (int): The height of the image. Defaults to 640
        clip (bool): If True, the boxes will be clipped to the image boundaries. Defaults to False
        eps (float): The minimum value of the box's width and height. Defaults to 0.0
    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height, normalized) format
    """
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def clean_str(s: str) -> str:
    """
    Cleans a string by replacing special characters with underscore _

    Args:
        s (str): a string needing special characters replaced

    Returns:
        (str): a string with special characters replaced by an underscore _
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def resample_segments(segments: List[np.ndarray], n: int = 1000) -> List[np.ndarray]:
    """
    Inputs a list of segments (n,2) and returns a list of segments (n,2) up-sampled to n points each.

    Args:
        segments (list): a list of (n,2) arrays, where n is the number of points in the segment.
        n (int): number of points to resample the segment to. Defaults to 1000

    Returns:
        segments (list): the resampled segments.
    """
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)])
            .reshape(2, -1)
            .T
        )  # segment xy
    return segments


def segment2box(
    segment: torch.Tensor, width: int = 640, height: int = 640
) -> np.ndarray:
    """
    Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)

    Args:
        segment (torch.Tensor): the segment label
        width (int): the width of the image. Defaults to 640
        height (int): The height of the image. Defaults to 640

    Returns:
        (np.ndarray): the minimum and maximum x and y values of the segment.
    """
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    (
        x,
        y,
    ) = (
        x[inside],
        y[inside],
    )
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )  # xyxy


def yaml_save(file: str = "data.yaml", data: dict = None) -> None:
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.

    Returns:
        (None): Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    for k, v in data.items():
        if isinstance(v, Path):
            data[k] = str(v)

    # Dump data to file in YAML format
    with open(file, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def scale_boxes(
    img1_shape: Tuple[int, int],
    boxes: Union[np.ndarray, torch.Tensor],
    img0_shape: Tuple[int, int],
    ratio_pad: Tuple[Tuple[float, float], Tuple[int, int]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
        calculated based on the size difference between the two images.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def polygons2masks(
    imgsz: Tuple[int, int],
    polygons: List[np.ndarray],
    color: int,
    downsample_ratio: int = 1,
) -> np.ndarray:
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M], N is number of polygons, M is number of points (M % 2 = 0)
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    masks = []
    for poly in polygons:
        mask = polygon2mask(imgsz, [poly.reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)


def polygons2masks_overlap(
    imgsz: Tuple[int, int], segments: List[torch.Tensor], downsample_ratio: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a (640, 640) overlap mask."""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for seg in segments:
        mask = polygon2mask(
            imgsz,
            [seg.reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def polygon2mask(
    imgsz: Tuple[int, int],
    polygons: List[np.ndarray],
    color: int = 1,
    downsample_ratio: int = 1,
) -> np.ndarray:
    """
    Args:
        imgsz (tuple): The image size.
        polygons (list[np.ndarray]): [N, M], N is the number of polygons, M is the number of points(Be divided by 2).
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def get_hash(paths: List[str]) -> str:
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def img2label_paths(img_paths: List[str]) -> List[str]:
    """Define label paths as a function of image paths."""
    sa, sb = (
        f"{os.sep}images{os.sep}",
        f"{os.sep}labels{os.sep}",
    )  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def ltwh2xyxy(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    It converts the bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right

    Args:
        x (np.ndarray | torch.Tensor): the input image

    Returns:
        y (np.ndarray | torch.Tensor): the xyxy coordinates of the bounding boxes.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 2] + x[:, 0]  # width
    y[:, 3] = x[:, 3] + x[:, 1]  # height
    return y


def xywh2ltwh(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert the bounding box format from [x, y, w, h] to [x1, y1, w, h], where x1, y1 are the top-left coordinates.

    Args:
        x (np.ndarray | torch.Tensor): The input tensor with the bounding box coordinates in the xywh format
    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xyltwh format
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    return y


def ltwh2xywh(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert nx4 boxes from [x1, y1, w, h] to [x, y, w, h] where xy1=top-left, xy=center

    Args:
            x (torch.Tensor): the input tensor
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] + x[:, 2] / 2  # center x
    y[:, 1] = x[:, 1] + x[:, 3] / 2  # center y
    return y


def xyxy2ltwh(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert nx4 bounding boxes from [x1, y1, x2, y2] to [x1, y1, w, h], where xy1=top-left, xy2=bottom-right

    Args:
        x (np.ndarray | torch.Tensor): The input tensor with the bounding boxes coordinates in the xyxy format
    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xyltwh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
