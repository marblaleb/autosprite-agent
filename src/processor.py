import base64
import io
import math
from datetime import datetime
from pathlib import Path

from PIL import Image
import rembg


class ProcessingError(Exception):
    pass


def decode_image(base64_str: str) -> Image.Image:
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGBA")


def slice_frames(image: Image.Image, num_frames: int) -> list:
    frame_width = image.width // num_frames
    if frame_width == 0:
        raise ValueError(
            f"Imagen demasiado pequeña ({image.width}px) para {num_frames} frames "
            "— reduce num_frames"
        )
    return [
        image.crop((i * frame_width, 0, (i + 1) * frame_width, image.height))
        for i in range(num_frames)
    ]


def remove_background(image: Image.Image) -> Image.Image:
    raise NotImplementedError


def assemble_spritesheet(frames: list, output_path: str) -> str:
    raise NotImplementedError
