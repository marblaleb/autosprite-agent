import base64
import io
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
    try:
        pad = max(image.width, image.height) // 8
        canvas = Image.new("RGBA", (image.width + 2 * pad, image.height + 2 * pad), (0, 0, 0, 0))
        canvas.paste(image, (pad, pad))
        result = rembg.remove(canvas)
        return result.crop((pad, pad, pad + image.width, pad + image.height))
    except Exception as e:
        raise ProcessingError(f"rembg falló: {e}")


def assemble_spritesheet(frames: list, output_path: str) -> str:
    if not frames:
        raise ValueError("No hay frames para ensamblar")

    min_w = min(f.width for f in frames)
    min_h = min(f.height for f in frames)
    normalized = [f.resize((min_w, min_h), Image.LANCZOS) for f in frames]

    sheet_w = min_w * len(normalized)
    sheet_h = min_h

    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
    for i, frame in enumerate(normalized):
        sheet.paste(frame, (i * min_w, 0), frame)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, "PNG")
    return output_path
