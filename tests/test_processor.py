import base64
import io
import os
import tempfile
import pytest
from PIL import Image
from unittest.mock import patch
from src.processor import decode_image, slice_frames


def make_b64_image(width=512, height=128):
    img = Image.new("RGBA", (width, height), (0, 255, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_pil_image(width=512, height=128):
    return Image.new("RGBA", (width, height), (0, 255, 0, 255))


def test_decode_image_returns_pil_image():
    result = decode_image(make_b64_image())
    assert isinstance(result, Image.Image)


def test_decode_image_correct_size():
    result = decode_image(make_b64_image(512, 128))
    assert result.size == (512, 128)


def test_decode_image_is_rgba():
    result = decode_image(make_b64_image())
    assert result.mode == "RGBA"


def test_slice_frames_returns_correct_count():
    frames = slice_frames(make_pil_image(512, 128), 4)
    assert len(frames) == 4


def test_slice_frames_correct_frame_width():
    frames = slice_frames(make_pil_image(512, 128), 4)
    assert frames[0].width == 128


def test_slice_frames_preserves_height():
    frames = slice_frames(make_pil_image(512, 128), 4)
    assert frames[0].height == 128


def test_slice_frames_zero_width_raises():
    with pytest.raises(ValueError, match="Imagen demasiado pequeña"):
        slice_frames(make_pil_image(3, 128), 4)
