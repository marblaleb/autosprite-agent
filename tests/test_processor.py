import base64
import io
import os
import tempfile
import pytest
from PIL import Image
from unittest.mock import patch
from src.processor import decode_image, slice_frames, remove_background, assemble_spritesheet, ProcessingError


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


def test_remove_background_returns_rgba():
    img = make_pil_image()
    with patch("rembg.remove", return_value=img):
        result = remove_background(img)
    assert result.mode == "RGBA"


def test_remove_background_failure_raises_processing_error():
    img = make_pil_image()
    with patch("rembg.remove", side_effect=Exception("model error")):
        with pytest.raises(ProcessingError, match="rembg falló"):
            remove_background(img)


def test_assemble_spritesheet_creates_file():
    frames = [make_pil_image(128, 128) for _ in range(4)]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_sheet.png")
        result = assemble_spritesheet(frames, output_path)
        assert result == output_path
        assert os.path.exists(output_path)


def test_assemble_spritesheet_output_is_png_rgba():
    frames = [make_pil_image(128, 128) for _ in range(4)]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_sheet.png")
        assemble_spritesheet(frames, output_path)
        with Image.open(output_path) as saved:
            assert saved.mode == "RGBA"


def test_assemble_spritesheet_empty_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="No hay frames"):
            assemble_spritesheet([], os.path.join(tmpdir, "out.png"))
