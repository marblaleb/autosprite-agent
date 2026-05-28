import pytest
from unittest.mock import MagicMock, patch
from src.orchestrator import Orchestrator

CONFIG = {
    "api_url": "http://127.0.0.1:8188",
    "comfy": {
        "checkpoint": "test-model.safetensors",
        "poll_interval": 0,
        "poll_timeout": 120,
    },
    "generation_defaults": {
        "steps": 25,
        "cfg_scale": 7.5,
        "width": 512,
        "height": 512,
        "denoising_strength": 0.75,
    },
    "processing": {
        "output_dir": "./output",
    },
}


def _make_mock_frame():
    mock_img = MagicMock()
    mock_img.mode = "RGBA"
    return mock_img


def test_run_txt2img_calls_generate_per_frame():
    orch = Orchestrator(CONFIG)
    orch.client = MagicMock()
    orch.client.generate_txt2img.return_value = "fake_b64"
    mock_frame = _make_mock_frame()

    with patch("src.orchestrator.decode_image", return_value=mock_frame), \
         patch("src.orchestrator.remove_background", return_value=mock_frame), \
         patch("src.orchestrator.assemble_spritesheet", return_value="/tmp/out.png"):

        result = orch.run("walk", "red knight", 4)

    assert orch.client.generate_txt2img.call_count == 4
    orch.client.generate_img2img.assert_not_called()
    assert result == "/tmp/out.png"


def test_run_img2img_called_per_frame_when_reference_provided(tmp_path):
    ref_file = tmp_path / "ref.png"
    ref_file.write_bytes(b"\x89PNG\r\n")

    orch = Orchestrator(CONFIG)
    orch.client = MagicMock()
    orch.client.generate_img2img.return_value = "fake_b64"
    mock_frame = _make_mock_frame()

    with patch("src.orchestrator.decode_image", return_value=mock_frame), \
         patch("src.orchestrator.remove_background", return_value=mock_frame), \
         patch("src.orchestrator.assemble_spritesheet", return_value="/tmp/out.png"):

        orch.run("attack", "warrior", 4, reference_path=str(ref_file))

    assert orch.client.generate_img2img.call_count == 4
    orch.client.generate_txt2img.assert_not_called()


def test_run_generates_correct_number_of_frames():
    orch = Orchestrator(CONFIG)
    orch.client = MagicMock()
    orch.client.generate_txt2img.return_value = "fake_b64"
    mock_frame = _make_mock_frame()

    with patch("src.orchestrator.decode_image", return_value=mock_frame), \
         patch("src.orchestrator.remove_background", return_value=mock_frame), \
         patch("src.orchestrator.assemble_spritesheet", return_value="/tmp/out.png") as mock_assemble:

        orch.run("run", "hero", 6)

    assembled_frames = mock_assemble.call_args[0][0]
    assert len(assembled_frames) == 6


def test_run_progress_callback_called_per_frame():
    orch = Orchestrator(CONFIG)
    orch.client = MagicMock()
    orch.client.generate_txt2img.return_value = "fake_b64"
    mock_frame = _make_mock_frame()
    progress_calls = []

    with patch("src.orchestrator.decode_image", return_value=mock_frame), \
         patch("src.orchestrator.remove_background", return_value=mock_frame), \
         patch("src.orchestrator.assemble_spritesheet", return_value="/tmp/out.png"):

        orch.run("walk", "knight", 3, progress_callback=lambda c, t: progress_calls.append((c, t)))

    assert progress_calls == [(1, 3), (2, 3), (3, 3)]


def test_run_connection_error_propagates():
    orch = Orchestrator(CONFIG)
    orch.client = MagicMock()
    orch.client.generate_txt2img.side_effect = ConnectionError("ComfyUI no está corriendo en http://127.0.0.1:8188")

    with pytest.raises(ConnectionError, match="ComfyUI no está corriendo"):
        orch.run("walk", "knight", 4)
