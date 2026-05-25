import pytest
from unittest.mock import MagicMock, patch
from src.orchestrator import Orchestrator

CONFIG = {
    "api_url": "http://127.0.0.1:7860",
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


def _mock_processor_chain():
    mock_img = MagicMock()
    mock_img.mode = "RGBA"
    return {
        "decode_image": mock_img,
        "remove_background": mock_img,
        "slice_frames": [mock_img],
        "assemble_spritesheet": "/tmp/spritesheet_walk_123.png",
    }


def test_run_txt2img_calls_generate_txt2img():
    orch = Orchestrator(CONFIG)
    orch.client = MagicMock()
    orch.client.generate_txt2img.return_value = "fake_b64"
    p = _mock_processor_chain()

    with patch("src.orchestrator.decode_image", return_value=p["decode_image"]), \
         patch("src.orchestrator.remove_background", return_value=p["remove_background"]), \
         patch("src.orchestrator.slice_frames", return_value=p["slice_frames"]), \
         patch("src.orchestrator.assemble_spritesheet", return_value=p["assemble_spritesheet"]):

        result = orch.run("walk", "red knight", 4)

    orch.client.generate_txt2img.assert_called_once()
    orch.client.generate_img2img.assert_not_called()
    assert result == p["assemble_spritesheet"]


def test_run_img2img_called_when_reference_provided(tmp_path):
    ref_file = tmp_path / "ref.png"
    ref_file.write_bytes(b"\x89PNG\r\n")

    orch = Orchestrator(CONFIG)
    orch.client = MagicMock()
    orch.client.generate_img2img.return_value = "fake_b64"
    p = _mock_processor_chain()

    with patch("src.orchestrator.decode_image", return_value=p["decode_image"]), \
         patch("src.orchestrator.remove_background", return_value=p["remove_background"]), \
         patch("src.orchestrator.slice_frames", return_value=p["slice_frames"]), \
         patch("src.orchestrator.assemble_spritesheet", return_value=p["assemble_spritesheet"]):

        orch.run("attack", "warrior", 4, reference_path=str(ref_file))

    orch.client.generate_img2img.assert_called_once()
    orch.client.generate_txt2img.assert_not_called()


def test_run_passes_correct_num_frames_to_slice():
    orch = Orchestrator(CONFIG)
    orch.client = MagicMock()
    orch.client.generate_txt2img.return_value = "fake_b64"
    p = _mock_processor_chain()

    with patch("src.orchestrator.decode_image", return_value=p["decode_image"]), \
         patch("src.orchestrator.remove_background", return_value=p["remove_background"]), \
         patch("src.orchestrator.slice_frames", return_value=p["slice_frames"]) as mock_slice, \
         patch("src.orchestrator.assemble_spritesheet", return_value=p["assemble_spritesheet"]):

        orch.run("run", "hero", 6)

    mock_slice.assert_called_once()
    assert mock_slice.call_args[0][1] == 6


def test_run_connection_error_propagates():
    orch = Orchestrator(CONFIG)
    orch.client = MagicMock()
    orch.client.generate_txt2img.side_effect = ConnectionError("A1111 no está corriendo")

    with pytest.raises(ConnectionError, match="A1111 no está corriendo"):
        orch.run("walk", "knight", 4)
