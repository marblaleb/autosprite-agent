import base64
import io
import pytest
import requests as req
from unittest.mock import patch, MagicMock
from PIL import Image
from src.api_client import ComfyUIClient

BASE_URL = "http://127.0.0.1:8188"
PROMPT_ID = "abc-123"


def make_client(poll_timeout=120, poll_interval=0):
    return ComfyUIClient(
        api_url=BASE_URL,
        checkpoint="test-model.safetensors",
        poll_interval=poll_interval,
        poll_timeout=poll_timeout,
    )


def make_minimal_txt2img_workflow():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "_meta": {"title": "AutoSprite_CheckpointLoader"}, "inputs": {"ckpt_name": ""}},
        "2": {"class_type": "CLIPTextEncode", "_meta": {"title": "AutoSprite_PositivePrompt"}, "inputs": {"text": "", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "_meta": {"title": "AutoSprite_NegativePrompt"}, "inputs": {"text": "", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "_meta": {"title": "AutoSprite_LatentImage"}, "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "5": {"class_type": "KSampler", "_meta": {"title": "AutoSprite_KSampler"}, "inputs": {"steps": 20, "cfg": 7.0, "denoise": 1.0, "seed": 42, "sampler_name": "euler", "scheduler": "normal", "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "_meta": {"title": "AutoSprite_VAEDecode"}, "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "_meta": {"title": "AutoSprite_SaveImage"}, "inputs": {"filename_prefix": "AutoSprite", "images": ["6", 0]}},
    }


def make_png_bytes():
    buf = io.BytesIO()
    Image.new("RGBA", (4, 4)).save(buf, format="PNG")
    return buf.getvalue()


def make_prompt_mock():
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {"prompt_id": PROMPT_ID}
    return m


def make_history_mock(ready=True):
    m = MagicMock()
    m.raise_for_status = MagicMock()
    if ready:
        m.json.return_value = {
            PROMPT_ID: {"outputs": {"7": {"images": [{"filename": "out.png", "subfolder": ""}]}}}
        }
    else:
        m.json.return_value = {}
    return m


def make_history_error_mock():
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {PROMPT_ID: {"outputs": {}, "error": "KSampler failed"}}
    return m


def make_view_mock():
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.content = make_png_bytes()
    return m


def standard_get_side(url, **kw):
    if "/history/" in url:
        return make_history_mock(ready=True)
    return make_view_mock()


# ── txt2img full flow ──────────────────────────────────────────────────────


def test_txt2img_returns_base64():
    client = make_client()
    with patch.object(client, "_load_workflow", return_value=make_minimal_txt2img_workflow()), \
         patch("requests.post", return_value=make_prompt_mock()), \
         patch("requests.get", side_effect=standard_get_side), \
         patch("time.sleep"):
        result = client.generate_txt2img("knight", "bad", 25, 7.5, 512, 512)
    assert result == base64.b64encode(make_png_bytes()).decode("utf-8")


def test_txt2img_posts_to_prompt_endpoint():
    client = make_client()
    with patch.object(client, "_load_workflow", return_value=make_minimal_txt2img_workflow()), \
         patch("requests.post", return_value=make_prompt_mock()) as mock_post, \
         patch("requests.get", side_effect=standard_get_side), \
         patch("time.sleep"):
        client.generate_txt2img("knight", "bad", 25, 7.5, 512, 512)
    assert mock_post.call_args[0][0] == f"{BASE_URL}/prompt"


# ── polling ────────────────────────────────────────────────────────────────


def test_polling_retries_until_ready():
    client = make_client(poll_timeout=60, poll_interval=0)
    call_count = [0]

    def get_side(url, **kw):
        if "/history/" in url:
            call_count[0] += 1
            return make_history_mock(ready=(call_count[0] >= 3))
        return make_view_mock()

    with patch.object(client, "_load_workflow", return_value=make_minimal_txt2img_workflow()), \
         patch("requests.post", return_value=make_prompt_mock()), \
         patch("requests.get", side_effect=get_side), \
         patch("time.sleep"):
        client.generate_txt2img("knight", "bad", 25, 7.5, 512, 512)

    assert call_count[0] == 3


def test_polling_timeout_raises_timeout_error():
    # poll_timeout=-1 ensures deadline is already past before first iteration
    client = make_client(poll_timeout=-1, poll_interval=0)
    with patch.object(client, "_load_workflow", return_value=make_minimal_txt2img_workflow()), \
         patch("requests.post", return_value=make_prompt_mock()):
        with pytest.raises(TimeoutError, match="Tiempo de espera agotado"):
            client.generate_txt2img("knight", "bad", 25, 7.5, 512, 512)


# ── error handling ─────────────────────────────────────────────────────────


def test_connection_error_raises_connection_error():
    client = make_client()
    with patch.object(client, "_load_workflow", return_value=make_minimal_txt2img_workflow()), \
         patch("requests.post", side_effect=req.exceptions.ConnectionError()):
        with pytest.raises(ConnectionError, match="ComfyUI no está corriendo"):
            client.generate_txt2img("knight", "bad", 25, 7.5, 512, 512)


def test_http_error_on_prompt_raises_runtime_error():
    client = make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_resp.raise_for_status.side_effect = req.exceptions.HTTPError(response=mock_resp)
    with patch.object(client, "_load_workflow", return_value=make_minimal_txt2img_workflow()), \
         patch("requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="500"):
            client.generate_txt2img("knight", "bad", 25, 7.5, 512, 512)


def test_comfy_error_in_history_raises_runtime_error():
    client = make_client()

    def get_side(url, **kw):
        if "/history/" in url:
            return make_history_error_mock()
        return make_view_mock()

    with patch.object(client, "_load_workflow", return_value=make_minimal_txt2img_workflow()), \
         patch("requests.post", return_value=make_prompt_mock()), \
         patch("requests.get", side_effect=get_side), \
         patch("time.sleep"):
        with pytest.raises(RuntimeError, match="KSampler failed"):
            client.generate_txt2img("knight", "bad", 25, 7.5, 512, 512)


def test_missing_node_title_raises_value_error():
    client = make_client()
    # Workflow missing AutoSprite_KSampler
    broken = {
        "1": {"class_type": "CheckpointLoaderSimple", "_meta": {"title": "AutoSprite_CheckpointLoader"}, "inputs": {"ckpt_name": ""}},
        "2": {"class_type": "CLIPTextEncode", "_meta": {"title": "AutoSprite_PositivePrompt"}, "inputs": {"text": ""}},
        "3": {"class_type": "CLIPTextEncode", "_meta": {"title": "AutoSprite_NegativePrompt"}, "inputs": {"text": ""}},
        "4": {"class_type": "EmptyLatentImage", "_meta": {"title": "AutoSprite_LatentImage"}, "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    }
    with patch.object(client, "_load_workflow", return_value=broken):
        with pytest.raises(ValueError, match="AutoSprite_KSampler"):
            client.generate_txt2img("knight", "bad", 25, 7.5, 512, 512)


def test_poll_completion_without_images_raises_runtime_error():
    client = make_client()

    def get_side(url, **kw):
        if "/history/" in url:
            m = MagicMock()
            m.raise_for_status = MagicMock()
            m.json.return_value = {
                PROMPT_ID: {"outputs": {"7": {"images": []}}}
            }
            return m
        return make_view_mock()

    with patch.object(client, "_load_workflow", return_value=make_minimal_txt2img_workflow()), \
         patch("requests.post", return_value=make_prompt_mock()), \
         patch("requests.get", side_effect=get_side), \
         patch("time.sleep"):
        with pytest.raises(RuntimeError, match="no images"):
            client.generate_txt2img("knight", "bad", 25, 7.5, 512, 512)
