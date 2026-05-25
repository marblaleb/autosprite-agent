import pytest
import requests as req
from unittest.mock import patch, MagicMock
from src.api_client import A1111Client

BASE_URL = "http://127.0.0.1:7860"


def make_client():
    return A1111Client(api_url=BASE_URL, timeout=10)


def make_mock_response(image_b64="abc123"):
    mock = MagicMock()
    mock.json.return_value = {"images": [image_b64]}
    mock.raise_for_status = MagicMock()
    return mock


def test_txt2img_returns_base64():
    with patch("requests.post", return_value=make_mock_response("img_b64")):
        result = make_client().generate_txt2img("knight", "bad", 25, 7.5, 512, 512)
    assert result == "img_b64"


def test_txt2img_posts_to_correct_endpoint():
    with patch("requests.post", return_value=make_mock_response()) as mock_post:
        make_client().generate_txt2img("knight", "bad", 25, 7.5, 512, 512)
    assert mock_post.call_args[0][0] == f"{BASE_URL}/sdapi/v1/txt2img"


def test_img2img_returns_base64():
    with patch("requests.post", return_value=make_mock_response("img_b64")):
        result = make_client().generate_img2img("knight", "bad", "ref_b64", 0.75, 25, 7.5, 512, 512)
    assert result == "img_b64"


def test_img2img_posts_to_correct_endpoint():
    with patch("requests.post", return_value=make_mock_response()) as mock_post:
        make_client().generate_img2img("knight", "bad", "ref_b64", 0.75, 25, 7.5, 512, 512)
    assert mock_post.call_args[0][0] == f"{BASE_URL}/sdapi/v1/img2img"


def test_img2img_includes_init_image_in_payload():
    with patch("requests.post", return_value=make_mock_response()) as mock_post:
        make_client().generate_img2img("knight", "bad", "ref_b64", 0.75, 25, 7.5, 512, 512)
    payload = mock_post.call_args[1]["json"]
    assert payload["init_images"] == ["ref_b64"]


def test_connection_error_raises_connection_error():
    with patch("requests.post", side_effect=req.exceptions.ConnectionError()):
        with pytest.raises(ConnectionError, match="A1111 no está corriendo"):
            make_client().generate_txt2img("knight", "bad", 25, 7.5, 512, 512)


def test_timeout_raises_timeout_error():
    with patch("requests.post", side_effect=req.exceptions.Timeout()):
        with pytest.raises(TimeoutError, match="Tiempo de espera agotado"):
            make_client().generate_txt2img("knight", "bad", 25, 7.5, 512, 512)


def test_http_error_raises_runtime_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_resp.raise_for_status.side_effect = req.exceptions.HTTPError(response=mock_resp)
    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="500"):
            make_client().generate_txt2img("knight", "bad", 25, 7.5, 512, 512)
