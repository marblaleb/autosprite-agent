# AutoSprite ComfyUI Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar `A1111Client` por `ComfyUIClient` en `src/api_client.py` manteniendo la interfaz pública intacta para que `Orchestrator` y `main.py` no requieran cambios sustanciales.

**Architecture:** `ComfyUIClient` carga un workflow JSON desde `workflows/`, inyecta parámetros buscando nodos por `_meta.title`, encola la generación via `POST /prompt`, hace polling a `GET /history/{id}`, y descarga la imagen via `GET /view`. La interfaz pública (`generate_txt2img` / `generate_img2img` → base64) es idéntica a `A1111Client`.

**Tech Stack:** Python 3.10+, requests, pytest, unittest.mock

---

## File Structure

```
AutoSprite Agent/
├── workflows/
│   ├── txt2img.json          # NEW: ComfyUI workflow template (txt2img)
│   └── img2img.json          # NEW: ComfyUI workflow template (img2img)
├── config.json               # MODIFIED: api_url + comfy section
├── src/
│   └── api_client.py         # REWRITE: A1111Client → ComfyUIClient
└── tests/
    ├── test_api_client.py    # REWRITE: tests for ComfyUI flow
    └── test_orchestrator.py  # SMALL UPDATE: CONFIG dict
```

---

## Task 1: Workflow Templates + Config

**Files:**
- Create: `workflows/txt2img.json`
- Create: `workflows/img2img.json`
- Modify: `config.json`

- [ ] **Step 1: Crear `workflows/txt2img.json`**

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "_meta": {"title": "AutoSprite_CheckpointLoader"},
    "inputs": {"ckpt_name": "pixel-art-v1.safetensors"}
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "AutoSprite_PositivePrompt"},
    "inputs": {"text": "", "clip": ["1", 1]}
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "AutoSprite_NegativePrompt"},
    "inputs": {"text": "", "clip": ["1", 1]}
  },
  "4": {
    "class_type": "EmptyLatentImage",
    "_meta": {"title": "AutoSprite_LatentImage"},
    "inputs": {"width": 512, "height": 512, "batch_size": 1}
  },
  "5": {
    "class_type": "KSampler",
    "_meta": {"title": "AutoSprite_KSampler"},
    "inputs": {
      "seed": 42,
      "steps": 25,
      "cfg": 7.5,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1.0,
      "model": ["1", 0],
      "positive": ["2", 0],
      "negative": ["3", 0],
      "latent_image": ["4", 0]
    }
  },
  "6": {
    "class_type": "VAEDecode",
    "_meta": {"title": "AutoSprite_VAEDecode"},
    "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
  },
  "7": {
    "class_type": "SaveImage",
    "_meta": {"title": "AutoSprite_SaveImage"},
    "inputs": {"filename_prefix": "AutoSprite", "images": ["6", 0]}
  }
}
```

- [ ] **Step 2: Crear `workflows/img2img.json`**

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "_meta": {"title": "AutoSprite_CheckpointLoader"},
    "inputs": {"ckpt_name": "pixel-art-v1.safetensors"}
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "AutoSprite_PositivePrompt"},
    "inputs": {"text": "", "clip": ["1", 1]}
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "AutoSprite_NegativePrompt"},
    "inputs": {"text": "", "clip": ["1", 1]}
  },
  "4": {
    "class_type": "LoadImage",
    "_meta": {"title": "AutoSprite_LoadImage"},
    "inputs": {"image": "", "upload": "image"}
  },
  "5": {
    "class_type": "VAEEncode",
    "_meta": {"title": "AutoSprite_VAEEncode"},
    "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}
  },
  "6": {
    "class_type": "KSampler",
    "_meta": {"title": "AutoSprite_KSampler"},
    "inputs": {
      "seed": 42,
      "steps": 25,
      "cfg": 7.5,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 0.75,
      "model": ["1", 0],
      "positive": ["2", 0],
      "negative": ["3", 0],
      "latent_image": ["5", 0]
    }
  },
  "7": {
    "class_type": "VAEDecode",
    "_meta": {"title": "AutoSprite_VAEDecode"},
    "inputs": {"samples": ["6", 0], "vae": ["1", 2]}
  },
  "8": {
    "class_type": "SaveImage",
    "_meta": {"title": "AutoSprite_SaveImage"},
    "inputs": {"filename_prefix": "AutoSprite", "images": ["7", 0]}
  }
}
```

- [ ] **Step 3: Reemplazar `config.json`**

```json
{
  "api_url": "http://127.0.0.1:8188",
  "comfy": {
    "checkpoint": "pixel-art-v1.safetensors",
    "poll_interval": 2,
    "poll_timeout": 120
  },
  "generation_defaults": {
    "steps": 25,
    "cfg_scale": 7.5,
    "width": 512,
    "height": 512,
    "denoising_strength": 0.75
  },
  "processing": {
    "remove_background_method": "rembg",
    "export_format": "png",
    "output_dir": "./output"
  }
}
```

- [ ] **Step 4: Verificar que los tests existentes no se rompen**

```
python -m pytest tests/test_templates.py tests/test_processor.py -v
```

Resultado esperado: 21 passed.

- [ ] **Step 5: Commit**

```
git add workflows/txt2img.json workflows/img2img.json config.json
git commit -m "chore: add ComfyUI workflow templates and update config"
```

---

## Task 2: ComfyUIClient — txt2img, polling y error handling

**Files:**
- Rewrite: `tests/test_api_client.py`
- Rewrite: `src/api_client.py`

- [ ] **Step 1: Reemplazar `tests/test_api_client.py` con los tests de txt2img**

```python
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
```

- [ ] **Step 2: Ejecutar para verificar que los tests fallan**

```
python -m pytest tests/test_api_client.py -v
```

Resultado esperado: `ImportError: cannot import name 'ComfyUIClient' from 'src.api_client'`

- [ ] **Step 3: Reemplazar `src/api_client.py` con `ComfyUIClient`**

```python
import base64
import copy
import json
import time
import uuid
from pathlib import Path

import requests

WORKFLOW_DIR = Path(__file__).parent.parent / "workflows"

_TXT2IMG_INJECTIONS = {
    "AutoSprite_CheckpointLoader": {"ckpt_name": "checkpoint"},
    "AutoSprite_PositivePrompt": {"text": "prompt"},
    "AutoSprite_NegativePrompt": {"text": "negative_prompt"},
    "AutoSprite_LatentImage": {"width": "width", "height": "height"},
    "AutoSprite_KSampler": {"steps": "steps", "cfg": "cfg_scale", "denoise": "denoise"},
}

_IMG2IMG_INJECTIONS = {
    "AutoSprite_CheckpointLoader": {"ckpt_name": "checkpoint"},
    "AutoSprite_PositivePrompt": {"text": "prompt"},
    "AutoSprite_NegativePrompt": {"text": "negative_prompt"},
    "AutoSprite_LoadImage": {"image": "init_image_filename"},
    "AutoSprite_KSampler": {"steps": "steps", "cfg": "cfg_scale", "denoise": "denoising_strength"},
}


class ComfyUIClient:
    def __init__(
        self,
        api_url: str,
        checkpoint: str,
        poll_interval: int = 2,
        poll_timeout: int = 120,
    ):
        self.api_url = api_url.rstrip("/")
        self.checkpoint = checkpoint
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout

    def generate_txt2img(
        self,
        prompt: str,
        negative_prompt: str,
        steps: int,
        cfg_scale: float,
        width: int,
        height: int,
    ) -> str:
        workflow = copy.deepcopy(self._load_workflow("txt2img.json"))
        params = {
            "checkpoint": self.checkpoint,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "denoise": 1.0,
        }
        self._inject(workflow, _TXT2IMG_INJECTIONS, params)
        return self._submit_and_wait(workflow)

    def generate_img2img(
        self,
        prompt: str,
        negative_prompt: str,
        init_image_b64: str,
        denoising_strength: float,
        steps: int,
        cfg_scale: float,
        width: int,
        height: int,
    ) -> str:
        raise NotImplementedError

    def _load_workflow(self, filename: str) -> dict:
        with open(WORKFLOW_DIR / filename) as f:
            return json.load(f)

    def _inject(self, workflow: dict, injections: dict, params: dict) -> None:
        found = set()
        for node in workflow.values():
            title = node.get("_meta", {}).get("title", "")
            if title in injections:
                found.add(title)
                for input_key, param_key in injections[title].items():
                    node["inputs"][input_key] = params[param_key]
        missing = set(injections.keys()) - found
        if missing:
            raise ValueError(f"Workflow missing required nodes: {sorted(missing)}")

    def _upload_image(self, image_b64: str) -> str:
        raise NotImplementedError

    def _submit_and_wait(self, workflow: dict) -> str:
        client_id = str(uuid.uuid4())
        prompt_id = self._queue_prompt(workflow, client_id)
        filename, subfolder = self._poll_until_done(prompt_id)
        return self._download_as_b64(filename, subfolder)

    def _queue_prompt(self, workflow: dict, client_id: str) -> str:
        try:
            resp = requests.post(
                f"{self.api_url}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            )
            resp.raise_for_status()
            return resp.json()["prompt_id"]
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"ComfyUI no está corriendo en {self.api_url}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Error en /prompt {e.response.status_code}: {e.response.text}"
            )

    def _poll_until_done(self, prompt_id: str) -> tuple:
        deadline = time.time() + self.poll_timeout
        while time.time() < deadline:
            resp = requests.get(f"{self.api_url}/history/{prompt_id}")
            resp.raise_for_status()
            data = resp.json()
            if prompt_id in data:
                history = data[prompt_id]
                if "error" in history:
                    raise RuntimeError(f"ComfyUI error: {history['error']}")
                for node_output in history.get("outputs", {}).values():
                    for img in node_output.get("images", []):
                        return img["filename"], img.get("subfolder", "")
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"Tiempo de espera agotado ({self.poll_timeout}s) "
            "— ComfyUI no respondió"
        )

    def _download_as_b64(self, filename: str, subfolder: str) -> str:
        resp = requests.get(
            f"{self.api_url}/view",
            params={"filename": filename, "subfolder": subfolder, "type": "output"},
        )
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("utf-8")
```

- [ ] **Step 4: Ejecutar los tests de Task 2**

```
python -m pytest tests/test_api_client.py -v
```

Resultado esperado: 8 passed.

- [ ] **Step 5: Commit**

```
git add src/api_client.py tests/test_api_client.py
git commit -m "feat: add ComfyUIClient with txt2img, polling and error handling"
```

---

## Task 3: ComfyUIClient — img2img

**Files:**
- Modify: `tests/test_api_client.py` (agregar 2 tests al final)
- Modify: `src/api_client.py` (implementar `generate_img2img` y `_upload_image`)

- [ ] **Step 1: Agregar los 2 tests de img2img al final de `tests/test_api_client.py`**

```python
def make_minimal_img2img_workflow():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "_meta": {"title": "AutoSprite_CheckpointLoader"}, "inputs": {"ckpt_name": ""}},
        "2": {"class_type": "CLIPTextEncode", "_meta": {"title": "AutoSprite_PositivePrompt"}, "inputs": {"text": "", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "_meta": {"title": "AutoSprite_NegativePrompt"}, "inputs": {"text": "", "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "_meta": {"title": "AutoSprite_LoadImage"}, "inputs": {"image": ""}},
        "5": {"class_type": "KSampler", "_meta": {"title": "AutoSprite_KSampler"}, "inputs": {"steps": 20, "cfg": 7.0, "denoise": 0.75, "seed": 42, "sampler_name": "euler", "scheduler": "normal", "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "_meta": {"title": "AutoSprite_VAEDecode"}, "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "_meta": {"title": "AutoSprite_SaveImage"}, "inputs": {"filename_prefix": "AutoSprite", "images": ["6", 0]}},
    }


def test_img2img_uploads_image_before_prompt():
    client = make_client()
    call_order = []
    init_b64 = base64.b64encode(b"fake_png_bytes").decode()

    def post_side(url, **kw):
        call_order.append(url)
        if "/upload/image" in url:
            m = MagicMock()
            m.raise_for_status = MagicMock()
            m.json.return_value = {"name": "ref.png"}
            return m
        return make_prompt_mock()

    with patch.object(client, "_load_workflow", return_value=make_minimal_img2img_workflow()), \
         patch("requests.post", side_effect=post_side), \
         patch("requests.get", side_effect=standard_get_side), \
         patch("time.sleep"):
        client.generate_img2img("knight", "bad", init_b64, 0.75, 25, 7.5, 512, 512)

    assert len(call_order) == 2
    assert "/upload/image" in call_order[0]
    assert "/prompt" in call_order[1]


def test_img2img_returns_base64():
    client = make_client()
    init_b64 = base64.b64encode(b"fake_png_bytes").decode()

    def post_side(url, **kw):
        if "/upload/image" in url:
            m = MagicMock()
            m.raise_for_status = MagicMock()
            m.json.return_value = {"name": "ref.png"}
            return m
        return make_prompt_mock()

    with patch.object(client, "_load_workflow", return_value=make_minimal_img2img_workflow()), \
         patch("requests.post", side_effect=post_side), \
         patch("requests.get", side_effect=standard_get_side), \
         patch("time.sleep"):
        result = client.generate_img2img("knight", "bad", init_b64, 0.75, 25, 7.5, 512, 512)

    assert result == base64.b64encode(make_png_bytes()).decode("utf-8")
```

- [ ] **Step 2: Ejecutar para verificar que los nuevos tests fallan**

```
python -m pytest tests/test_api_client.py::test_img2img_uploads_image_before_prompt tests/test_api_client.py::test_img2img_returns_base64 -v
```

Resultado esperado: FAILED con `NotImplementedError`.

- [ ] **Step 3: Reemplazar los dos stubs en `src/api_client.py`**

Reemplazar:

```python
    def generate_img2img(
        self,
        prompt: str,
        negative_prompt: str,
        init_image_b64: str,
        denoising_strength: float,
        steps: int,
        cfg_scale: float,
        width: int,
        height: int,
    ) -> str:
        raise NotImplementedError
```

Por:

```python
    def generate_img2img(
        self,
        prompt: str,
        negative_prompt: str,
        init_image_b64: str,
        denoising_strength: float,
        steps: int,
        cfg_scale: float,
        width: int,
        height: int,
    ) -> str:
        filename = self._upload_image(init_image_b64)
        workflow = copy.deepcopy(self._load_workflow("img2img.json"))
        params = {
            "checkpoint": self.checkpoint,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "denoising_strength": denoising_strength,
            "init_image_filename": filename,
        }
        self._inject(workflow, _IMG2IMG_INJECTIONS, params)
        return self._submit_and_wait(workflow)
```

Reemplazar:

```python
    def _upload_image(self, image_b64: str) -> str:
        raise NotImplementedError
```

Por:

```python
    def _upload_image(self, image_b64: str) -> str:
        image_bytes = base64.b64decode(image_b64)
        try:
            resp = requests.post(
                f"{self.api_url}/upload/image",
                files={"image": ("reference.png", image_bytes, "image/png")},
            )
            resp.raise_for_status()
            return resp.json()["name"]
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"ComfyUI no está corriendo en {self.api_url}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Error subiendo imagen {e.response.status_code}: {e.response.text}"
            )
```

- [ ] **Step 4: Ejecutar todos los tests de api_client**

```
python -m pytest tests/test_api_client.py -v
```

Resultado esperado: 10 passed.

- [ ] **Step 5: Commit**

```
git add src/api_client.py tests/test_api_client.py
git commit -m "feat: add ComfyUIClient img2img with image upload"
```

---

## Task 4: Wire Orchestrator + verificación final

**Files:**
- Modify: `src/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 1: Actualizar `src/orchestrator.py`**

Reemplazar:

```python
from src.api_client import A1111Client
```

Por:

```python
from src.api_client import ComfyUIClient
```

Reemplazar:

```python
        self.client = A1111Client(api_url=config["api_url"], timeout=120)
```

Por:

```python
        comfy = config.get("comfy", {})
        self.client = ComfyUIClient(
            api_url=config["api_url"],
            checkpoint=comfy.get("checkpoint", "v1-5-pruned-emaonly.safetensors"),
            poll_interval=comfy.get("poll_interval", 2),
            poll_timeout=comfy.get("poll_timeout", 120),
        )
```

- [ ] **Step 2: Actualizar el CONFIG dict en `tests/test_orchestrator.py`**

Reemplazar:

```python
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
```

Por:

```python
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
```

- [ ] **Step 3: Ejecutar la suite completa**

```
python -m pytest tests/ -v --tb=short
```

Resultado esperado: 35 passed (9 templates + 10 api_client + 12 processor + 4 orchestrator).

- [ ] **Step 4: Verificar imports**

```
python -c "import src.templates, src.api_client, src.processor, src.orchestrator; print('OK')"
```

Resultado esperado: `OK`

- [ ] **Step 5: Commit final**

```
git add src/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: wire Orchestrator to ComfyUIClient"
```
