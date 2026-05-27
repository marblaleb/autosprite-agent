# ComfyUI Migration Design

**Date:** 2026-05-27  
**Scope:** Replace A1111 backend with ComfyUI in AutoSprite Agent  
**Status:** Approved

---

## Goal

Reemplazar `A1111Client` por `ComfyUIClient` como backend de generación de imágenes, manteniendo la interfaz pública intacta para que `Orchestrator`, `main.py` y los demás módulos no requieran cambios.

---

## What Changes / What Stays

### Changed
| File | Change |
|------|--------|
| `src/api_client.py` | Complete rewrite: `A1111Client` → `ComfyUIClient` |
| `tests/test_api_client.py` | Complete rewrite for ComfyUI API mocking |
| `config.json` | `api_url` → `http://127.0.0.1:8188`; add `"comfy"` section |

### Added
| File | Purpose |
|------|---------|
| `workflows/txt2img.json` | ComfyUI API-format workflow template for txt2img |
| `workflows/img2img.json` | ComfyUI API-format workflow template for img2img |

### Unchanged
- `src/templates.py`
- `src/processor.py`
- `src/orchestrator.py`
- `main.py`
- `tests/test_templates.py`
- `tests/test_processor.py`
- `tests/test_orchestrator.py`

---

## ComfyUI API Flow

```
generate_txt2img / generate_img2img
  │
  ├─ 1. Load workflow JSON from workflows/txt2img.json or img2img.json
  │
  ├─ 2. [img2img only] POST /upload/image  →  server_filename
  │
  ├─ 3. Inject parameters into workflow (by node title lookup)
  │
  ├─ 4. POST /prompt  →  { "prompt_id": "abc-123" }
  │
  ├─ 5. Poll GET /history/{prompt_id} every poll_interval seconds
  │       until output appears or poll_timeout exceeded
  │
  ├─ 6. GET /view?filename=...&type=output  →  PNG bytes
  │
  └─ 7. base64-encode bytes  →  return str
```

---

## Workflow Template Convention

Workflow JSON files are standard ComfyUI API-format exports. Key nodes are identified by their `_meta.title` field. The client searches all nodes for these titles and injects the corresponding parameter.

| Node Title | Input Modified | Used In |
|---|---|---|
| `AutoSprite_KSampler` | `steps`, `cfg`, `denoise` | both |
| `AutoSprite_PositivePrompt` | `text` | both |
| `AutoSprite_NegativePrompt` | `text` | both |
| `AutoSprite_CheckpointLoader` | `ckpt_name` | both |
| `AutoSprite_LatentImage` | `width`, `height` | txt2img |
| `AutoSprite_LoadImage` | `image` | img2img |

Users can redesign their ComfyUI workflow (add upscalers, ControlNet, etc.) without changing code, as long as these node titles are preserved.

---

## config.json Changes

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

---

## Public Interface (unchanged)

```python
class ComfyUIClient:
    def __init__(self, api_url: str, timeout: int = 120): ...

    def generate_txt2img(
        self, prompt, negative_prompt, steps, cfg_scale, width, height
    ) -> str: ...  # returns base64 PNG

    def generate_img2img(
        self, prompt, negative_prompt, init_image_b64, denoising_strength,
        steps, cfg_scale, width, height
    ) -> str: ...  # returns base64 PNG
```

---

## Error Handling

| Situation | Exception |
|---|---|
| ComfyUI not running | `ConnectionError` |
| Polling timeout exceeded | `TimeoutError` |
| ComfyUI returns `error` in history | `RuntimeError` |
| Expected node title not found in workflow | `ValueError` |
| HTTP error on any endpoint | `RuntimeError` |

---

## Test Plan

All tests mock `requests.post` and `requests.get`. No live ComfyUI needed.

| Test | Validates |
|---|---|
| `test_generate_txt2img_returns_base64` | Full flow: prompt → history → view → base64 |
| `test_generate_img2img_uploads_image_first` | `/upload/image` called before `/prompt` |
| `test_polling_retries_until_ready` | History returns `{}` twice, then result |
| `test_polling_timeout_raises_timeout_error` | History never returns result |
| `test_connection_error_raises_connection_error` | `ConnectionError` on `/prompt` |
| `test_comfy_error_in_history_raises_runtime_error` | Error node in history output |
| `test_missing_node_title_raises_value_error` | Workflow missing expected title |
