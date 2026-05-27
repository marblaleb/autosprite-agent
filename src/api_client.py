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
        image_bytes = base64.b64decode(image_b64)
        try:
            resp = requests.post(
                f"{self.api_url}/upload/image",
                files={"image": ("reference.png", image_bytes, "image/png")},
            )
            resp.raise_for_status()
            data = resp.json()
            if "name" not in data:
                raise RuntimeError(f"Respuesta inesperada de /upload/image: {data!r}")
            return data["name"]
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"ComfyUI no está corriendo en {self.api_url}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Error subiendo imagen {e.response.status_code}: {e.response.text}"
            )

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

    def _poll_until_done(self, prompt_id: str) -> tuple[str, str]:
        deadline = time.time() + self.poll_timeout
        while time.time() < deadline:
            try:
                resp = requests.get(f"{self.api_url}/history/{prompt_id}")
                resp.raise_for_status()
            except requests.exceptions.ConnectionError:
                raise ConnectionError(f"ComfyUI no está corriendo en {self.api_url}")
            except requests.exceptions.HTTPError as e:
                raise RuntimeError(
                    f"Error en /history {e.response.status_code}: {e.response.text}"
                )
            data = resp.json()
            if prompt_id in data:
                history = data[prompt_id]
                if "error" in history:
                    raise RuntimeError(f"ComfyUI error: {history['error']}")
                for node_output in history.get("outputs", {}).values():
                    for img in node_output.get("images", []):
                        return img["filename"], img.get("subfolder", "")
                raise RuntimeError("ComfyUI reported completion but returned no images")
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"Tiempo de espera agotado ({self.poll_timeout}s) "
            "— ComfyUI no respondió"
        )

    def _download_as_b64(self, filename: str, subfolder: str) -> str:
        try:
            resp = requests.get(
                f"{self.api_url}/view",
                params={"filename": filename, "subfolder": subfolder, "type": "output"},
            )
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode("utf-8")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"ComfyUI no está corriendo en {self.api_url}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Error descargando imagen {e.response.status_code}: {e.response.text}"
            )
