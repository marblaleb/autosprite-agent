import requests


class A1111Client:
    def __init__(self, api_url: str, timeout: int = 120):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def generate_txt2img(
        self,
        prompt: str,
        negative_prompt: str,
        steps: int,
        cfg_scale: float,
        width: int,
        height: int,
    ) -> str:
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
        }
        return self._post("/sdapi/v1/txt2img", payload)

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
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "init_images": [init_image_b64],
            "denoising_strength": denoising_strength,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
        }
        return self._post("/sdapi/v1/img2img", payload)

    def _post(self, endpoint: str, payload: dict) -> str:
        try:
            response = requests.post(
                f"{self.api_url}{endpoint}",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["images"][0]
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"A1111 no está corriendo en {self.api_url}")
        except requests.exceptions.Timeout:
            raise TimeoutError("Tiempo de espera agotado — intenta reducir steps o resolución")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Error A1111 {e.response.status_code}: {e.response.text}")
