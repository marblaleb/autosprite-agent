import base64
from datetime import datetime
from pathlib import Path

from src.api_client import ComfyUIClient
from src.processor import decode_image, remove_background, slice_frames, assemble_spritesheet
from src.templates import build_prompt, NEGATIVE_PROMPT


class Orchestrator:
    def __init__(self, config: dict):
        self.config = config
        comfy = config.get("comfy", {})
        self.client = ComfyUIClient(
            api_url=config["api_url"],
            checkpoint=comfy.get("checkpoint", "v1-5-pruned-emaonly.safetensors"),
            poll_interval=comfy.get("poll_interval", 2),
            poll_timeout=comfy.get("poll_timeout", 120),
        )

    def run(
        self,
        animation_type: str,
        character_desc: str,
        num_frames: int,
        reference_path: str = None,
    ) -> str:
        gen = self.config["generation_defaults"]
        proc = self.config["processing"]

        prompt = build_prompt(character_desc, animation_type, num_frames)

        if reference_path:
            init_b64 = self._image_to_base64(reference_path)
            raw_b64 = self.client.generate_img2img(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                init_image_b64=init_b64,
                denoising_strength=gen.get("denoising_strength", 0.75),
                steps=gen["steps"],
                cfg_scale=gen["cfg_scale"],
                width=gen["width"],
                height=gen["height"],
            )
        else:
            raw_b64 = self.client.generate_txt2img(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                steps=gen["steps"],
                cfg_scale=gen["cfg_scale"],
                width=gen["width"],
                height=gen["height"],
            )

        image = decode_image(raw_b64)
        image = remove_background(image)
        frames = slice_frames(image, num_frames)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = proc.get("output_dir", "./output")
        output_path = str(Path(output_dir) / f"spritesheet_{animation_type}_{timestamp}.png")

        return assemble_spritesheet(frames, output_path)

    def _image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
