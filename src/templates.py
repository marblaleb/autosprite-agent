ANIMATION_TEMPLATES = {
    "walk": "walking animation cycle, weight shift, legs alternating",
    "run": "running animation cycle, fast stride, body lean forward",
    "jump": "jump animation sequence, crouch takeoff, apex, landing",
    "attack": "sword attack animation, wind-up, strike, follow-through",
}

STYLE_SUFFIX = (
    "2d game sprite, pixel art style, clean outlines, no shadow, "
    "top lighting, white border between frames"
)

NEGATIVE_PROMPT = (
    "3d, realistic, photorealistic, gradient background, shadow, blur, "
    "deformed limbs, text, watermark, bad anatomy, extra limbs, merged frames"
)


def build_prompt(character_desc: str, animation_type: str, num_frames: int) -> str:
    if animation_type not in ANIMATION_TEMPLATES:
        raise ValueError(
            f"Unknown animation type: {animation_type!r}. "
            f"Valid: {list(ANIMATION_TEMPLATES.keys())}"
        )
    template_base = ANIMATION_TEMPLATES[animation_type]
    grid_suffix = (
        f"{num_frames} frames sprite sheet, arranged horizontally in a single row, "
        "flat solid bright green background, uniform frame size, consistent character scale"
    )
    return f"{character_desc}, {template_base}, {grid_suffix}, {STYLE_SUFFIX}"
