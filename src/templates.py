ANIMATION_TEMPLATES = {
    "walk": "walking animation cycle",
    "run": "running animation cycle",
    "jump": "jump animation sequence",
    "attack": "sword attack animation",
}

ANIMATION_FRAME_POSES = {
    "walk": [
        "neutral stance, weight centered, both feet flat on ground",
        "right foot heel strike, left arm swinging forward, slight weight shift",
        "mid-step, right leg bearing weight, left leg lifting off",
        "left foot heel strike, right arm swinging forward, slight weight shift",
        "mid-step, left leg bearing weight, right leg lifting off",
        "full stride, right foot forward, left foot pushing off",
        "full stride, left foot forward, right foot pushing off",
        "recovery step, feet close together, body upright",
    ],
    "run": [
        "right foot contact, arms at midpoint, slight forward lean",
        "airborne phase, right knee driving forward, left leg trailing",
        "left foot contact, arms at midpoint, slight forward lean",
        "airborne phase, left knee driving forward, right leg trailing",
        "right foot push-off, arms spread wide, maximum lean",
        "full airborne, both knees bent, arms crossing chest",
        "left foot push-off, arms spread wide, maximum lean",
        "recovery phase, legs swinging through, arms pumping",
    ],
    "jump": [
        "crouch anticipation, knees bent deep, arms pulled back",
        "takeoff, legs fully extending, arms swinging upward",
        "ascending, feet leaving ground, arms rising overhead",
        "apex, body fully extended, arms wide for balance",
        "descending, body beginning to fall, arms lowering",
        "pre-landing, knees bending to absorb impact, arms forward",
        "landing impact, knees deeply bent, arms out for balance",
        "recovery, rising from crouch, returning to neutral",
    ],
    "attack": [
        "ready stance, sword at side, weight balanced, alert posture",
        "wind-up, sword drawn back over shoulder, body twisting",
        "swing start, sword moving forward, torso rotating",
        "mid-swing, sword horizontal, arm extended mid-way",
        "strike, sword fully extended, arm straight, body committed",
        "follow-through, sword past the strike point, body momentum",
        "deceleration, sword slowing, body recovering balance",
        "return to guard, sword back in ready position",
    ],
}

STYLE_SUFFIX = (
    "2d game sprite, pixel art style, clean outlines, no shadow, "
    "top lighting, isolated character, full body visible, centered, "
    "flat solid bright green background"
)

NEGATIVE_PROMPT = (
    "3d, realistic, photorealistic, gradient background, shadow, blur, "
    "deformed limbs, text, watermark, bad anatomy, extra limbs, merged frames, "
    "multiple characters, sprite sheet, grid, tiled"
)


def build_prompt(character_desc: str, animation_type: str, num_frames: int, frame_index: int) -> str:
    if animation_type not in ANIMATION_TEMPLATES:
        raise ValueError(
            f"Unknown animation type: {animation_type!r}. "
            f"Valid: {list(ANIMATION_TEMPLATES.keys())}"
        )
    animation_base = ANIMATION_TEMPLATES[animation_type]
    poses = ANIMATION_FRAME_POSES[animation_type]
    frame_pose = poses[frame_index % len(poses)]
    frame_label = f"frame {frame_index + 1} of {num_frames}"
    return f"{character_desc}, {animation_base}, {frame_pose}, {frame_label}, {STYLE_SUFFIX}"
