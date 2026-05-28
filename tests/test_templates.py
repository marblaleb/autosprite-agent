import pytest
from src.templates import build_prompt, NEGATIVE_PROMPT


def test_build_prompt_includes_character():
    result = build_prompt("red knight", "walk", 4, 0)
    assert "red knight" in result


def test_build_prompt_walk_base():
    result = build_prompt("warrior", "walk", 4, 0)
    assert "walking animation cycle" in result


def test_build_prompt_run_base():
    result = build_prompt("warrior", "run", 4, 0)
    assert "running animation cycle" in result


def test_build_prompt_jump_base():
    result = build_prompt("warrior", "jump", 4, 0)
    assert "jump animation sequence" in result


def test_build_prompt_attack_base():
    result = build_prompt("warrior", "attack", 4, 0)
    assert "sword attack animation" in result


def test_build_prompt_includes_frame_label():
    result = build_prompt("warrior", "walk", 6, 2)
    assert "frame 3 of 6" in result


def test_build_prompt_different_frames_produce_different_prompts():
    r0 = build_prompt("warrior", "walk", 4, 0)
    r1 = build_prompt("warrior", "walk", 4, 1)
    assert r0 != r1


def test_build_prompt_invalid_type_raises():
    with pytest.raises(ValueError, match="Unknown animation type"):
        build_prompt("hero", "dance", 4, 0)


def test_negative_prompt_not_empty():
    assert len(NEGATIVE_PROMPT) > 0


def test_negative_prompt_excludes_3d():
    assert "3d" in NEGATIVE_PROMPT
