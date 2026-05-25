import pytest
from src.templates import build_prompt, NEGATIVE_PROMPT


def test_build_prompt_includes_character():
    result = build_prompt("red knight", "walk", 4)
    assert "red knight" in result


def test_build_prompt_walk_base():
    result = build_prompt("warrior", "walk", 4)
    assert "walking animation cycle" in result


def test_build_prompt_run_base():
    result = build_prompt("warrior", "run", 4)
    assert "running animation cycle" in result


def test_build_prompt_jump_base():
    result = build_prompt("warrior", "jump", 4)
    assert "jump animation sequence" in result


def test_build_prompt_attack_base():
    result = build_prompt("warrior", "attack", 4)
    assert "sword attack animation" in result


def test_build_prompt_includes_frame_count():
    result = build_prompt("warrior", "walk", 6)
    assert "6 frames" in result


def test_build_prompt_invalid_type_raises():
    with pytest.raises(ValueError, match="Unknown animation type"):
        build_prompt("hero", "dance", 4)


def test_negative_prompt_not_empty():
    assert len(NEGATIVE_PROMPT) > 0


def test_negative_prompt_excludes_3d():
    assert "3d" in NEGATIVE_PROMPT
