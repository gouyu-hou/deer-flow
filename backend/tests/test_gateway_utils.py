"""Tests for shared Gateway utility helpers."""

import pytest

from app.gateway.utils import constant_time_equals, sanitize_log_param


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("token", "token", True),
        ("token", "other", False),
        ("tok\xe9n", "tok\xe9n", True),
        ("token", "tok\xe9n", False),
        ("tok\xe9n", "token", False),
        ("tok\ud800n", "token", False),
        ("", "", True),
    ],
)
def test_constant_time_equals_never_raises_on_non_ascii(a, b, expected):
    assert constant_time_equals(a, b) is expected


@pytest.mark.parametrize(
    "raw",
    [
        "abc\n",
        "a\r\nb",
        "a\rb",
        "a\x00b",
        "a\x1bb",
        "\x1b[2K\x1b[1Gforged",
        "a\x7fb",
        "a\x85b",
        "a\x9bb",
        "a\u2028b",
        "a\u2029b",
    ],
)
def test_sanitize_log_param_removes_every_line_break_and_control_char(raw):
    """Log injection goes through a terminal escape or a lone line separator too."""
    cleaned = sanitize_log_param(raw)

    assert not any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in cleaned)
    assert "\u2028" not in cleaned and "\u2029" not in cleaned


def test_sanitize_log_param_keeps_ordinary_text():
    assert sanitize_log_param("thread-01J8Z / ünïcode") == "thread-01J8Z / ünïcode"
