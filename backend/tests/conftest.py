"""Shared pytest fixtures for the OMNIMIND backend test suite.

This module wires up environment variables and any cross-cutting fixtures so
that individual test modules can stay focused on behaviour rather than setup.
The ``set_env`` fixture is ``autouse=True`` which means it runs before every
test in the suite without an explicit request, guaranteeing that imported
modules see deterministic configuration even when they read environment
variables at import time.
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Inject deterministic environment variables for every test.

    The values here are intentionally non-functional (``sk-test``,
    ``tvly-test``) so that any accidental network call is rejected by the
    upstream API rather than silently succeeding against a real account.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    monkeypatch.setenv("CHROMA_PORT", "8000")
    yield


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch) -> Iterator[str]:
    """Switch CWD to a per-test temp directory and restore it afterwards."""
    original = os.getcwd()
    monkeypatch.chdir(tmp_path)
    try:
        yield str(tmp_path)
    finally:
        os.chdir(original)
