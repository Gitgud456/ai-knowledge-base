"""Shared pytest fixtures.

Forces a clean settings load so per-test path overrides don't leak across tests.
"""

from __future__ import annotations

import pytest

from akb.config import reset_settings_cache


@pytest.fixture(autouse=True)
def _clean_settings() -> None:
    reset_settings_cache()
