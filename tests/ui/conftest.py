"""Streamlit AppTest helpers.

``make_app`` starts the real ``app.py`` against the per-test database. A real
session token (from ``authentication.login``) is placed in server-side session
state, exactly where the sign-in form puts it, so page tests do not need to
drive the form each time (the form itself is tested in test_shell.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from auth import authentication
from components.session import TOKEN_KEY
from tests.conftest import DEFAULT_PASSWORD

ROOT = Path(__file__).resolve().parents[2]
APP_PATH = str(ROOT / "app.py")


@pytest.fixture
def make_app():
    def _make(username: str | None = None, *, page: str | None = None, timeout: int = 120) -> AppTest:
        app = AppTest.from_file(APP_PATH, default_timeout=timeout)
        if username:
            app.session_state[TOKEN_KEY] = authentication.login(username, DEFAULT_PASSWORD).token
        app.run()
        if page:
            app.switch_page(page).run()
        return app

    return _make


def texts(app: AppTest) -> str:
    """All visible text of the current page (titles, markdown, captions, alerts)."""
    parts = []
    for group in (
        app.title,
        app.header,
        app.subheader,
        app.markdown,
        app.caption,
        app.info,
        app.success,
        app.warning,
        app.error,
    ):
        parts.extend(str(element.value) for element in group)
    return "\n".join(parts)
