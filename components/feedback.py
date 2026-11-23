"""Uniform user feedback: safe error display, progress, confirmations, headers."""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from typing import TypeVar

import pandas as pd
import streamlit as st

from components.session import current_ctx
from core.errors import AppError
from services.audit_service import record_error

T = TypeVar("T")

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")


def md_escape(text: object) -> str:
    """Escape Markdown so dataset/user text renders literally.

    Streamlit renders Markdown in st.markdown, st.write, st.caption, st.info,
    st.error and similar calls. Uploaded data is untrusted: without escaping, a
    value such as ![x](http://example.com/p.png) would load a remote image and
    leak data. Use this (or st.text / st.dataframe) for any dynamic text.
    """
    value = "" if text is None else str(text)
    return _MD_SPECIAL.sub(r"\\\1", value.replace("\r", " ").replace("\n", " "))


def run_action(action: Callable[[], T], *, success: str | None = None, component: str = "ui") -> T | None:
    """Run a service call and show the outcome.

    Expected failures (AppError) show their user-safe message. Anything else is
    logged with a reference number and the user sees a generic message, never a
    stack trace, file path or SQL.
    """
    try:
        result = action()
    except AppError as exc:
        st.error(md_escape(exc.user_message), icon=":material/error:")
        return None
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the UI
        reference = record_error(component, exc, ctx=current_ctx())
        st.error(
            f"Something went wrong and nothing was changed. Reference: {reference}",
            icon=":material/report:",
        )
        return None
    if success:
        st.success(success, icon=":material/check_circle:")
    return result


def page_header(title: str, description: str | None = None) -> None:
    st.title(title)
    if description:
        st.caption(description)


class Progress:
    """Adapter from service progress callbacks to a Streamlit progress bar."""

    def __init__(self, label: str) -> None:
        self._bar = st.progress(0.0, text=label)

    def __call__(self, fraction: float, message: str) -> None:
        self._bar.progress(min(max(fraction, 0.0), 1.0), text=message or None)

    def done(self) -> None:
        self._bar.empty()


def confirm(label: str, key: str) -> bool:
    """An explicit confirmation checkbox for destructive actions."""
    return st.checkbox(label, key=key, value=False)


def show_table(
    df: pd.DataFrame | None, *, height: int | None = None, empty_message: str = "Nothing to show."
) -> None:
    if df is None or df.empty:
        st.info(empty_message)
        return
    kwargs: dict[str, object] = {"width": "stretch", "hide_index": True}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)


def escape(text: object) -> str:
    """HTML-escape any dynamic text before it is placed inside custom HTML."""
    return html.escape("" if text is None else str(text), quote=True)


def feature_unavailable(message: str) -> None:
    st.info(message, icon=":material/lock:")
