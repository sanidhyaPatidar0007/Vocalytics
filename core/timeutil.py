"""Time helpers. All timestamps are stored as naive UTC datetimes.

Code calls ``timeutil.utcnow()`` through the module (not ``from ... import``)
so tests can monkeypatch the clock in one place.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def month_start(moment: datetime) -> datetime:
    return moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
