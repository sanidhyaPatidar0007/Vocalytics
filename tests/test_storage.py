"""Storage safety: path traversal, generated names, atomic writes, cleanup."""

from __future__ import annotations

import pytest

from core import storage
from core.errors import PermissionDeniedError, ValidationError


@pytest.mark.parametrize(
    "relative",
    [
        "../outside.txt",
        "orgs/1/../../../etc/passwd",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
        "orgs\\..\\..\\x",
        "",
        "x:stream",
    ],
)
def test_resolve_relative_refuses_traversal(relative):
    with pytest.raises(PermissionDeniedError):
        storage.resolve_relative(relative)


def test_resolve_relative_accepts_paths_inside_root(app_settings):
    path = storage.new_version_path(1, 2, 3)
    assert storage.resolve_relative(storage.to_relative(path)) == path.resolve()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("../../secret/report.csv", "report.csv"),
        ("..\\..\\boot.ini", "boot.ini"),
        ('bad<>:"|?*name.csv', "badname.csv"),
        ("   ", "upload"),
        ("tab\tname.csv", "tab name.csv"),
    ],
)
def test_sanitize_display_filename(name, expected):
    assert storage.sanitize_display_filename(name) == expected


def test_generated_names_never_collide():
    paths = {storage.new_original_path(1, 1, ".csv") for _ in range(200)}
    assert len(paths) == 200
    assert all(p.name.endswith(".csv") and len(p.stem) == 32 for p in paths)


def test_unsupported_extension_and_bad_ids_are_rejected():
    with pytest.raises(ValidationError):
        storage.new_original_path(1, 1, ".exe")
    with pytest.raises(ValidationError):
        storage.new_version_path(0, 1, 1)
    with pytest.raises(ValidationError):
        storage.org_dir(True)  # bool is not an organization id


def test_atomic_output_removes_partial_file_on_error(tmp_path):
    folder = tmp_path / "work"
    dest = folder / "final.bin"
    with pytest.raises(RuntimeError):
        with storage.atomic_output(dest) as tmp:
            tmp.write_bytes(b"partial")
            raise RuntimeError("crash mid-write")
    assert not dest.exists()
    assert list(folder.iterdir()) == []


def test_atomic_write_replaces_completely(tmp_path):
    folder = tmp_path / "work"
    dest = folder / "file.txt"
    storage.atomic_write_bytes(dest, b"one")
    storage.atomic_write_bytes(dest, b"two")
    assert dest.read_bytes() == b"two"
    assert [p.name for p in folder.iterdir()] == ["file.txt"]


def test_temporary_directory_is_removed_even_on_error(app_settings):
    with pytest.raises(ValueError):
        with storage.temporary_directory(1) as tmp:
            (tmp / "a.txt").write_text("x")
            created = tmp
            raise ValueError("fail")
    assert not created.exists()


def test_read_only_files_can_still_be_removed_by_cleanup(tmp_path):
    path = tmp_path / "ro.csv"
    path.write_text("a")
    storage.make_read_only(path)
    assert storage.remove_file_quietly(path)
    assert not path.exists()
