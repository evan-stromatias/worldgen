import sys
from pathlib import Path

import pytest

from worldgen.cave.__main__ import main
from worldgen.cave.export import (
    ARRAYS_FILE,
    MANIFEST_FILE,
    PREVIEW_FILE,
    RECIPE_FILE,
)


def test_cli_generate_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "cli_npz"
    monkeypatch.setattr(
        sys,
        "argv",
        ["cavegen", "--seed", "7", "--out", str(out)],
    )
    main()

    assert (out / RECIPE_FILE).exists()
    assert (out / ARRAYS_FILE).exists()
    assert (out / PREVIEW_FILE).exists()


def test_cli_generate_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "cli_raw"
    monkeypatch.setattr(
        sys,
        "argv",
        ["cavegen", "--seed", "11", "--format", "raw", "--out", str(out)],
    )
    main()

    assert (out / MANIFEST_FILE).exists()
    assert (out / RECIPE_FILE).exists()
    assert (out / PREVIEW_FILE).exists()


def test_cli_preview_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "cli_prev"
    monkeypatch.setattr(
        sys,
        "argv",
        ["cavegen", "--seed", "3", "--preview-only", "--out", str(out)],
    )
    main()

    assert (out / PREVIEW_FILE).exists()
    assert not (out / ARRAYS_FILE).exists()


def test_cli_negative_seed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["cavegen", "--seed", "-1"])
    with pytest.raises(SystemExit):
        main()
