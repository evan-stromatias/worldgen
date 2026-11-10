"""The terrain generator UI's log: its messages and each stage's time, and the Log button's news."""

import copy

import yaml

from common import SEED, SMALL, inline_executor
from worldgen.ui.controller import Controller, log_news
from worldgen.ui.model.log import Log, Source
from worldgen.ui.model.recipe.recipe import Recipe
from worldgen.ui.view import events as ev


def texts(log: Log) -> list[tuple[Source, str]]:
    return [(line.source, line.text) for line in log.lines()]


def test_the_ui_logs_its_messages_and_each_stage(tmp_path):
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(copy.deepcopy(SMALL)))
    c = Controller(Recipe.load(path), SEED, tmp_path / "terrain", executor=inline_executor())
    c.act(ev.RunAll())
    c.update()
    logged = texts(c.log)
    assert logged[0] == (Source.UI, "running elevation, erosion, hydrology, climate, biomes")
    assert any(text.startswith("  hydrology") and text.endswith("s") for _, text in logged)
    assert (Source.UI, "  erosion    skipped: erosion off") in logged
    assert logged[-1][1].startswith("ran elevation")
    c.act(ev.SeedCommitted("x"))
    assert c.log.lines()[-1].source is Source.UI_ERROR
    c.act(ev.ClearLog())
    assert c.log.lines() == []


def test_the_log_button_says_what_came_while_it_was_not_seen():
    log = Log()
    assert log_news(log, 0) == ("", False)
    log.add(Source.OUT, "a")
    log.add(Source.UI, "b")
    assert log_news(log, 0) == ("2 new", False)
    log.add(Source.ERR, "Traceback")
    assert log_news(log, 0) == ("1 new error", True)
    assert log_news(log, log.added) == ("", False)  # seen
