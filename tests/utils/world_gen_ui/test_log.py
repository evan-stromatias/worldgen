"""The log's model: printed lines captured from any thread, kept up to a limit."""

import sys
import threading

from worldgen.utils.world_gen_ui.model.log import Log, Source, capture


def texts(log: Log) -> list[tuple[Source, str]]:
    return [(line.source, line.text) for line in log.lines()]


def test_printed_lines_are_kept_whole_and_still_printed(capsys):
    log = Log()
    before = sys.stdout, sys.stderr
    with capture(log):
        print("one", end="")
        print(" line\n\ntwo")
        print("oops", file=sys.stderr)
        worker = threading.Thread(target=print, args=("from a worker",))
        worker.start()
        worker.join()
    assert (sys.stdout, sys.stderr) == before
    assert texts(log) == [(Source.OUT, "one line"), (Source.OUT, "two"), (Source.ERR, "oops"),
                          (Source.OUT, "from a worker")]
    assert capsys.readouterr().out == "one line\n\ntwo\nfrom a worker\n"  # the terminal still gets it all


def test_old_lines_are_dropped_and_clear_empties_it():
    log = Log(max_lines=3)
    for i in range(5):
        log.add(Source.OUT, str(i))
    assert [line.text for line in log.lines()] == ["2", "3", "4"] and log.added == 5
    log.clear()
    assert log.lines() == [] and log.added == 5
