"""What the program says while it runs, kept for the UI's log panel: the lines Python prints to standard
output and standard error (the stages' warnings, a failed stage's traceback), from any thread, and the
UI's own messages. Printed lines still reach the terminal.

raylib's own messages are printed by C, past Python's streams, so they show only in the terminal.
"""

import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TextIO

MAX_LINES = 2000  # older lines are dropped


class Source(Enum):
    OUT = "out"  # standard output
    ERR = "err"  # standard error
    UI = "ui"  # the UI's news
    UI_ERROR = "ui error"


@dataclass(frozen=True, slots=True)
class Line:
    at: float  # time.time()
    source: Source
    text: str

    def __str__(self) -> str:
        return f"{time.strftime('%H:%M:%S', time.localtime(self.at))} {self.text}"


class Log:
    def __init__(self, max_lines: int = MAX_LINES):
        self._lines: deque[Line] = deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self.added = 0  # lines ever added, so a reader can tell what is new

    def add(self, source: Source, text: str) -> None:
        line = Line(time.time(), source, text.rstrip())
        with self._lock:
            self._lines.append(line)
            self.added += 1

    def lines(self) -> list[Line]:
        with self._lock:
            return list(self._lines)

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()

    def text(self) -> str:
        return "\n".join(map(str, self.lines()))


class Tee:
    """A text stream that writes through to `stream` (when there is one) and adds each whole line to `log`."""

    def __init__(self, stream: TextIO | None, log: Log, source: Source):
        self.stream, self.log, self.source = stream, log, source
        self._partial = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        if self.stream is not None:
            self.stream.write(text)
        with self._lock:
            *whole, self._partial = (self._partial + text).split("\n")
        for line in whole:
            if line.strip():
                self.log.add(self.source, line)
        return len(text)

    def flush(self) -> None:
        if self.stream is not None:
            self.stream.flush()

    def __getattr__(self, name: str):
        return getattr(self.stream, name)  # encoding, isatty, ... as the stream has them


@contextmanager
def capture(log: Log) -> Iterator[None]:
    """Copy what is printed to standard output and standard error into `log` while inside."""
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(out, log, Source.OUT), Tee(err, log, Source.ERR)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = out, err
