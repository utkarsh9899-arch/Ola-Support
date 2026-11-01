"""Tee stdout into transcripts/<name>.txt so every demo leaves graded evidence."""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import sys

import config


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def transcript(name: str):
    config.TRANSCRIPT_DIR.mkdir(exist_ok=True)
    path = config.TRANSCRIPT_DIR / f"{name}.txt"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Transcript: {name}\n")
        fh.write(f"# MOCK_LLM={int(config.MOCK_LLM)}  generated {dt.datetime.now().isoformat(timespec='seconds')}\n\n")
        tee = _Tee(sys.stdout, fh)
        with contextlib.redirect_stdout(tee):
            yield
    print(f"[transcript saved -> {path.relative_to(config.ROOT)}]")
