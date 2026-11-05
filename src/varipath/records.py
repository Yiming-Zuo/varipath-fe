from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch


ROLES = {"D-fit": 1, "D-select": 2, "P": 3, "I": 4, "F": 5, "R": 6}
METHODS = ("V", "V0", "G", "MG", "FEAT", "MNF-IC", "E-MBAR", "EB-MBAR")


def independent_seed(seed, system, role, replica=0, method=None, chain=0):
    if role not in ROLES or min(seed, replica, chain) < 0:
        raise ValueError("Invalid role or negative random-stream identifier")
    system_id = {"harmonic": 0, "double-well": 1, "ALDP": 2, "ALA4": 3}[system]
    method_id = 0 if method is None else METHODS.index(method) + 1
    return int(np.random.SeedSequence([seed, system_id, ROLES[role], replica,
                                      method_id, chain]).generate_state(1)[0])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Budget:
    def __init__(self, seconds, previous=None, *, research=False):
        if seconds is None:
            if not research:
                raise ValueError("Only an explicit research budget may omit its wall-clock cap")
            self.seconds = None
        else:
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float, np.integer, np.floating)) or not np.isfinite(seconds) or seconds <= 0:
                raise ValueError("Budget must be finite and positive")
            self.seconds = float(seconds)
        self.research = bool(research)
        self.previous = float((previous or {}).get("wall_seconds", 0))
        self.started = time.monotonic()
        self.cpu_started = time.process_time()
        self.phases = list((previous or {}).get("phases", []))
        self.counts = dict((previous or {}).get("counts", {}))

    @property
    def elapsed(self):
        return self.previous + time.monotonic() - self.started

    def check(self):
        if self.seconds is not None and self.elapsed >= self.seconds:
            raise TimeoutError("Cold-start budget exhausted; discard incomplete primary batch")

    def count(self, key, amount=1):
        self.counts[key] = self.counts.get(key, 0) + amount

    @contextmanager
    def phase(self, name):
        self.check()
        started, cpu = time.monotonic(), time.process_time()
        status = "complete"
        try:
            yield
            self.check()
        except BaseException:
            status = "failed"
            raise
        finally:
            self.phases.append(dict(name=name, status=status,
                                    wall_seconds=time.monotonic() - started,
                                    cpu_seconds=time.process_time() - cpu))

    def record(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return dict(wall_seconds=self.elapsed, cpu_seconds_current_process=time.process_time()
                    - self.cpu_started, phases=self.phases, counts=self.counts,
                    peak_rss_bytes=int(rss*(1 if sys.platform == "darwin" else 1024)),
                    budget_seconds=self.seconds, budget_kind="research" if self.research else "cold_start")


def check_pool_roles(*pools):

    seen = {}
    for pool in pools:
        role = pool["role"]
        if role not in ROLES:
            raise ValueError(f"Unknown pool role {role}")
        for unit in set(pool["ladder_ids"]):
            if unit in seen and seen[unit] != role:
                raise ValueError(f"Ladder {unit} leaks between {seen[unit]} and {role}")
            seen[unit] = role
