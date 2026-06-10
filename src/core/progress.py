"""Default progress reporter for the cache build."""
from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

from src.core.pipeline import ProgressCb


def make_progress_reporter(
    min_count: int, batch_size: int
) -> Tuple[ProgressCb, float]:
    """Return ``(callback, t0)``: a progress reporter that prints rows/s.

    The callback is invoked after each batch with
    ``(rows_written, last_count_all)``. *t0* is the start time so the caller
    can print a final timing line.
    """
    print(f"Streaming rows with count_all >= {min_count:,} in batches of {batch_size:,} ...")
    t0 = time.time()
    last_report = t0

    def _progress(n: int, last_count: Optional[int]) -> None:
        nonlocal last_report
        now = time.time()
        if now - last_report >= 5.0 or n == batch_size:
            rate = n / max(now - t0, 1e-3)
            print(f"  {n:,} rows written  ({rate:,.0f} rows/s, last count_all={last_count})")
            last_report = now

    return _progress, t0
