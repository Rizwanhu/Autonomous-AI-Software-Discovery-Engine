"""Budget & rate limiting — the safety cage every page load must pass through."""
from __future__ import annotations

import threading
import time

from ..config import SafetyConfig


class BudgetExhausted(Exception):
    """Raised when the run hits a hard safety limit. Always catch & wrap up."""


class BudgetGuard:
    def __init__(self, config: SafetyConfig):
        self.cfg = config
        self._lock = threading.Lock()
        self._last_load = 0.0
        self.loads = 0
        self.started = time.time()

    def spend_load(self) -> None:
        with self._lock:
            elapsed_min = (time.time() - self.started) / 60.0
            if elapsed_min > self.cfg.max_wall_time_min:
                raise BudgetExhausted(
                    f"wall-time budget exhausted ({elapsed_min:.1f} min)")
            if self.loads >= self.cfg.max_page_loads:
                raise BudgetExhausted(
                    f"page-load budget exhausted ({self.loads} loads)")
            # polite spacing between navigations
            wait = self._last_load + self.cfg.min_interval_between_loads_s - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_load = time.time()
            self.loads += 1

    def remaining(self) -> int:
        return max(0, self.cfg.max_page_loads - self.loads)

    def elapsed_min(self) -> float:
        return (time.time() - self.started) / 60.0

    def summary(self) -> dict:
        return {"page_loads": self.loads,
                "elapsed_min": round(self.elapsed_min(), 1),
                "remaining": self.remaining()}
