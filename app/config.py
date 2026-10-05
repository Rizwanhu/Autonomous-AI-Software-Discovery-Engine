"""Target, safety and budget configuration.

The engine is read-only toward the server. All interventions (blocking,
delaying, mocking) are executed inside our own browser context via request
routing, so the origin never observes anything but ordinary page loads.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

BASE_URL = "https://www.techabout.com"

# Pages investigated by default (the brand surfaces named in the brief).
DEFAULT_PAGES = [
    "/",
    "/services",
    "/portfolio",
    "/careers",
]

# Third-party origins observed on the site (used for labelling factors).
THIRD_PARTY_DOMAINS = [
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "googleapis.com",
    "gstatic.com",
    "google-analytics.com",
    "googletagmanager.com",
    "cloudflare.com",
    "jsdelivr.net",
    "unpkg.com",
]


@dataclass
class SafetyConfig:
    """Hard limits that every component must respect."""
    min_interval_between_loads_s: float = 1.5    # be polite to the origin
    max_page_loads: int = 140                    # total budget of navigations
    max_wall_time_min: float = 45.0              # overall time budget
    allowed_methods: tuple = ("GET", "HEAD")     # never POST/PUT/DELETE at the origin
    max_pages_per_run: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LabConfig:
    base_url: str = BASE_URL
    pages: list = field(default_factory=lambda: list(DEFAULT_PAGES))
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    repeats_per_arm: int = 2          # stochastic robustness for each arm
    support_threshold: float = 0.85   # posterior needed before review board
    refute_threshold: float = 0.08    # posterior below which a hypothesis dies
    noise_tolerance: float = 0.90     # P(prediction holds | hypothesis true)
    max_hypotheses_active: int = 14
    discovery_repeats: int = 2        # independent verifier re-runs

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "LabConfig":
        d = json.loads(path.read_text(encoding="utf-8"))
        safety = SafetyConfig(**d.pop("safety", {}))
        return LabConfig(safety=safety, **d)


# The default observational conditions under which pages are passively
# measured. Each entry yields one Observation per page. Comparing across
# these conditions is what lets the engine DISCOVER which components are
# responsive (desktop vs mobile) and which are JS-injected (js on vs off)
# instead of hardcoding site-specific assumptions.
OBSERVATIONAL_CONDITIONS = [
    {"viewport": {"width": 1440, "height": 900}, "throttle": "none",
     "label": "desktop-broadband"},
    {"viewport": {"width": 390, "height": 844}, "throttle": "none",
     "label": "mobile-broadband"},
    {"viewport": {"width": 1440, "height": 900}, "throttle": "slow3g",
     "label": "desktop-slow3g"},
    {"viewport": {"width": 1440, "height": 900}, "throttle": "none",
     "js_disabled": True, "label": "desktop-nojs"},
]
