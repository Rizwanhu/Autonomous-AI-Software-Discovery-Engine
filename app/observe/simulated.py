"""SimulatedSiteDriver — an in-memory website with *seeded ground-truth
causality*, used to validate the discovery engine offline.

The simulator implements a plausible TechAbout-like site: deferred JS modules,
a self-hosted Tailwind build, Google Fonts, a JS-injected chat widget, a
server-rendered newsletter form, responsive mobile menu. Interventions change
observable metrics through explicit causal rules below. Because ground truth
is known, unit tests can assert that the engine's *discovered* graph matches
the *true* graph — the falsifiability check for the engine itself.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from ..types import Observation, ResourceRecord, new_id
from .driver import THROTTLE_PROFILES

PAGES = {
    "/": dict(sections=12, images=18, forms=1, chat=True, careers_feed=False),
    "/services": dict(sections=10, images=9, forms=1, chat=True, careers_feed=False),
    "/portfolio": dict(sections=9, images=22, forms=0, chat=True, careers_feed=False),
    "/careers": dict(sections=8, images=6, forms=1, chat=True, careers_feed=True),
}

JS_MODULES = ["header-account.js", "forms.js", "cart.js", "chat.js"]
CSS_MODULES = ["tailwind.css"]
THIRD_PARTY = ["fonts.googleapis.com/css2", "fonts.gstatic.com/inter.woff2",
               "fonts.gstatic.com/main.woff2"]

# Ground truth: intervention -> metric -> (control, treatment) behaviour.
# This is what a correct discovery engine must recover.
GROUND_TRUTH = [
    {"cause": "block:tailwind.css", "effect": "dom.doc_height_px", "note": "layout collapse"},
    {"cause": "block:chat.js", "effect": "dom.comp_visible.ta-chat", "note": "widget injected by JS"},
    {"cause": "block:header-account.js", "effect": "dom.comp_visible.mobile-menu",
     "note": "menu wired by module"},
    {"cause": "block:forms.js", "effect": "dom.forms_js_hooks",
     "note": "validation hooks registered by module"},
    {"cause": "block:css2", "effect": "dom.fonts_status_loaded",
     "note": "third-party font CSS"},
    {"cause": "throttle:slow3g", "effect": "perf.goto_ms", "note": "latency slows paint"},
    {"cause": "cond.viewport_width", "effect": "dom.comp_visible.mobile-menu",
     "note": "responsive breakpoint at 768px"},
    {"cause": "delay:chat.js", "effect": "perf.domContentLoaded_ms",
     "note": "deferred script delays DCL"},
    {"cause": "js_disabled", "effect": "dom.comp_present.ta-chat",
     "note": "no JS, no injected widget"},
]


def export_ground_truth(path: Path) -> None:
    path.write_text(json.dumps(GROUND_TRUTH, indent=2), encoding="utf-8")


class SimulatedSiteDriver:
    """Deterministic, noisy simulator implementing the same driver contract
    as PlaywrightDriver."""

    def __init__(self, base_url: str = "https://sim.invalid", seed: int = 7):
        self.base_url = base_url
        self.rng = random.Random(seed)
        self._calls = 0

    def start(self):  # API compat
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    # ------------------------------------------------------------------
    def load(self, page: str, conditions: dict, interventions: list | None = None,
             screenshot_path: Optional[str] = None) -> Observation:
        interventions = interventions or []
        conditions = dict(conditions or {})
        self._calls += 1
        rng = random.Random(self._calls)
        page_spec = PAGES.get(page, PAGES["/"])
        blocked = {s["pattern"] for s in interventions if s.get("type") == "block"}
        delayed = {s["pattern"]: s.get("delay_ms", 2000)
                   for s in interventions if s.get("type") == "delay"}
        js_disabled = conditions.get("js_disabled", False)
        throttle = conditions.get("throttle", "none")
        vp_w = (conditions.get("viewport") or {}).get("width", 1440)
        t_mult = 6.0 if throttle == "slow3g" else 1.0

        def noisy(v, pct=0.03):
            return v * (1.0 + rng.uniform(-pct, pct))

        # ---- resources ---------------------------------------------------
        resources: list[ResourceRecord] = []
        order = ["tailwind.css", "header-account.js", "forms.js", "cart.js",
                 "chat.js", "fonts.googleapis.com/css2",
                 "fonts.gstatic.com/inter.woff2",
                 "fonts.gstatic.com/main.woff2"] + \
                [f"img-{i}.webp" for i in range(page_spec["images"])]
        sizes = {"tailwind.css": 48000, "header-account.js": 12000,
                 "forms.js": 9000, "cart.js": 7000, "chat.js": 15000,
                 "fonts.googleapis.com/css2": 2000,
                 "fonts.gstatic.com/inter.woff2": 28000,
                 "fonts.gstatic.com/main.woff2": 31000}

        t_cursor = 80.0
        for name in order:
            is_res = name in sizes
            if any(name.startswith(b) or b in name for b in blocked):
                resources.append(ResourceRecord(
                    url=f"{self.base_url}/assets/{name}", kind=self._kind(name),
                    status=None, size_bytes=0, duration_ms=None,
                    third_party=("fonts" in name), error="net::ERR_FAILED"))
                continue
            if is_res:
                dur = noisy(sizes[name] / 12000.0 * 900.0 * t_mult)
            else:  # images load fast and in parallel in real browsers
                dur = noisy((150.0 + 100.0) * t_mult)
            if name in delayed:
                dur += delayed[name]
            t_cursor += dur
            resources.append(ResourceRecord(
                url=f"{self.base_url}/assets/{name}", kind=self._kind(name),
                status=200, size_bytes=sizes.get(name, 30000 + rng.randint(0, 9000)),
                duration_ms=round(dur, 1), third_party=("fonts" in name)))

        def has(name: str) -> bool:
            return not any(name.startswith(b) or b in name for b in blocked)

        # ---- DOM ---------------------------------------------------------
        dom_nodes = 980 + page_spec["sections"] * 140 + page_spec["images"] * 4
        css_ok = has("tailwind.css")
        doc_height = noisy(2600 + page_spec["sections"] * 180 if css_ok else 940)

        comp_present = {
            "main-content": 1, "newsletter-form": page_spec["forms"],
            "footer-newsletter-email": page_spec["forms"],
            "ta-cookie": 1, "video-modal": 1, "ta-chat-panel": 0,
            "ta-chat-form": 0, "ta-chat": 0, "mobile-menu": 1,
        }
        comp_visible = {k: 0 for k in comp_present}
        comp_visible["main-content"] = 1
        comp_visible["ta-cookie"] = 1 if css_ok else 0
        fonts_css_ok = has("fonts.googleapis.com/css2")
        if not js_disabled:
            # chat widget injected by chat.js once it loads
            comp_present["ta-chat"] = int(has("chat.js") and page_spec["chat"])
            comp_present["ta-chat-panel"] = comp_present["ta-chat"]
            comp_present["ta-chat-form"] = comp_present["ta-chat"]
            comp_visible["ta-chat"] = comp_present["ta-chat"]
            # mobile menu wired + visible below 768px (requires CSS + module)
            comp_visible["mobile-menu"] = int(
                vp_w < 768 and css_ok and has("header-account.js"))
            comp_visible["newsletter-form"] = page_spec["forms"]
        forms_hooks = int(has("forms.js") and not js_disabled) * \
            (2 * page_spec["forms"])

        dom = {
            "dom_nodes": int(noisy(dom_nodes if not js_disabled else dom_nodes * 0.97)),
            "headings": 14 + page_spec["sections"],
            "forms": page_spec["forms"],
            "inputs": 3 + 2 * page_spec["forms"],
            "buttons": 18,
            "links": 60 + page_spec["sections"] * 3,
            "images": page_spec["images"],
            "images_missing_alt": 2 if page_spec["images"] > 8 else 0,
            "landmarks": 5,
            "scripts_in_dom": 4,
            "doc_height_px": round(doc_height, 1),
            "fonts_status_loaded": int(fonts_css_ok and not js_disabled),
            "fonts_count": 2 if fonts_css_ok else 0,
            "forms_js_hooks": forms_hooks,
            "comp_present": comp_present,
            "comp_visible": comp_visible,
        }

        # ---- performance -------------------------------------------------
        total_res_ms = sum(r.duration_ms or 0 for r in resources
                           if not r.error)
        goto = noisy(max(1800.0, total_res_ms * 0.35))
        perf = {"goto_ms": round(goto, 1)}
        if not js_disabled:
            perf["fcp_ms"] = round(noisy(goto * 0.55), 1)
            perf["lcp_ms"] = round(noisy(goto * 0.82), 1)
            dcl = goto * 0.9
            if "chat.js" in delayed and has("chat.js"):
                dcl += delayed["chat.js"] * 0.9
            perf["domContentLoaded_ms"] = round(noisy(dcl), 1)
            perf["loadEvent_ms"] = round(noisy(goto), 1)

        # ---- console ------------------------------------------------------
        console = []
        if "chat.js" not in blocked and not js_disabled:
            pass  # clean site
        if any(b in "header-account.js" for b in blocked) and not js_disabled:
            console.append({"type": "error",
                            "text": "Uncaught ReferenceError: TAHeader not defined"})
        if not css_ok:
            console.append({"type": "warning",
                            "text": "rendered without stylesheet"})

        return Observation(
            id=new_id("obs"), page=page,
            conditions={"label": conditions.get("label", ""), **conditions},
            resources=resources, dom=dom, console=console, performance=perf,
            a11y={"images_missing_alt": dom["images_missing_alt"],
                  "headings": dom["headings"], "landmarks": dom["landmarks"]},
            screenshot=None,
        )

    @staticmethod
    def _kind(name: str) -> str:
        if name.endswith(".js"):
            return "script"
        if name.endswith(".css") or "css2" in name:
            return "stylesheet"
        if ".woff2" in name:
            return "font"
        return "image"
