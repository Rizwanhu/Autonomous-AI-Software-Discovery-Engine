"""PlaywrightDriver — the real-browser observation & intervention instrument.

All manipulations happen *inside this browser context* via request routing,
CDP emulation and context options. The origin server only ever sees ordinary
GET page loads, rate-limited by the BudgetGuard.
"""
from __future__ import annotations

import html.parser
import time
from pathlib import Path
from typing import Optional, Protocol

from ..config import THIRD_PARTY_DOMAINS
from ..types import Observation, ResourceRecord, new_id
from .instrument import measurement_js
from .budget import BudgetGuard

THROTTLE_PROFILES = {
    # name -> (latency_ms, download B/s, upload B/s)
    "none": (0, 0, 0),
    "slow3g": (400, 50 * 1024, 20 * 1024),
    "fast3g": (150, 180 * 1024, 84 * 1024),
}

_LCP_INIT = """
window.__lcp = null;
window.__cls = 0;
try {
  new PerformanceObserver((l) => {
    const es = l.getEntries();
    if (es.length) window.__lcp = es[es.length - 1].startTime;
  }).observe({type: 'largest-contentful-paint', buffered: true});
} catch (e) {}
"""

_MEASURE_WITH_LCP = """
() => {
  const base = (%BASE%)();
  base.lcp_ms = window.__lcp;
  return base;
}
"""


class SiteDriver(Protocol):
    """Anything that can load a page under conditions+interventions and
    return a measured Observation. Implementations: real browser, simulator."""

    def load(self, page: str, conditions: dict, interventions: list | None,
             screenshot_path: Optional[str] = None) -> Observation: ...


class _HTMLMetricParser(html.parser.HTMLParser):
    """Fallback DOM measurement when JavaScript is disabled."""

    def __init__(self):
        super().__init__()
        self.counts = {"dom_nodes": 0, "headings": 0, "forms": 0, "inputs": 0,
                       "buttons": 0, "links": 0, "images": 0,
                       "images_missing_alt": 0, "landmarks": 0,
                       "scripts_in_dom": 0}
        self.ids: dict[str, int] = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.counts["dom_nodes"] += 1
        if tag in ("h1", "h2", "h3", "h4"):
            self.counts["headings"] += 1
        elif tag == "form":
            self.counts["forms"] += 1
        elif tag in ("input", "textarea", "select"):
            self.counts["inputs"] += 1
        elif tag == "button":
            self.counts["buttons"] += 1
        elif tag == "a":
            self.counts["links"] += 1
        elif tag == "img":
            self.counts["images"] += 1
            if "alt" not in a:
                self.counts["images_missing_alt"] += 1
        elif tag in ("header", "nav", "main", "footer", "aside"):
            self.counts["landmarks"] += 1
        elif tag == "script":
            self.counts["scripts_in_dom"] += 1
        if "id" in a:
            self.ids[a["id"]] = 1


def classify_resource(url: str, resource_type: str | None) -> str:
    t = (resource_type or "").lower()
    if t in ("script", "stylesheet", "font", "image", "document"):
        return t if t != "stylesheet" else "stylesheet"
    lower = url.lower().split("?")[0]
    if lower.endswith(".js") or lower.endswith(".mjs"):
        return "script"
    if lower.endswith(".css"):
        return "stylesheet"
    if lower.endswith((".woff", ".woff2", ".ttf", ".otf")):
        return "font"
    if lower.endswith((".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif",
                       ".ico", ".avif")):
        return "image"
    return t if t else "other"


def is_third_party(url: str, base_url: str) -> bool:
    from urllib.parse import urlparse
    host = urlparse(url).netloc.lower()
    origin = urlparse(base_url).netloc.lower()
    if host.endswith(origin) or origin.endswith(host):
        return False
    return any(d in host for d in THIRD_PARTY_DOMAINS) or bool(host)


class PlaywrightDriver:
    """Real Chromium instrumentation of the live site."""

    def __init__(self, base_url: str, budget: BudgetGuard, headless: bool = True):
        self.base_url = base_url.rstrip("/")
        self.budget = budget
        self.headless = headless
        self._pw = None
        self._browser = None
        self._shared_contexts: dict[str, object] = {}
        self.page_load_errors = 0

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)

    def close(self):
        try:
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        except Exception:
            pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.close()

    # -- main entry ---------------------------------------------------------
    def load(self, page: str, conditions: dict, interventions: list | None = None,
             screenshot_path: Optional[str] = None) -> Observation:
        interventions = interventions or []
        url = self.base_url + ("/" + page.lstrip("/") if page != "/" else "/")
        conditions = dict(conditions or {})
        js_enabled = not conditions.get("js_disabled", False)

        self.budget.spend_load()

        ctx_kwargs = {
            "viewport": conditions.get("viewport") or {"width": 1440, "height": 900},
            "java_script_enabled": js_enabled,
            "service_workers": "block",
        }
        vp = ctx_kwargs["viewport"]
        if vp.get("is_mobile"):
            ctx_kwargs["is_mobile"] = True
            ctx_kwargs["has_touch"] = True

        session = conditions.get("session", "fresh")
        if session == "shared":
            key = repr(sorted(conditions.items()))
            context = self._shared_contexts.get(key)
            if context is None:
                context = self._browser.new_context(**ctx_kwargs)
                self._shared_contexts[key] = context
        else:
            context = self._browser.new_context(**ctx_kwargs)

        pg = context.new_page()
        resources: list[ResourceRecord] = []
        console: list[dict] = []
        started = {"goto": None}
        req_t0: dict = {}

        def on_request(req):
            req_t0[req] = time.time()

        def on_response(resp):
            try:
                body_len = 0
                try:
                    hdr = resp.headers.get("content-length")
                    body_len = int(hdr) if hdr else 0
                except Exception:
                    pass
                dur = None
                if resp.request in req_t0:
                    dur = round((time.time() - req_t0[resp.request]) * 1000, 1)
                rt = resp.request.resource_type
                resources.append(ResourceRecord(
                    url=resp.url, kind=classify_resource(resp.url, rt),
                    status=resp.status, size_bytes=body_len,
                    duration_ms=dur,
                    third_party=is_third_party(resp.url, self.base_url)))
            except Exception:
                pass

        def on_failed(req):
            try:
                resources.append(ResourceRecord(
                    url=req.url, kind=classify_resource(req.url, req.resource_type),
                    status=None, size_bytes=0, duration_ms=None,
                    third_party=is_third_party(req.url, self.base_url),
                    error=req.failure or "failed"))
            except Exception:
                pass

        def on_console(msg):
            entry = {"type": msg.type, "text": msg.text[:400]}
            console.append(entry)

        def on_pageerror(err):
            console.append({"type": "pageerror", "text": str(err)[:400]})

        pg.on("request", on_request)
        pg.on("response", on_response)
        pg.on("requestfailed", on_failed)
        pg.on("console", on_console)
        pg.on("pageerror", on_pageerror)

        # -- interventions via request routing --------------------------------
        def route_handler(route):
            url = route.request.url
            for step in interventions:
                t = step.get("type")
                if t == "block" and step["pattern"] in url:
                    route.abort()
                    return
                if t == "mock_api" and step["pattern"] in url:
                    route.fulfill(status=step.get("status", 200),
                                  body=step.get("body", ""),
                                  content_type=step.get("content_type",
                                                        "application/json"))
                    return
                if t == "delay" and step["pattern"] in url:
                    time.sleep(step.get("delay_ms", 2000) / 1000.0)
                    break
            route.continue_()

        pg.route("**/*", route_handler)

        if js_enabled:
            pg.add_init_script(_LCP_INIT)

        # network throttling via CDP
        throttle = conditions.get("throttle", "none")
        cdp = None
        if throttle in THROTTLE_PROFILES and throttle != "none":
            latency, down, up = THROTTLE_PROFILES[throttle]
            cdp = context.new_cdp_session(pg)
            cdp.send("Network.enable")
            cdp.send("Network.emulateNetworkConditions",
                     {"offline": False, "latency": latency,
                      "downloadThroughput": down, "uploadThroughput": up})

        t0 = time.time()
        nav_error = None
        try:
            pg.goto(url, timeout=45000, wait_until="load")
        except Exception as e:
            nav_error = str(e)[:200]
            self.page_load_errors += 1
        goto_ms = (time.time() - t0) * 1000.0

        # let deferred scripts / fonts settle
        try:
            pg.wait_for_load_state("networkidle", timeout=6000)
        except Exception:
            pass
        time.sleep(1.0)

        dom: dict = {}
        perf: dict = {"goto_ms": round(goto_ms, 1)}
        if js_enabled:
            try:
                js = _MEASURE_WITH_LCP.replace("%BASE%", measurement_js())
                data = pg.evaluate(js)
                comp_p = data.pop("comp_present", {}) or {}
                comp_v = data.pop("comp_visible", {}) or {}
                dom = {k: v for k, v in data.items() if v is not None}
                dom["comp_present"] = comp_p
                dom["comp_visible"] = comp_v
                for k in ("fcp_ms", "domContentLoaded_ms", "loadEvent_ms", "lcp_ms"):
                    if data.get(k) is not None:
                        perf[k] = round(float(data[k]), 1)
            except Exception as e:
                dom = {"measure_error": 1}
                console.append({"type": "instrumentation", "text": str(e)[:200]})
        else:
            try:
                raw = pg.content()
                parser = _HTMLMetricParser()
                parser.feed(raw)
                dom = dict(parser.counts)
                # convention: with JS disabled an element that is absent is
                # definitionally not visible; any element found in the raw
                # HTML is marked visible since CSS state cannot be evaluated
                dom["comp_present"] = parser.ids
                dom["comp_visible"] = dict(parser.ids)
            except Exception as e:
                dom = {"measure_error": 1}
                console.append({"type": "instrumentation", "text": str(e)[:200]})

        screenshot = None
        if screenshot_path:
            try:
                Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
                pg.screenshot(path=screenshot_path, full_page=False)
                screenshot = str(screenshot_path)
            except Exception:
                pass

        if nav_error:
            console.append({"type": "navigation", "text": nav_error})

        try:
            if cdp:
                cdp.detach()
        except Exception:
            pass
        if session != "shared":
            context.close()

        a11y = {"images_missing_alt": dom.get("images_missing_alt", 0),
                "headings": dom.get("headings", 0),
                "landmarks": dom.get("landmarks", 0)}

        return Observation(
            id=new_id("obs"),
            page=page,
            conditions={"label": conditions.get("label", ""), **conditions},
            resources=resources,
            dom=dom,
            console=console,
            performance=perf,
            a11y=a11y,
            screenshot=screenshot,
        )
