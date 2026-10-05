"""Measurement instrumentation.

`flatten_observation` turns a raw Observation into the flat numeric metric
dict every downstream component (hypothesis prediction, experiment delta,
information gain) reasons over. One metric name — one source of truth.

Component tracking is site-agnostic: every element carrying an `id` is
probed for presence and visibility (capped), so the same instrumentation
works for any website without configuration.
"""
from __future__ import annotations

from typing import Dict

MAX_TRACKED_COMPONENTS = 40

_MEASURE_JS = """
() => {
  const visible = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    const st = getComputedStyle(el);
    return (r.width > 0 && r.height > 0 &&
            st.visibility !== 'hidden' && st.display !== 'none') ? 1 : 0;
  };
  const q = (s) => document.querySelectorAll(s).length;
  const nav = performance.getEntriesByType('navigation')[0];
  const paints = performance.getEntriesByType('paint');
  const fcp = paints.find(p => p.name === 'first-contentful-paint');
  let lcp = null;
  try {
    const buf = performance.getEntriesByType('largest-contentful-paint');
    // LCP entries are only exposed via observer; try both
    lcp = buf.length ? buf[buf.length - 1].startTime : null;
  } catch (e) {}
  const out = {
    dom_nodes: document.getElementsByTagName('*').length,
    headings: q('h1,h2,h3,h4'),
    forms: q('form'),
    inputs: q('input,textarea,select'),
    buttons: q('button'),
    links: q('a'),
    images: q('img'),
    images_missing_alt:
      Array.from(document.querySelectorAll('img'))
        .filter(i => !i.hasAttribute('alt')).length,
    landmarks: q('header,nav,main,footer,aside'),
    scripts_in_dom: q('script'),
    fcp_ms: fcp ? fcp.startTime : null,
    domContentLoaded_ms: nav ? nav.domContentLoadedEventEnd : null,
    loadEvent_ms: nav ? nav.loadEventEnd : null,
    fonts_status_loaded: (document.fonts && document.fonts.status === 'loaded') ? 1 : 0,
    fonts_count: document.fonts ? document.fonts.size : null,
    doc_height_px: document.documentElement.scrollHeight,
    comp_present: {},
    comp_visible: {},
  };
  // site-agnostic component probe: every element with an id (capped)
  const withIds = Array.from(document.querySelectorAll('[id]'))
    .slice(0, __MAX_COMPONENTS__);
  for (const el of withIds) {
    out.comp_present[el.id] = 1;
    const v = visible(el);
    out.comp_visible[el.id] = (v === null) ? 0 : v;
  }
  return out;
}
"""


def measurement_js() -> str:
    return _MEASURE_JS.replace("__MAX_COMPONENTS__",
                               str(MAX_TRACKED_COMPONENTS))


def flatten_observation(obs) -> Dict[str, float]:
    """Observation -> flat {metric_name: number} used by all reasoning code."""
    m: Dict[str, float] = {}

    perf = obs.performance or {}
    for k in ("fcp_ms", "domContentLoaded_ms", "loadEvent_ms", "goto_ms"):
        v = perf.get(k)
        if isinstance(v, (int, float)):
            m[f"perf.{k}"] = float(v)

    dom = obs.dom or {}
    for k, v in dom.items():
        if k in ("comp_present", "comp_visible"):
            for cid, cv in (v or {}).items():
                m[f"dom.{k}.{cid.lstrip('_')}"] = float(cv or 0)
        elif isinstance(v, (int, float, bool)):
            m[f"dom.{k}"] = float(v)

    a11y = obs.a11y or {}
    for k, v in a11y.items():
        if isinstance(v, (int, float, bool)):
            m[f"a11y.{k}"] = float(v)

    resources = obs.resources or []
    m["res.count"] = float(len(resources))
    m["res.failed"] = float(sum(1 for r in resources
                                if r.error or (r.status or 0) >= 400))
    m["res.third_party"] = float(sum(1 for r in resources if r.third_party))
    m["res.scripts"] = float(sum(1 for r in resources if r.kind == "script"))
    m["res.stylesheets"] = float(sum(1 for r in resources if r.kind == "stylesheet"))
    m["res.fonts"] = float(sum(1 for r in resources if r.kind == "font"))
    m["res.images"] = float(sum(1 for r in resources if r.kind == "image"))
    m["res.total_bytes"] = float(sum(r.size_bytes for r in resources))
    # load order position of each resource (by observation append order)
    for i, r in enumerate(resources):
        m[f"seq.{r.basename}"] = float(i)

    console = obs.console or []
    m["err.console"] = float(sum(1 for c in console if c.get("type") == "error"))
    m["err.pageerror"] = float(sum(1 for c in console if c.get("type") == "pageerror"))

    cond = obs.conditions or {}
    vp = cond.get("viewport") or {}
    if "width" in vp:
        m["cond.viewport_width"] = float(vp["width"])

    return m


def load_order(obs) -> list:
    """Resource basenames in completion order — the observable 'load sequence'."""
    return [r.basename for r in (obs.resources or [])]
