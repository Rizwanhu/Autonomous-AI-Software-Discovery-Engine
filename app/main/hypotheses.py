"""Automated hypothesis generation (req. 4, 8) and revision (req. 1, 9).

Three generative channels:

1. ANOMALY     — something observable looks broken or surprising (failed
                resource, console error, missing component, perf cliff).
                Each anomaly spawns a COMPETING SET of explanations which
                experiments must discriminate between (req. 6).
2. CORRELATION — co-variation mined across observational conditions. These
                are recorded as CORRELATION edges only; turning them into
                causal claims requires an intervention.
3. SCREENING   — systematic, assumption-free probes of plausible
                cause/effect pairs among observed factors. Screening
                hypotheses start with LOW priors: nothing is assumed to
                exist until an intervention says so.

Every hypothesis is emitted with concrete predictions and a falsification
criterion — a hypothesis we cannot currently falsify is not emitted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..types import (FactorKind, Hypothesis, HypStatus, Observation,
                     Prediction, new_id)
from ..observe.instrument import flatten_observation, load_order

# ---------------------------------------------------------------------------
# Factor naming helpers — canonical keys across the whole engine
# ---------------------------------------------------------------------------


def res_factor(basename: str, third_party: bool) -> tuple[str, FactorKind, str]:
    if basename.endswith(".js"):
        return f"res:{basename}", FactorKind.JS_MODULE, basename
    if basename.endswith(".css"):
        return f"res:{basename}", FactorKind.CSS_RESOURCE, basename
    if third_party or "fonts" in basename or "css2" in basename:
        return f"res:{basename}", FactorKind.THIRD_PARTY, basename
    return f"res:{basename}", FactorKind.NETWORK_REQUEST, basename


def metric_factor(metric: str) -> tuple[str, FactorKind]:
    if metric.startswith("perf."):
        return metric, FactorKind.PERF_METRIC
    if metric.startswith("err."):
        return metric, FactorKind.RUNTIME_ERROR
    if metric.startswith("a11y."):
        return metric, FactorKind.A11Y
    if metric.startswith("dom.comp"):
        return metric, FactorKind.VISUAL_STATE
    if metric.startswith("dom."):
        return metric, FactorKind.DOM_COMPONENT
    if metric.startswith("res.") or metric.startswith("seq."):
        return metric, FactorKind.NETWORK_REQUEST
    return metric, FactorKind.DOM_COMPONENT


def condition_factor(name: str) -> tuple[str, FactorKind]:
    return f"cond:{name}", FactorKind.BROWSER_CONDITION


# ---------------------------------------------------------------------------
# Anomaly detection over passive observations
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    kind: str                 # failed_resource | console_error | missing_component |
                              # perf_cliff | font_not_loaded | a11y_gap
    page: str
    detail: str
    conditions: dict
    suspects: list = field(default_factory=list)   # factor keys

    def describe(self) -> str:
        return f"[{self.kind}] page={self.page} cond={self.conditions.get('label','')} {self.detail}"


def detect_anomalies(observations: list[Observation]) -> list[Anomaly]:
    out: list[Anomaly] = []

    # baseline = best case seen per page (no throttle)
    by_page: dict[str, list[Observation]] = {}
    for o in observations:
        by_page.setdefault(o.page, []).append(o)

    for o in observations:
        m = flatten_observation(o)

        for r in o.resources:
            if r.error or (r.status or 0) >= 400:
                key, _, _ = res_factor(r.basename, r.third_party)
                out.append(Anomaly(
                    "failed_resource", o.page,
                    f"resource failed: {r.basename} "
                    f"(status={r.status} error={r.error})",
                    o.conditions, suspects=[key]))

        if m.get("err.pageerror", 0) > 0:
            out.append(Anomaly(
                "console_error", o.page,
                f"{int(m['err.pageerror'])} runtime JS error(s): "
                + "; ".join(c["text"][:80] for c in o.console
                            if c["type"] == "pageerror")[:200],
                o.conditions))

        if m.get("dom.fonts_status_loaded", 1) == 0 and m.get("res.count", 0) > 3:
            out.append(Anomaly(
                "font_not_loaded", o.page,
                "document.fonts not loaded though page has resources",
                o.conditions, suspects=["res:css2", "res:fonts"]))

        # perf cliff: same page >3.5x slower than the fastest condition
        peers = [flatten_observation(p).get("perf.goto_ms")
                 for p in by_page[o.page]]
        peers = [p for p in peers if p]
        if peers and m.get("perf.goto_ms"):
            fastest = min(peers)
            if m["perf.goto_ms"] > max(3500.0, fastest * 3.5):
                out.append(Anomaly(
                    "perf_cliff", o.page,
                    f"navigation took {m['perf.goto_ms']:.0f}ms vs fastest "
                    f"{fastest:.0f}ms under {o.conditions.get('label','')}",
                    o.conditions,
                    suspects=[o.conditions.get("throttle", "none")]))

    # missing components relative to site-wide presence
    all_comp_names = set()
    for o in observations:
        for k in flatten_observation(o):
            if k.startswith("dom.comp_present."):
                all_comp_names.add(".".join(k.split(".")[2:]))
    for o in observations:
        m = flatten_observation(o)
        for c in sorted(all_comp_names):
            present = m.get(f"dom.comp_present.{c}")
            if present == 0 and c not in ("ta-chat-panel", "ta-chat-form"):
                out.append(Anomaly(
                    "missing_component", o.page,
                    f"component #{c} absent from DOM",
                    o.conditions))

    # de-duplicate identical anomalies
    seen, uniq = set(), []
    for a in out:
        k = (a.kind, a.page, a.detail)
        if k not in seen:
            seen.add(k)
            uniq.append(a)
    return uniq


# ---------------------------------------------------------------------------
# Observational correlation mining (never causal by itself)
# ---------------------------------------------------------------------------

def mine_correlations(observations: list[Observation], graph) -> list[tuple]:
    """Pairwise metric co-variation across observational conditions.
    Returns [(cause_metric, effect_metric, pearson_r, n)] for |r|>=0.6."""
    import numpy as np

    rows = [flatten_observation(o) for o in observations]
    if len(rows) < 3:
        return []
    keys = sorted(set.intersection(*[set(r.keys()) for r in rows]))
    keys = [k for k in keys
            if not k.startswith("seq.")
            and k not in ("dom.measure_error",)]
    found = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            va = np.array([r.get(a, np.nan) for r in rows], dtype=float)
            vb = np.array([r.get(b, np.nan) for r in rows], dtype=float)
            mask = ~(np.isnan(va) | np.isnan(vb))
            if mask.sum() < 3:
                continue
            va, vb = va[mask], vb[mask]
            if np.std(va) == 0 or np.std(vb) == 0:
                continue
            r = float(np.corrcoef(va, vb)[0, 1])
            if abs(r) >= 0.6:
                found.append((a, b, r, int(mask.sum())))
    return sorted(found, key=lambda t: -abs(t[2]))


# ---------------------------------------------------------------------------
# Hypothesis generation
# ---------------------------------------------------------------------------

# name-affinity: which DOM components/metrics a script plausibly drives
# (used to ORDER screening candidates; evidence still decides truth).
_AFFINITY = {
    "chat.js": ["ta-chat", "ta-chat-panel", "ta-chat-form"],
    "forms.js": ["newsletter-form", "footer-newsletter-email",
                 "dom.forms_js_hooks"],
    "cart.js": [],
    "header-account.js": ["mobile-menu"],
    "tailwind.css": ["mobile-menu", "ta-cookie", "newsletter-form",
                     "main-content"],
}


@dataclass
class CompetingSet:
    """Hypotheses that explain the SAME anomaly — experiments should be
    chosen to discriminate between them (req. 6)."""
    anomaly: str
    hypothesis_ids: list[str] = field(default_factory=list)


def _mk(cause_key, cause_kind, cause_label, effect_metric, mechanism,
        origin, predictions, falsification, prior, parent=None) -> Hypothesis:
    ek, ekind = metric_factor(effect_metric)
    h = Hypothesis(
        id=new_id("hyp"), claim=f"{cause_label} causally affects {effect_metric}",
        cause=cause_key, effect=ek, mechanism=mechanism, origin=origin,
        parent_id=parent, prior=prior, posterior=prior,
        predictions=predictions,
        falsification_criteria=falsification)
    return h


def generate(observations: list[Observation], graph, max_hypotheses: int = 14,
             existing: list[Hypothesis] | None = None) -> tuple[list[Hypothesis], list[CompetingSet]]:
    """Generate the next wave of hypotheses from anomalies + screening."""
    existing = existing or []
    # every hypothesis ever staked on a pair blocks naive regeneration:
    # refuted pairs may only return through revise_from_refutation (req. 9)
    taken = {(h.cause, h.effect) for h in existing}
    hyps: list[Hypothesis] = []
    sets: list[CompetingSet] = []

    # ---- channel 1: anomalies -> competing explanation sets ----------------
    for anom in detect_anomalies(observations):
        cs = CompetingSet(anomaly=anom.describe())
        candidates: list[Hypothesis] = []

        # explanation family A: a named resource is necessary for the outcome
        detail_metric = _anomaly_metric(anom)
        for suspect in anom.suspects:
            if not suspect.startswith("res:"):
                continue
            base = suspect.split(":", 1)[1]
            if detail_metric is None:
                continue
            pred = Prediction(
                experiment_design={"type": "block", "pattern": base},
                metric=detail_metric,
                direction=_anomaly_direction(anom),
                min_effect=1.0,
                rationale=f"if {base} is necessary, removing it reproduces the anomaly")
            key, kind, label = res_factor(base, False)
            if (key, detail_metric) in taken:
                continue
            candidates.append(_mk(
                key, kind, label, detail_metric,
                f"{anomaly_page_note(anom)}{anom.kind}: the anomaly may "
                f"depend on {base} ({anom.detail})", "anomaly", [pred],
                f"blocking {base} must change {detail_metric}; "
                f"no change falsifies the dependency",
                prior=0.45))
            taken.add((key, detail_metric))

        # explanation family B: browser conditions
        if detail_metric is not None:
            vw = anom.conditions.get("viewport", {}).get("width")
            if vw and vw < 900:
                key, kind = condition_factor("viewport_width")
                if (key, detail_metric) not in taken:
                    candidates.append(_mk(
                        key, kind, "viewport width", detail_metric,
                        "anomaly may be a responsive-behavior difference "
                        "tied to viewport width", "anomaly",
                        [Prediction(
                            experiment_design={"type": "viewport",
                                              "width": 1440},
                            metric=detail_metric, direction=-1,
                            min_effect=1.0,
                            rationale="desktop viewport should remove a "
                            "mobile-only condition")],
                        "if the metric is unchanged when the viewport "
                        "changes, viewport is not the cause", 0.4))
                    taken.add((key, detail_metric))

        if candidates:
            n = len(candidates)
            for c in candidates:
                c.prior = c.posterior = round(1.0 / (n + 1), 3)  # + "other causes"
            hyps.extend(candidates)
            cs.hypothesis_ids = [c.id for c in candidates]
            sets.append(cs)

    # ---- channel 3: assumption-free screening probes ------------------------
    unsettled = [h for h in existing
                 if h.status in (HypStatus.ACTIVE, HypStatus.PROPOSED)]
    screening_budget = max(0, max_hypotheses - len(unsettled))
    if screening_budget > 0:
        hyps.extend(_screening(observations, taken, screening_budget))

    return hyps[:max_hypotheses], sets



def anomaly_page_note(anom) -> str:
    page = getattr(anom, "page", "")
    return f"[{page}] " if page else ""

def _anomaly_metric(anom: Anomaly):
    return {"failed_resource": "res.failed",
            "console_error": "err.pageerror",
            "font_not_loaded": "dom.fonts_status_loaded",
            "perf_cliff": "perf.goto_ms",
            "missing_component": None}.get(anom.kind)


def _anomaly_direction(anom: Anomaly) -> int:
    """Direction the anomaly metric moves when the cause is removed."""
    return {"failed_resource": 1,
            "console_error": 1,
            "font_not_loaded": -1,
            "perf_cliff": 1}.get(anom.kind, 1)


def _screening(observations: list[Observation], taken: set,
               budget: int) -> list[Hypothesis]:
    """Probe (resource/condition -> observable metric) pairs without assuming
    any relationship holds. Priors start LOW; only interventions raise them.

    Components invisible under standard desktop conditions get probed with a
    mobile-context design: testing a floor-pinned boolean at desktop cannot
    detect anything, so the experiment context must move the baseline."""
    out: list[Hypothesis] = []
    scripts, styles, third_party = set(), set(), set()
    comps = set()
    # metric values under standard (desktop, unthrottled, JS on) conditions,
    # mobile conditions and JS-off conditions — used both for design
    # feasibility checks and to DISCOVER which components are worth probing
    std_value: dict[str, float] = {}
    alt_value: dict[str, float] = {}
    nojs_value: dict[str, float] = {}
    observed_metrics: set = set()
    mobile_vp = {"viewport": {"width": 390, "height": 844}}
    for o in observations:
        cond = o.conditions
        is_std = (cond.get("throttle", "none") == "none"
                  and not cond.get("js_disabled", False)
                  and (cond.get("viewport") or {}).get("width", 1440) >= 1000)
        is_mobile = (cond.get("throttle", "none") == "none"
                     and not cond.get("js_disabled", False)
                     and (cond.get("viewport") or {}).get("width", 1440) < 1000)
        is_nojs = bool(cond.get("js_disabled", False))
        for k, v in flatten_observation(o).items():
            observed_metrics.add(k)
            if is_std:
                std_value[k] = v
            if is_mobile:
                alt_value[k] = v
            if is_nojs:
                nojs_value[k] = v

    def feasible_design(target: str, direction: int) -> dict | None:
        """A decrease-prediction on a boolean metric needs head-room: the
        metric must be 1 in SOME context we can create. If it is pinned at 0
        everywhere we can observe, the hypothesis is not falsifiable-as-
        designed and is simply not emitted."""
        if not target.startswith("dom.comp_visible") or direction >= 0:
            return {}
        if std_value.get(target, 0) > 0:
            return {}
        if alt_value.get(target, 0) > 0:
            return {"context": dict(mobile_vp)}
        return None
    for o in observations:
        for r in o.resources:
            if r.kind == "script" and not r.third_party:
                scripts.add(r.basename)
            elif r.kind == "stylesheet" and not r.third_party:
                styles.add(r.basename)
            elif r.third_party:
                third_party.add(r.basename)
        for k in flatten_observation(o):
            if k.startswith("dom.comp_visible."):
                comps.add(k)

    def add(cause, label, kind, effect, mechanism, design, direction,
            min_effect, prior=0.3):
        if (cause, effect) in taken or len(out) >= budget:
            return
        feas = feasible_design(effect, direction)
        if feas is None:
            return  # unfalsifiable-as-designed: not emitted
        if feas:
            design = {**design, **feas}
        taken.add((cause, effect))  # dedupe within a single wave too
        out.append(_mk(
            cause, kind, label, effect, mechanism, "screening",
            [Prediction(experiment_design=design, metric=effect,
                        direction=direction, min_effect=min_effect,
                        rationale="necessity probe: remove/alter the cause "
                        "and measure the effect")],
            f"no significant change in {effect} when the intervention is "
            "applied falsifies this hypothesis", prior))

    # Build probe queues per category, then interleave round-robin so every
    # wave contains a MIX: component dependencies, layout, fonts, responsive
    # conditions, network, timing, and JS-execution probes.
    cat_comp, cat_css, cat_font, cat_cond, cat_delay = [], [], [], [], []

    for s in sorted(scripts):
        key, kind, label = res_factor(s, False)
        affinity = _AFFINITY.get(s, [])
        targets = []
        for a in affinity:
            if a.startswith("dom."):
                targets.append(a)
            else:
                targets.append(f"dom.comp_visible.{a}")
        targets += [c for c in comps
                    if not any(c.endswith(a) for a in affinity)
                    and not any(a.startswith("dom.") for a in affinity
                                if c == a)]
        # only stake claims on metrics the target actually exposes: probing
        # a metric the site never measures is unfalsifiable noise
        targets = [t for t in targets if t in observed_metrics]
        seen_t = set()
        for t in targets:
            if t in seen_t:
                continue
            seen_t.add(t)
            cat_comp.append((key, label, kind, t,
                             f"screening: {s} may inject or drive {t}",
                             {"type": "block", "pattern": s}, -1, 1.0, 0.32))

    for c in sorted(styles):
        key, kind, label = res_factor(c, False)
        cat_css.append((key, label, kind, "dom.doc_height_px",
                        f"screening: layout height may depend on {c}",
                        {"type": "block", "pattern": c}, -1, 300.0, 0.35))

    for t in sorted(third_party):
        if "font" not in t and "css2" not in t:
            continue
        key, kind, label = res_factor(t, True)
        cat_font.append((key, label, kind, "dom.fonts_status_loaded",
                         f"screening: font loading may depend on {t}",
                         {"type": "block", "pattern": t}, -1, 1.0, 0.35))

    # condition probes — DISCOVERED from observational contrasts rather
    # than hardcoded for any specific site:
    #  * components whose visibility differs desktop vs mobile are
    #    responsive-behaviour candidates (viewport intervention);
    #  * components present with JS but absent without it are
    #    JS-injection candidates (disable_js intervention).
    key, kind = condition_factor("viewport_width")
    responsive = sorted(
        k.split("dom.comp_visible.", 1)[1]
        for k in std_value
        if k.startswith("dom.comp_visible.")
        and k in alt_value and std_value[k] != alt_value[k])
    for comp in responsive[:4]:
        direction = 1 if alt_value.get(
            f"dom.comp_visible.{comp}", 0) > std_value.get(
            f"dom.comp_visible.{comp}", 0) else -1
        cat_cond.append((key, "viewport width", kind,
                         f"dom.comp_visible.{comp}",
                         f"screening: observation shows #{comp} visibility "
                         "differs between desktop and mobile viewports — a "
                         "responsive breakpoint may control it",
                         {"type": "viewport", "width": 390}, direction, 1.0,
                         0.4))

    key, kind = condition_factor("throttle")
    cat_cond.append((key, "network throttling", kind, "perf.goto_ms",
                     "screening: latency should slow navigation",
                     {"type": "throttle", "profile": "slow3g"}, +1, 1500.0,
                     0.5))

    key, kind = condition_factor("js_disabled")
    js_injected = sorted(
        k.split("dom.comp_present.", 1)[1]
        for k in std_value
        if k.startswith("dom.comp_present.")
        and k in nojs_value and std_value[k] > nojs_value[k])
    for comp in js_injected[:4]:
        cat_cond.append((key, "JavaScript execution", kind,
                         f"dom.comp_present.{comp}",
                         f"screening: observation shows #{comp} present with "
                         "JS but absent without it — it may be JS-injected "
                         "rather than server-rendered",
                         {"type": "disable_js"}, -1, 1.0, 0.4))

    for s in sorted(scripts):
        key, kind, label = res_factor(s, False)
        cat_delay.append((key, label, kind, "perf.domContentLoaded_ms",
                          f"screening: delaying {s} may delay "
                          "DOMContentLoaded if it is render-blocking or "
                          "deferred-synchronous",
                          {"type": "delay", "pattern": s, "delay_ms": 4000},
                          +1, 2000.0, 0.25))

    queues = [cat_comp, cat_css, cat_font, cat_cond, cat_delay]
    i = 0
    while len(out) < budget and any(queues):
        emitted = False
        for q in queues:
            if q:
                cause, label, kind, eff, mech, design, direction, me, pr = \
                    q.pop(0)
                add(cause, label, kind, eff, mech, design, direction, me, pr)
                emitted = True
                if len(out) >= budget:
                    break
        if not emitted:
            break
        i += 1

    return out


# ---------------------------------------------------------------------------
# Revision after refutation (req. 1: Revised Hypothesis step)
# ---------------------------------------------------------------------------

def revise_from_refutation(h: Hypothesis, experiment, result) -> Hypothesis | None:
    """When an intervention shows NO effect (or the opposite), the original
    mechanism is dead. Emit its falsification-driven complement."""
    rc = result.result_class.value
    if rc == "no_effect":
        new_claim_mechanism = (
            f"REVISED after refutation of {h.id}: {h.effect} appears "
            f"INDEPENDENT of {h.cause} — the outcome may be server-rendered, "
            f"cached, or driven by another factor")
        direction, prior = 0.0, 0.3
        predictions = [Prediction(
            experiment_design=experiment.arms[1].intervention
            if len(experiment.arms) > 1 else {},
            metric=h.effect, direction=0, min_effect=1.0,
            rationale="independence claim: re-test must again show no "
            "effect in the opposite direction")]
    elif rc == "opposite":
        new_claim_mechanism = (
            f"REVISED after opposite effect vs {h.id}: the true direction of "
            f"{h.cause} -> {h.effect} is REVERSED")
        predictions = [Prediction(
            experiment_design=experiment.arms[1].intervention
            if len(experiment.arms) > 1 else {},
            metric=h.effect, direction=-h.predictions[0].direction
            if h.predictions else -1, min_effect=1.0,
            rationale="direction must replicate reversed")]
        direction, prior = -1, 0.35
    else:
        return None

    revised = Hypothesis(
        id=new_id("hyp"),
        claim=f"(revised) {h.cause} does not cause {h.effect} as originally "
              f"claimed; {new_claim_mechanism}",
        cause=h.cause, effect=h.effect, mechanism=new_claim_mechanism,
        origin="revision", parent_id=h.id, prior=prior, posterior=prior,
        predictions=predictions,
        falsification_criteria=
            "a replicated significant effect in the original direction "
            "falsifies this revision",
        evidence_against=list(h.evidence_for))
    return revised
