"""Human-readable investigation reports (Markdown) + machine state (JSON)."""
from __future__ import annotations

import json
from pathlib import Path

from .types import (Discovery, Experiment, ExperimentResult, HypStatus,
                    Hypothesis)
from .observe.instrument import flatten_observation


def write_run_report(out_path: Path, *, config, graph, observations,
                     hypotheses: list[Hypothesis],
                     experiments: dict[str, tuple[Experiment, ExperimentResult]],
                     discoveries: list[Discovery], merit, budget,
                     anomalies, run_meta: dict) -> Path:
    lines: list[str] = []
    A = lines.append

    A("# TechAbout Causal Lab — Autonomous Investigation Report")
    A("")
    A(f"- generated: {run_meta.get('generated_at')}")
    A(f"- target: {config.base_url} (pages: {', '.join(config.pages)})")
    A(f"- budget used: {budget.summary()}")
    A(f"- scientific merit: **{merit.total():.1f}** — {merit.summary()}")
    A("")
    A("## 1. Scientific loop status")
    A("")
    A("```")
    A("Observation -> Hypothesis -> Experiment Design -> Controlled Experiment")
    A("           -> Evidence -> Falsification -> Revised Hypothesis")
    A("           -> Independent Verification -> Discovery")
    A("```")
    active = [h for h in hypotheses if h.status in (HypStatus.ACTIVE,
                                                    HypStatus.PROPOSED)]
    refuted = [h for h in hypotheses if h.status == HypStatus.REFUTED]
    supported = [h for h in hypotheses if h.status == HypStatus.SUPPORTED]
    A(f"- passive observations: {len(observations)} over "
      f"{len(set(o.page for o in observations))} pages")
    A(f"- anomalies detected: {len(anomalies)}")
    A(f"- hypotheses: {len(hypotheses)} total | {len(active)} active | "
      f"{len(refuted)} refuted (preserved) | {len(supported)} supported")
    A(f"- controlled experiments: {len(experiments)}")
    A(f"- verified discoveries: {sum(1 for d in discoveries if d.verified)}"
      f"/{len(discoveries)}")
    A("")

    A("## 2. Verified discoveries (survived independent review)")
    A("")
    if discoveries:
        for d in discoveries:
            A(f"### {d.id}: {d.statement}")
            A("")
            A(f"- hypothesis: `{d.hypothesis_id}`  ")
            A(f"- experiments: {', '.join(d.experiment_ids)}  ")
            A(f"- reproducibility package: `{d.package_path}`  ")
            A(f"- independently verified: **{d.verified}**")
            if d.review:
                for agent in d.review.get("agents", []):
                    A(f"- reviewer `{agent.get('role')}`: "
                      f"{'APPROVED' if agent.get('approved') else 'REJECTED'} — "
                      f"{str(agent.get('reasoning', ''))[:200]}")
            if d.counterfactual:
                A("")
                A("**Counterfactual analysis**")
                A("")
                A(f"- OBSERVED: {d.counterfactual['observed']['statement']}")
                A("")
                A(f"- SIMULATED: {d.counterfactual['counterfactual']['statement']}")
                A("")
                A(f"- VALIDATION: "
                  f"{d.counterfactual['validation_experiment']['statement']}")
            A("")
    else:
        A("_No hypothesis crossed the support threshold this run._")
    A("")

    A("## 3. Refuted hypotheses (scientific memory — never deleted)")
    A("")
    if refuted:
        A("| hypothesis | claim | refutation evidence |")
        A("|---|---|---|")
        for h in refuted:
            ev = ", ".join(h.evidence_against[-3:]) or "—"
            A(f"| `{h.id}` | {h.claim[:80]} | {ev} |")
    else:
        A("_None refuted this run._")
    A("")

    A("## 4. Controlled experiment ledger")
    A("")
    A("Record per experiment: Control -> Variable Changed -> Expected "
      "Effect -> Actual Effect -> Evidence.")
    A("")
    A("| experiment | page | variable changed | expected | actual | result |")
    A("|---|---|---|---|---|---|")
    for exp, res in experiments.values():
        design = exp.arms[1].intervention if len(exp.arms) > 1 else {}
        d = (res.deltas or {}).get(exp.target_metric, {})
        A(f"| `{exp.id}` | {exp.page} | `{design.get('type')}"
          f":{design.get('pattern', design.get('width', ''))}` | "
          f"{exp.expected_direction:+d}×≥{exp.min_effect} on "
          f"`{exp.target_metric}` | "
          f"Δ={d.get('delta', 'n/a')} ({d.get('control')}→{d.get('treatment')}) | "
          f"**{res.result_class.value}** |")
    A("")

    A("## 5. Causal graph")
    A("")
    stats = graph.stats()
    A(f"- version: v{stats['version']}  ")
    A(f"- factors: {stats['factors']}  ")
    A(f"- edges by kind: {stats['by_kind']}")
    A("")
    A("```mermaid")
    A(graph.to_mermaid())
    A("```")
    A("")
    A("Edge legend: `==>` experimentally supported causal | `-->` candidate "
      "under test | `-.->` observed correlation only | `--x` refuted "
      "(preserved).")
    A("")

    A("## 6. Anomalies detected during passive observation")
    A("")
    if anomalies:
        for a in anomalies:
            A(f"- {a.describe()}")
    else:
        A("_Site behaved uniformly under all observational conditions — "
          "screening hypotheses drove the search instead._")
    A("")

    A("## 7. Reproducibility")
    A("")
    A("Every discovery ships a package (`experiment.json`, `evidence.json`, "
      "`environment.json`, screenshots, `rerun.py`, `VERIFICATION.md`). "
      "Reproduce any discovery with:")
    A("")
    A("```bash")
    A("python main.py verify --package data/packages/<DISCOVERY-ID>")
    A("```")
    A("")
    A("## 8. Honesty statement")
    A("")
    A("- Correlation edges are observational only and are never promoted "
      "without intervention evidence (enforced in `CausalGraph.promote()`).")
    A("- Counterfactual sections labelled SIMULATED are extrapolations from "
      "measured effect sizes, never observations of the live site.")
    A("- All interventions execute inside the lab's own browser context; "
      "the origin only sees rate-limited ordinary GET loads.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def save_state(data_dir: Path, *, graph, hypotheses, experiments,
               discoveries, anomalies) -> Path:
    state = {
        "graph_stats": graph.stats(),
        "hypotheses": [h.to_dict() for h in hypotheses],
        "experiments": {
            eid: {"experiment": {
                "id": e.id, "title": e.title, "page": e.page,
                "arms": [{"name": a.name, "intervention": a.intervention}
                         for a in e.arms],
                "target_metric": e.target_metric,
                "expected_direction": e.expected_direction,
                "min_effect": e.min_effect,
                "hypothesis_ids": e.hypothesis_ids,
                "design_rationale": e.design_rationale,
                "discriminates": e.discriminates},
                "result": r.to_dict()}
            for eid, (e, r) in experiments.items()},
        "discoveries": [dict(d.__dict__) for d in discoveries],
        "anomalies": [a.__dict__ if hasattr(a, "__dict__") else str(a)
                      for a in anomalies],
    }
    p = data_dir / "state.json"
    p.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# End-of-run audit summary: issues found, what to fix, what to investigate next
# ---------------------------------------------------------------------------

def _implication(cause: str, effect: str, design: dict, delta) -> str:
    """Translate a verified causal finding into practical site advice."""
    d = design.get("type", "")
    if effect == "perf.domContentLoaded_ms" and d == "delay":
        return (f"`{cause}` sits on the DOMContentLoaded critical path: "
                f"delaying it by {design.get('delay_ms', '?')}ms moved DCL "
                f"by {delta}ms. Consider `async`/module splitting, preloading "
                "or deferring non-critical work so DCL is not serialized "
                "behind this resource.")
    if effect == "perf.goto_ms" and d == "throttle":
        return (f"Navigation time is network-latency-bound (throttling added "
                f"{delta}ms). Reduce round-trips and critical bytes — "
                "compute is not the bottleneck.")
    if effect.startswith("dom.doc_height") and d == "block":
        return (f"Layout depends critically on this stylesheet: without it "
                f"the rendered document height changes by {delta}px. Inline "
                "critical CSS so first paint is not layout-unstable.")
    if effect.startswith("dom.comp_present.") and d == "disable_js":
        comp = effect.split("dom.comp_present.", 1)[1]
        return (f"`#{comp}` is JS-injected (absent without JavaScript). "
                "If it carries content, provide a server-rendered fallback "
                "for no-JS/error conditions.")
    if effect.startswith("dom.comp_visible.") and d == "viewport":
        comp = effect.split("dom.comp_visible.", 1)[1]
        return (f"`#{comp}` genuinely responds to viewport width — responsive "
                "behaviour verified experimentally.")
    if effect.startswith("dom.fonts"):
        return ("Font loading depends on this resource; consider "
                "self-hosting or `font-display: optional` to avoid text "
                "movement.")
    return ("Experimentally verified dependency — see the reproducibility "
            "package for exact conditions and effect size.")


def write_summary_report(out_path: Path, *, config, graph, observations,
                         hypotheses: list[Hypothesis],
                         experiments: dict[str, tuple[Experiment,
                                                      ExperimentResult]],
                         discoveries: list[Discovery], anomalies, merit,
                         budget, run_meta: dict) -> Path:
    lines: list[str] = []
    A = lines.append

    A("# Audit Summary & Improvement Plan")
    A("")
    A(f"- target: **{config.base_url}**")
    A(f"- pages audited: {', '.join(config.pages)}")
    A(f"- generated: {run_meta.get('generated_at')} (mode: {run_meta.get('mode')})")
    A(f"- budget used: {budget.summary()}; scientific merit: "
      f"**{merit.total():.1f}**")
    A("- companion documents: `investigation.md` (full scientific record), "
      "`packages/` (reproducibility), `graph/` (versioned causal model)")
    A("")

    # ------------------------------------------------------------------ 1
    A("## 1. Site health issues observed (passive evidence)")
    A("")
    issues = 0
    failed = [(o.page, r) for o in observations for r in o.resources
              if r.error or (r.status or 0) >= 400]
    if failed:
        issues += len(failed)
        A(f"### Failed resources ({len(failed)})")
        for page, r in failed[:10]:
            A(f"- `{r.basename}` on {page} — status={r.status} "
              f"error={r.error} ({r.url[:100]})")
        A("")
    errs = [(o.page, c) for o in observations for c in o.console
            if c.get("type") in ("error", "pageerror")]
    if errs:
        issues += len(errs)
        A(f"### Runtime / console errors ({len(errs)})")
        for page, c in errs[:10]:
            A(f"- {page}: `[{c['type']}]` {c['text'][:140]}")
        A("")
    a11y = [(o.page, o.conditions.get("label", ""),
             flatten_observation(o).get("a11y.images_missing_alt", 0))
            for o in observations]
    alt_issues = [x for x in a11y if x[2]]
    if alt_issues:
        issues += 1
        A("### Accessibility: images without alt text")
        worst = max(alt_issues, key=lambda x: x[2])
        A(f"- worst case: **{int(worst[2])} images missing `alt`** on "
          f"{worst[0]} ({worst[1]}) — add descriptive alt attributes.")
        A("")
    perf_rows = [(o.page, o.conditions.get("label", ""),
                  flatten_observation(o).get("perf.goto_ms"))
                 for o in observations]
    slow = [(p, lab, t) for p, lab, t in perf_rows if t]
    if slow:
        fastest = min(t for _, _, t in slow)
        slow_page, slow_label, slowest = max(slow, key=lambda x: x[2])
        A("### Performance envelope (observed navigation time)")
        A(f"- fastest {fastest:.0f}ms → slowest {slowest:.0f}ms "
          f"({slow_page} under {slow_label})")
        A("")
    third = sum(1 for o in observations for r in o.resources
                if r.third_party) / max(1, len(observations))
    weight = sum(flatten_observation(o).get("res.total_bytes", 0)
                 for o in observations
                 if o.conditions.get("label") == "desktop-broadband") \
        / max(1, sum(1 for o in observations
                     if o.conditions.get("label") == "desktop-broadband"))
    A(f"### Exposure summary")
    A(f"- average third-party requests per load: **{third:.0f}**")
    A(f"- average transferred bytes (desktop): "
      f"**{weight / 1024:.0f} KiB**")
    if anomalies:
        A("")
        A("### Anomalies flagged during observation")
        for a in anomalies[:10]:
            A(f"- {a.describe()}")
    if issues == 0 and not anomalies:
        A("_No passive-observation issues detected (the site was healthy "
          "under every observational condition)._")
    A("")

    # ------------------------------------------------------------------ 2
    A("## 2. Verified causal findings — what they mean for the site")
    A("")
    if discoveries:
        for d in discoveries:
            design = {}
            for eid, (e, r) in experiments.items():
                if eid in d.experiment_ids and len(e.arms) > 1:
                    design = e.arms[1].intervention
            delta = None
            if d.counterfactual:
                delta = d.counterfactual.get("relationship", {}).get(
                    "measured_effect")
            A(f"### {d.statement}")
            A("")
            A(f"- practical implication: "
              f"{_implication(_cause_of(d), _effect_of(d), design, delta)}")
            A(f"- reproduce: `python main.py verify --package "
              f"{d.package_path}`")
            A("")
    else:
        A("_No hypothesis reached the independent-verification threshold "
          "yet — see section 3 for what to run next._")
    A("")

    # ------------------------------------------------------------------ 3
    A("## 3. Open questions — hypotheses still to test or improve")
    A("")
    open_hyps = [h for h in hypotheses
                 if h.status in (HypStatus.ACTIVE, HypStatus.PROPOSED,
                                 HypStatus.UNRESOLVED)]
    if open_hyps:
        open_hyps.sort(key=lambda h: -h.posterior)
        A("| posterior | hypothesis | origin | recommended next experiment |")
        A("|---|---|---|---|")
        for h in open_hyps[:15]:
            design = h.predictions[0].experiment_design if h.predictions \
                else {}
            nxt = (f"`{design.get('type')}"
                   f":{design.get('pattern', design.get('width', ''))}` → "
                   f"{h.predictions[0].metric}" if h.predictions else "—")
            A(f"| {h.posterior:.2f} | {h.claim[:70]} | {h.origin} | {nxt} |")
        A("")
        A("Re-run `discover` with a larger `--max-experiments` budget: the "
          "investigation resumes automatically and prioritises exactly "
          "these.")
    else:
        A("_No open hypotheses — the current hypothesis space is settled._")
    A("")

    # ------------------------------------------------------------------ 4
    refuted = [h for h in hypotheses if h.status == HypStatus.REFUTED]
    A("## 4. Beliefs the audit ruled out (preserved with evidence)")
    A("")
    if refuted:
        for h in refuted[:12]:
            ev = ", ".join(h.evidence_against[-2:]) or "—"
            A(f"- ~~{h.claim[:80]}~~ — refuted by {ev}")
        A(f"- …and {max(0, len(refuted) - 12)} more (see "
          "`investigation.md`)")
    else:
        A("_None refuted this run._")
    A("")

    # ------------------------------------------------------------------ 5
    A("## 5. Experiment quality review (engine self-critique)")
    A("")
    errors = [(e, r) for e, r in experiments.values()
              if r.result_class.value == "error"]
    if errors:
        A(f"### Experiments that failed to execute ({len(errors)})")
        for e, r in errors[:8]:
            design = e.arms[1].intervention if len(e.arms) > 1 else {}
            A(f"- `{e.id}` {design.get('type')}:"
              f"{design.get('pattern', '')} on {e.page} — intervention did "
              "not engage or the target metric was unmeasurable; check the "
              "pattern matches a real resource, or that the metric exists "
              "on this site.")
        A("")
    pinned = sum(1 for e, r in experiments.values()
                 if r.result_class.value == "no_effect"
                 and any(v["control"] in (0.0, 1.0)
                         for v in (r.deltas or {}).values()
                         if isinstance(v, dict)
                         and isinstance(v.get("control"), (int, float))))
    A(f"### Design quality notes")
    A(f"- no-effect experiments with a floor/ceiling control arm: "
      f"**{pinned}** — where the control was pinned these carry no "
      "evidential weight (the interpreter ignored them)")
    untested = [h for h in open_hyps
                if not any(eid in (h.evidence_for + h.evidence_against)
                           for eid in experiments)]
    A(f"- open hypotheses with no experimental evidence yet: "
      f"**{len(untested)}**")
    A("- components that only respond to user actions (menus, modals) "
      "cannot be causally probed by passive loads alone — interaction "
      "probes are a natural extension")
    A("")

    # ------------------------------------------------------------------ 6
    A("## 6. Recommended next actions")
    A("")
    A("**For the site:**")
    recs = []
    if failed:
        recs.append(f"fix the {len(failed)} failing resource(s) listed in "
                    "section 1")
    if errs:
        recs.append("eliminate the runtime/console errors observed on load")
    if alt_issues:
        recs.append("add `alt` text to images")
    for d in discoveries:
        design = {}
        for eid, (e, r) in experiments.items():
            if eid in d.experiment_ids and len(e.arms) > 1:
                design = e.arms[1].intervention
        if design.get("type") == "delay" and \
                d.hypothesis_id and "domContentLoaded" in d.statement:
            recs.append(f"take `{_cause_of(d)}` off the DOMContentLoaded "
                        "critical path (async/defer/preload)")
    if recs:
        for r in recs[:8]:
            A(f"- {r}")
    else:
        A("- no site-side actions required from this audit")
    A("")
    A("**For the investigation:**")
    A(f"- run `python main.py discover --url {config.base_url}` again with "
      f"a larger `--max-experiments` budget to settle the {len(open_hyps)} "
      "open hypotheses")
    A("- widen coverage with `--pages` (e.g. article or form-heavy "
      "templates) to test whether verified relationships generalise")
    A("- re-run `verify --package data*/packages/...` after any site "
      "change: a failed replication means the causal structure changed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _cause_of(d: Discovery) -> str:
    return d.statement.split(" causally affects")[0].strip()


def _effect_of(d: Discovery) -> str:
    try:
        return d.statement.split(" causally affects")[1].split(" (")[0].strip()
    except IndexError:
        return ""
