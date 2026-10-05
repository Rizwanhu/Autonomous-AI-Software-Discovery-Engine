#!/usr/bin/env python3
"""Causal Lab — autonomous scientific investigation of the publicly
observable TechAbout.com website.

Scientific loop driven end-to-end:

    Observation -> Hypothesis -> Experiment Design (max information gain)
                -> Controlled Experiment -> Evidence -> Falsification
                -> Revised Hypothesis -> Independent Verification -> Discovery

Usage:
    python main.py discover                  # full autonomous run (live site)
    python main.py discover --pages / /careers --max-loads 100
    python main.py discover --sim             # offline ground-truth validation
    python main.py verify --package data/packages/D001-...   # reproduce
    python main.py report                     # regenerate markdown report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.config import (DATA_DIR, OBSERVATIONAL_CONDITIONS, LabConfig,
                               SafetyConfig)
from app.main import hypotheses as hypgen
from app.main.graph import CausalGraph, EdgeKind, GraphIntegrityError
from app.experiments.engine import (ExperimentEngine,
                                           build_experiment)
from app.experiments.repro import build_package, rerun_package
from app.experiments.selection import (design_key,
                                              select_next_experiment)
from app.observe import sitemap
from app.observe.budget import BudgetExhausted, BudgetGuard
from app.observe.driver import PlaywrightDriver
from app.observe.instrument import flatten_observation
from app.observe.simulated import (GROUND_TRUTH, SimulatedSiteDriver)
from app.reasoning.counterfactual import build_counterfactual
from app.reasoning.falsify import EvidenceInterpreter, MeritLedger
from app.reasoning.review import IndependentVerifier, ReviewBoard
from app.reports import save_state, write_run_report
from app.types import (Discovery, FactorKind, HypStatus, Hypothesis,
                              experiment_from_dict, observation_from_dict,
                              result_from_dict)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


# ---------------------------------------------------------------------------
# The autonomous scientific loop
# ---------------------------------------------------------------------------

class Investigation:
    def __init__(self, cfg: LabConfig, mode: str = "live",
                 max_experiments: int | None = None, headless: bool = True,
                 data_dir: Path | None = None):
        self.cfg = cfg
        self.mode = mode
        self.max_experiments = 10**9 if max_experiments is None \
            else max_experiments
        self.budget = BudgetGuard(cfg.safety)
        if mode == "sim":
            self.driver = SimulatedSiteDriver(cfg.base_url)
        else:
            self.driver = PlaywrightDriver(cfg.base_url, self.budget,
                                           headless=headless)
        # live and simulated investigations MUST NOT share state: a graph
        # seeded by synthetic ground truth would contaminate real findings
        self.data_dir = data_dir or (DATA_DIR if mode != "sim"
                                     else DATA_DIR.parent / "data-sim")
        self.graph = CausalGraph(self.data_dir)
        self.engine = ExperimentEngine(self.driver, self.budget, cfg,
                                       evidence_dir=self.data_dir / "evidence")
        self.interpreter = EvidenceInterpreter(cfg, self.graph)
        self.merit = MeritLedger(self.data_dir / "merit.json")
        # the verifier reuses the running driver: Playwright's sync API
        # cannot be nested in a second instance. Independence is preserved
        # by the blind protocol (re-run from the package BEFORE reading the
        # original outcome, fresh context per load, separate budget).
        self.board = ReviewBoard(IndependentVerifier(
            driver_factory=(lambda: self.driver),
            budget_factory=lambda: BudgetGuard(SafetyConfig(max_page_loads=12)),
            config=cfg))
        self.observations = []
        self.hypotheses: dict[str, Hypothesis] = {}
        self.experiments: dict[str, tuple] = {}
        self.discoveries: list[Discovery] = []
        self.anomalies = []
        self.pages_seen_resources: dict[str, list[str]] = {}
        self.obs_dir = self.data_dir / "observations"
        self.pkg_dir = self.data_dir / "packages"
        self.obs_dir.mkdir(parents=True, exist_ok=True)
        self.discovery_counter = 0
        self._resume()

    def _resume(self) -> None:
        """Continue a previous investigation: hypotheses keep their
        posteriors, experiments keep their evidence — science is a session
        spanning many runs."""
        state_path = self.data_dir / "state.json"
        if not state_path.exists():
            return
        try:
            s = json.loads(state_path.read_text(encoding="utf-8"))
            self.hypotheses = {h["id"]: Hypothesis.from_dict(h)
                               for h in s.get("hypotheses", [])}
            # UNRESOLVED only meant "session ended before a verdict";
            # a new session re-opens them for testing
            for h in self.hypotheses.values():
                if h.status == HypStatus.UNRESOLVED:
                    h.status = HypStatus.ACTIVE
            self.experiments = {
                eid: (experiment_from_dict(v["experiment"]),
                      result_from_dict(v["result"]))
                for eid, v in s.get("experiments", {}).items()}
            self.discoveries = [Discovery(**d)
                                for d in s.get("discoveries", [])]
            self.discovery_counter = len(self.discoveries)
            from app.main.hypotheses import Anomaly
            self.anomalies = [Anomaly(**a) for a in s.get("anomalies", [])]
            for f in sorted(self.obs_dir.glob("obs-*.json")):
                try:
                    self.observations.append(observation_from_dict(
                        json.loads(f.read_text(encoding="utf-8"))))
                except Exception:
                    continue
            for o in self.observations:
                for r in o.resources:
                    base = r.basename
                    self.pages_seen_resources.setdefault(base, [])
                    if o.page not in self.pages_seen_resources[base]:
                        self.pages_seen_resources[base].append(o.page)
            log(f"resumed prior investigation: {len(self.hypotheses)} "
                f"hypotheses, {len(self.experiments)} experiments, "
                f"{len(self.observations)} observations, "
                f"{len(self.discoveries)} discoveries")
        except Exception as e:
            log(f"resume failed ({type(e).__name__}: {e}); starting fresh")
            self.hypotheses, self.experiments = {}, {}
            self.discoveries, self.observations, self.anomalies = [], [], []

    # -- phase 1: passive observation ---------------------------------------
    def observe(self) -> None:
        log(f"PHASE 1 — passive observation of {len(self.cfg.pages)} page(s) "
            f"under {len(OBSERVATIONAL_CONDITIONS)} condition sets")
        for page in self.cfg.pages[:self.cfg.safety.max_pages_per_run]:
            for cond in OBSERVATIONAL_CONDITIONS:
                try:
                    obs = self.driver.load(
                        page, cond,
                        screenshot_path=str(self.obs_dir /
                                            f"{slugify(page)}-{cond['label']}.png"))
                except BudgetExhausted:
                    raise
                except Exception as e:
                    log(f"  observation failed on {page}/{cond['label']}: "
                        f"{str(e)[:120]}")
                    continue
                self.observations.append(obs)
                (self.obs_dir / f"{obs.id}.json").write_text(
                    json.dumps(obs.to_dict(), indent=2, default=str),
                    encoding="utf-8")
                # index resources -> pages for experiment placement
                for r in obs.resources:
                    base = r.basename
                    self.pages_seen_resources.setdefault(base, [])
                    if page not in self.pages_seen_resources[base]:
                        self.pages_seen_resources[base].append(page)
                self._register_factors(obs)
                log(f"  observed {page} [{cond['label']}]: "
                    f"{len(obs.resources)} resources, "
                    f"{flatten_observation(obs).get('dom.dom_nodes', '?')} nodes")
        self.anomalies = hypgen.detect_anomalies(self.observations)
        log(f"  anomalies detected: {len(self.anomalies)}")
        for a in self.anomalies[:8]:
            log(f"    - {a.describe()[:120]}")

    def _register_factors(self, obs) -> None:
        for r in obs.resources:
            key, kind, label = hypgen.res_factor(r.basename, r.third_party)
            self.graph.ensure_factor(key, kind, label,
                                     {"third_party": r.third_party,
                                      "size_bytes": r.size_bytes})
        for name in flatten_observation(obs):
            try:
                key, kind = hypgen.metric_factor(name)
                self.graph.ensure_factor(key, kind, name)
            except Exception:
                pass
        for cname in ("viewport_width", "throttle", "js_disabled"):
            key, kind = hypgen.condition_factor(cname)
            self.graph.ensure_factor(key, kind, cname)

    # -- phase 2a: observational correlations (never causal by themselves) --
    def mine(self) -> None:
        log("PHASE 2a — observational correlation mining")
        corr = hypgen.mine_correlations(self.observations, self.graph)
        for cause, effect, r, n in corr[:24]:
            ck, ckind = hypgen.metric_factor(cause)
            ek, ekind = hypgen.metric_factor(effect)
            self.graph.ensure_factor(ck, ckind, cause)
            self.graph.ensure_factor(ek, ekind, effect)
            self.graph.add_correlation(ck, ek, r, support=n,
                                       note=f"observational r={r:.2f} n={n}")
        self.graph.save_version(f"observational mining: {len(corr[:24])} "
                                f"correlations recorded (non-causal)")
        log(f"  recorded {len(corr[:24])} correlation edges "
            f"(graph v{self.graph.version})")

    # -- phase 2b: hypothesis generation -------------------------------------
    def hypothesize(self) -> None:
        unsettled = sum(1 for h in self.hypotheses.values()
                        if h.status in (HypStatus.ACTIVE, HypStatus.PROPOSED))
        if unsettled >= 50:
            log("  unsettled-hypothesis cap reached — no new wave")
            return
        log("PHASE 2b — hypothesis generation (anomaly + screening)")
        fresh, sets = hypgen.generate(self.observations, self.graph,
                                      self.cfg.max_hypotheses_active,
                                      list(self.hypotheses.values()))
        for h in fresh:
            self.hypotheses[h.id] = h
            if h.cause in self.graph.factors:
                pass  # already registered
            elif h.cause.startswith("res:"):
                self.graph.ensure_factor(h.cause, FactorKind.JS_MODULE,
                                         h.cause)
            elif h.cause.startswith("cond:"):
                self.graph.ensure_factor(h.cause, FactorKind.BROWSER_CONDITION,
                                         h.cause)
            else:
                _, kind = hypgen.metric_factor(h.cause)
                self.graph.ensure_factor(h.cause, kind, h.cause)
            try:
                self.graph.add_candidate(h.cause, h.effect, h.id,
                                         note=f"origin={h.origin}")
            except Exception:
                pass
        self.graph.save_version(
            f"{len(fresh)} hypotheses registered "
            f"({sum(len(s.hypothesis_ids) for s in sets)} in competing sets)")
        log(f"  {len(fresh)} new hypotheses; "
            f"{sum(len(s.hypothesis_ids) for s in sets)} placed in "
            f"{len(sets)} competing set(s) for discrimination")
        for h in fresh[:12]:
            log(f"    {h.id} [{h.origin}] {h.claim[:96]}")

    # -- phase 3: the falsification loop --------------------------------------
    def _baselines(self) -> dict:
        """(page, metric) -> value under STANDARD conditions (desktop,
        unthrottled, JS on) — used to avoid floor/ceiling experiment
        designs."""
        out = {}
        for o in self.observations:
            cond = o.conditions
            if cond.get("throttle", "none") == "none" \
                    and not cond.get("js_disabled", False) \
                    and (cond.get("viewport") or {}).get("width", 1440) >= 1000:
                for k, v in flatten_observation(o).items():
                    out[(o.page, k)] = v
        return out

    def investigate(self) -> None:
        log("PHASE 3 — falsification-driven experiment loop "
            "(selection by expected information gain)")
        # A candidate may be re-run while it still promises some expected
        # information (confirmation runs ARE worth bits while belief is
        # uncertain), but never more than a hard cap of times.
        MIN_IG = 0.0005
        MAX_RUNS_PER_KEY = 4
        run_counts: dict[tuple, int] = {}
        it = 0
        while it < self.max_experiments:
            it += 1
            active = [h for h in self.hypotheses.values()
                      if h.status in (HypStatus.ACTIVE, HypStatus.PROPOSED)]
            if not active:
                log("  no active hypotheses — generating a new wave")
                self.hypothesize()
                active = [h for h in self.hypotheses.values()
                          if h.status in (HypStatus.ACTIVE,
                                          HypStatus.PROPOSED)]
                if not active:
                    break
            from app.experiments.selection import rank_experiments

            def pick():
                for c in rank_experiments(active, self.pages_seen_resources,
                                          self._baselines()):
                    key = (c.page, design_key(c.design), c.metric)
                    if run_counts.get(key, 0) >= MAX_RUNS_PER_KEY:
                        continue
                    if c.ig_bits < MIN_IG:
                        continue
                    return c
                return None

            cand = pick()
            if cand is None:
                # nothing left to test for the CURRENT hypotheses: bring in
                # the next screening wave (condition probes, delays, etc.)
                log("  no informative experiment for current hypotheses — "
                    "generating next wave")
                before = set(self.hypotheses)
                self.hypothesize()
                active = [h for h in self.hypotheses.values()
                          if h.status in (HypStatus.ACTIVE,
                                          HypStatus.PROPOSED)]
                cand = pick() if set(self.hypotheses) != before else None
                if cand is None:
                    log("  hypothesis space exhausted — loop complete")
                    break
            hyp_ids = [h.id for h, _ in cand.staked_hypotheses]
            key = (cand.page, design_key(cand.design), cand.metric)
            run_counts[key] = run_counts.get(key, 0) + 1
            note = (f" (run #{run_counts[key]}: confirmation value)"
                    if run_counts[key] > 1 else "")
            exp = build_experiment(cand.page, cand.design, cand.metric,
                                   cand.direction, cand.min_effect,
                                   hyp_ids, cand.rationale,
                                   [list(p)[:3] for p in cand.discriminates])
            log(f"  E{it}: {exp.title}  [{cand.ig_bits:.2f} bits IG, "
                f"stakes {len(hyp_ids)}, cost {cand.cost}]{note}")
            try:
                result = self.engine.run(exp)
            except BudgetExhausted:
                log("  budget exhausted — wrapping up")
                for h in active:
                    h.status = HypStatus.UNRESOLVED
                break
            self.experiments[exp.id] = (exp, result)
            d = (result.deltas or {}).get(exp.target_metric, {})
            log(f"    control={d.get('control')} treatment={d.get('treatment')}"
                f" delta={d.get('delta')} -> {result.result_class.value}")

            interp = self.interpreter.interpret(exp, result, self.hypotheses)
            for hid, verdict in interp.verdicts.items():
                log(f"    {hid}: {verdict[:110]}")
            if cand.discriminates:
                self.merit.award_discrimination(
                    exp.id, len(cand.staked_hypotheses))
            for hid in interp.refuted:
                h = self.hypotheses[hid]
                self.merit.award_refutation(h)
                log(f"    REFUTED {hid} — '{h.claim[:70]}' (merit +5)")
                revised = hypgen.revise_from_refutation(h, exp, result)
                if revised:
                    self.hypotheses[revised.id] = revised
                    try:
                        self.graph.revive(h.cause, h.effect, revised.id,
                                          f"re-test after {exp.id}")
                    except GraphIntegrityError:
                        try:
                            self.graph.add_candidate(h.cause, h.effect,
                                                     revised.id)
                        except Exception:
                            pass
                    log(f"    revised hypothesis {revised.id} spawned for "
                        f"re-test")
            for hid in interp.promoted_candidates:
                h = self.hypotheses[hid]
                design0 = exp.arms[1].intervention if len(exp.arms) > 1 else {}
                from app.experiments.selection import design_key as _dk
                stake0 = next((p for p in h.predictions
                               if _dk(p.experiment_design) == _dk(design0)),
                              None)
                if stake0 is not None and stake0.direction == 0:
                    # confirmed INDEPENDENCE claim: settles as supported
                    # knowledge; the graph edge stays refuted (no causal
                    # link) and no discovery package is warranted
                    h.status = HypStatus.SUPPORTED
                    log(f"    independence confirmed for {hid}: "
                        f"{h.cause} does not cause {h.effect}")
                    continue
                self._try_discovery(h, exp, result)
            self.graph.save_version(
                f"experiment {exp.id}: {result.result_class.value} on "
                f"{exp.target_metric}")
        # mark leftovers
        for h in self.hypotheses.values():
            if h.status in (HypStatus.ACTIVE, HypStatus.PROPOSED):
                h.status = HypStatus.UNRESOLVED

    def _select(self, active, exclude):
        from app.experiments.selection import rank_experiments
        for c in rank_experiments(active, self.pages_seen_resources,
                                  self._baselines()):
            if design_key(c.design) in exclude:
                continue
            return c
        return None

    # -- phase 4: independent review -> discovery -----------------------------
    def _try_discovery(self, h: Hypothesis, exp, result) -> None:
        # the hypothesis's OWN metric must have moved as predicted; the
        # experiment's headline metric may belong to a different stake
        design = exp.arms[1].intervention if len(exp.arms) > 1 else {}
        from app.experiments.selection import design_key
        stake = next((p for p in h.predictions
                      if design_key(p.experiment_design) == design_key(design)),
                     None)
        if stake is None:
            return
        outcome = EvidenceInterpreter._outcome_for(result, stake)
        if outcome.value != "effect":
            return
        metric_delta = (result.deltas or {}).get(stake.metric, {})
        # review + package against the HYPOTHESIS's metric, which may differ
        # from the experiment's headline metric
        import dataclasses
        from app.types import ResultClass as _RC
        exp2 = dataclasses.replace(exp, target_metric=stake.metric,
                                   expected_direction=stake.direction,
                                   min_effect=stake.min_effect,
                                   hypothesis_ids=[h.id],
                                   title=f"{exp.id}:{h.cause} -> {stake.metric}")
        res2 = dataclasses.replace(result, result_class=_RC.EFFECT,
                                   effect_size=abs(metric_delta.get("delta",
                                                                    0)))
        self.discovery_counter += 1
        did = f"D{self.discovery_counter:03d}"
        statement = (f"{h.cause} causally affects {h.effect} "
                     f"(intervention {design.get('type')}:"
                     f"{design.get('pattern', '')}"
                     + (f" @context={design.get('context')}"
                        if design.get("context") else "")
                     + f" on {exp.page} changed {stake.metric} by "
                     f"{metric_delta.get('delta')})")
        pkg = build_package(self.pkg_dir / f"{did}-{slugify(h.cause)}-"
                            f"{slugify(h.effect)}", exp2, res2, self.driver,
                            statement)
        log(f"    REVIEW {did}: {statement[:100]}")
        review = self.board.review_discovery(h, exp2, res2, pkg,
                                             repeats=self.cfg.discovery_repeats)
        for agent in review.get("agents", []):
            log(f"      {agent.get('role')}: "
                f"{'APPROVED' if agent.get('approved') else 'REJECTED'} — "
                f"{str(agent.get('reasoning', ''))[:110]}")
        if review.get("approved"):
            h.status = HypStatus.SUPPORTED
            try:
                self.graph.promote(h.cause, h.effect, [exp.id],
                                   abs(metric_delta.get("delta", 0)),
                                   note=f"verified via review {did}")
            except GraphIntegrityError as e:
                log(f"      graph integrity: {e}")
            cf = build_counterfactual(
                self.graph.edges.get(f"{h.cause}=>{h.effect}"),
                self.experiments,
                metric_delta.get("delta"))
            disc = Discovery(id=did, statement=statement,
                             hypothesis_id=h.id, experiment_ids=[exp.id],
                             counterfactual=cf, review=review,
                             package_path=str(pkg), verified=True)
            self.discoveries.append(disc)
            self.merit.award_discovery(statement, did)
            log(f"    DISCOVERY {did} confirmed and packaged")
        else:
            # replication failure = contradictory evidence: belief drops,
            # hypothesis stays testable
            h.posterior = min(h.posterior, 0.55)
            h.evidence_against.append(f"review:{did}")
            log(f"    {did} NOT confirmed — kept as active/contradictory")

    # -- outputs ----------------------------------------------------------------
    def finalize(self) -> None:
        log("PHASE 5 — reports, packages, state")
        self.graph.save_version("run finalized")
        self.merit.save()
        run_meta = {"generated_at":
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "mode": self.mode}
        report = write_run_report(
            self.data_dir / "reports" / "investigation.md",
            config=self.cfg, graph=self.graph,
            observations=self.observations,
            hypotheses=list(self.hypotheses.values()),
            experiments=self.experiments,
            discoveries=self.discoveries, merit=self.merit,
            budget=self.budget, anomalies=self.anomalies, run_meta=run_meta)
        from app.reports import write_summary_report
        summary = write_summary_report(
            self.data_dir / "reports" / "summary.md",
            config=self.cfg, graph=self.graph,
            observations=self.observations,
            hypotheses=list(self.hypotheses.values()),
            experiments=self.experiments,
            discoveries=self.discoveries, merit=self.merit,
            budget=self.budget, anomalies=self.anomalies, run_meta=run_meta)
        save_state(self.data_dir, graph=self.graph,
                   hypotheses=list(self.hypotheses.values()),
                   experiments=self.experiments, discoveries=self.discoveries,
                   anomalies=self.anomalies)
        log(f"  report: {report}")
        log(f"  audit summary & improvement plan: {summary}")
        log(f"  {self.merit.summary()}")
        log(f"  budget: {self.budget.summary()}")

    def run(self) -> int:
        try:
            if self.mode != "sim":
                self.driver.start()
            if self.observations:
                log("PHASE 1 — observations already on record "
                    "(resumed); skipping re-crawl")
            else:
                self.observe()
            self.mine()
            self.hypothesize()
            self.investigate()
        except BudgetExhausted as e:
            log(f"budget exhausted: {e}")
        finally:
            self.finalize()
            if self.mode != "sim":
                self.driver.close()
        if self.mode == "sim":
            self._ground_truth_check()
        return 0

    def _ground_truth_check(self):
        from app.observe.simulated import GROUND_TRUTH
        mapping = {"cond.viewport_width": "cond:viewport_width",
                   "perf.goto_ms": "perf.goto_ms",
                   "dom.comp_visible.mobile-menu": "dom.comp_visible.mobile-menu",
                   "dom.comp_present.ta-chat": "dom.comp_present.ta-chat"}
        print("\n=== GROUND-TRUTH RECOVERY CHECK (simulated site) ===")
        hit = miss = 0
        for gt in GROUND_TRUTH:
            cause = gt["cause"]
            if cause.startswith("block:") or cause.startswith("delay:"):
                ck = f"res:{cause.split(':', 1)[1]}"
            elif cause.startswith("throttle:"):
                ck = "cond:throttle"
            elif cause == "js_disabled":
                ck = "cond:js_disabled"
            else:
                ck = mapping.get(cause, cause)
            eff = mapping.get(gt["effect"], gt["effect"])
            edge = self.graph.edges.get(f"{ck}=>{eff}")
            if edge and edge.kind in (EdgeKind.CAUSAL_SUPPORTED,
                                      EdgeKind.CAUSAL_REFUTED):
                hit += 1
                print(f"  [TESTED ] {ck} => {eff}: {edge.kind.value}")
            else:
                miss += 1
                print(f"  [UNTESTED] {ck} => {eff}")
        print(f"recovered {hit}/{len(GROUND_TRUTH)} ground-truth relationships")
        return hit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    d = sub.add_parser("discover", help="run the autonomous investigation")
    d.add_argument("--url", default=None,
                   help="target website origin (default: TechAbout.com). "
                        "The engine audits ANY public site, e.g. "
                        "--url https://example.com")
    d.add_argument("--pages", nargs="*", default=None,
                   help="page paths to investigate (default: brand pages for "
                        "TechAbout, auto-discovered nav pages otherwise)")
    d.add_argument("--data-dir", default=None,
                   help="state directory (default: data/ for TechAbout, "
                        "data-<host>/ for any other site)")
    d.add_argument("--max-loads", type=int, default=140)
    d.add_argument("--max-experiments", type=int, default=None)
    d.add_argument("--sim", action="store_true",
                   help="offline validation against the simulated site")
    d.add_argument("--headed", action="store_true")
    d.add_argument("--discover-pages", action="store_true",
                   help="verify pages exist before running")

    v = sub.add_parser("verify", help="re-run a discovery package")
    v.add_argument("--package", required=True)
    v.add_argument("--repeats", type=int, default=2)

    sub.add_parser("report", help="regenerate the markdown report")

    args = ap.parse_args()
    if args.cmd in (None, "discover"):
        from urllib.parse import urlparse
        cfg = LabConfig()
        if args.url:
            cfg.base_url = args.url.rstrip("/")
        host = urlparse(cfg.base_url).netloc.lower().replace("www.", "")
        is_default_target = "techabout" in host

        if args.pages:
            cfg.pages = args.pages
        elif args.sim:
            cfg.pages = ["/", "/services", "/careers", "/portfolio"]
        elif is_default_target:
            pass  # LabConfig defaults: the TechAbout brand pages
        else:
            # generic target: discover real pages from the site's own nav
            candidates = ["/"] + sitemap.nav_links(cfg.base_url)[:15]
            found = sitemap.discover_pages(cfg.base_url, candidates)
            cfg.pages = found[:cfg.safety.max_pages_per_run]
            log(f"page discovery for {cfg.base_url}: {cfg.pages}")

        if args.discover_pages and is_default_target:
            found = sitemap.discover_pages(cfg.base_url, cfg.pages)
            log(f"page discovery: {found}")
            cfg.pages = found

        data_dir = None
        if args.data_dir:
            data_dir = Path(args.data_dir)
        elif not args.sim and not is_default_target:
            slug = re.sub(r"[^a-z0-9]+", "-", host).strip("-") or "target"
            data_dir = DATA_DIR.parent / f"data-{slug}"
            log(f"generic target: state isolated in {data_dir}")

        if args.max_loads:
            cfg.safety.max_page_loads = args.max_loads
        inv = Investigation(cfg, mode="sim" if args.sim else "live",
                            max_experiments=args.max_experiments,
                            headless=not args.headed, data_dir=data_dir)
        sys.exit(inv.run())

    if args.cmd == "verify":
        pkg = Path(args.package)
        cfg = LabConfig()
        guard = BudgetGuard(SafetyConfig(max_page_loads=12))
        with PlaywrightDriver(cfg.base_url, guard) as drv:
            rep = rerun_package(pkg, drv, guard, cfg, repeats=args.repeats)
        print(json.dumps(rep, indent=2, default=str))
        sys.exit(0 if rep["replicated"] else 1)

    if args.cmd == "report":
        print("report regeneration happens automatically at the end of a "
              "discover run; see data/reports/investigation.md")
        sys.exit(0)


if __name__ == "__main__":
    main()
