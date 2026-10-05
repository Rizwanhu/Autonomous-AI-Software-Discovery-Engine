"""Offline unit tests for the Causal Lab science engine.

Runs entirely against the simulated site (no network, no browser) so the
falsification machinery itself is what is under test. Execute with:

    python tests/test_engine.py        # standalone
    python -m pytest tests/test_engine.py -q
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main.graph import (CausalGraph, EdgeKind,
                                     GraphIntegrityError)
from app.config import LabConfig, SafetyConfig
from app.experiments.engine import (ExperimentEngine,
                                           build_experiment)
from app.experiments.repro import build_package, rerun_package
from app.experiments.selection import (design_key, info_gain,
                                              rank_experiments)
from app.observe.budget import BudgetGuard
from app.observe.simulated import SimulatedSiteDriver
from app.reasoning.falsify import EvidenceInterpreter
from app.types import (Arm, HypStatus, Hypothesis, Prediction,
                              ResultClass)


def _graph(tmp=None):
    return CausalGraph(Path(tempfile.mkdtemp()) if tmp is None else tmp)


# ---------------------------------------------------------------------------
# causal graph integrity (req. 2, 9)
# ---------------------------------------------------------------------------

def test_correlation_cannot_be_promoted_without_evidence():
    g = _graph()
    g.add_correlation("res.total_bytes", "perf.lcp_ms", 0.91, support=5)
    try:
        g.promote("res.total_bytes", "perf.lcp_ms", [], 1.0)
        raise AssertionError("promotion without evidence must be refused")
    except GraphIntegrityError:
        pass
    assert g.edges["res.total_bytes=>perf.lcp_ms"].kind == EdgeKind.CORRELATION


def test_refuted_edges_are_preserved_not_deleted():
    g = _graph()
    g.add_candidate("res:chat.js", "dom.comp_visible.ta-chat", "hyp-1")
    g.promote("res:chat.js", "dom.comp_visible.ta-chat", ["exp-1"], 1.0)
    g.refute("res:chat.js", "dom.comp_visible.ta-chat", ["exp-2"],
             "later contradiction")
    edge = g.edges["res:chat.js=>dom.comp_visible.ta-chat"]
    assert edge.kind == EdgeKind.CAUSAL_REFUTED
    assert "exp-2" in edge.evidence           # contradictory evidence kept
    assert len(edge.history) >= 3             # full audit trail


def test_refuted_edge_requires_explicit_revive_to_retest():
    g = _graph()
    g.add_candidate("res:x.js", "dom.a", "hyp-1")
    g.refute("res:x.js", "dom.a", ["exp-1"])
    try:
        g.promote("res:x.js", "dom.a", ["exp-9"], 1.0)
        raise AssertionError("promoting a refuted edge must be refused")
    except GraphIntegrityError:
        pass
    g.revive("res:x.js", "dom.a", "hyp-2", "revised mechanism")
    assert g.edges["res:x.js=>dom.a"].kind == EdgeKind.CAUSAL_CANDIDATE


def test_versioning_appends_snapshots_and_ledger():
    g = _graph()
    g.add_correlation("a", "b", 0.7)
    v1 = g.save_version("first")
    g.refute("a", "b", ["exp-1"])
    v2 = g.save_version("second")
    assert (v1, v2) == (1, 2)
    assert (g.graph_dir / "v001.json").exists()
    assert (g.graph_dir / "v002.json").exists()
    ledger = (g.graph_dir / "ledger.jsonl").read_text().splitlines()
    assert len(ledger) == 2 and "second" in ledger[1]


# ---------------------------------------------------------------------------
# experiment engine + interventions (req. 3)
# ---------------------------------------------------------------------------

def _sim_engine():
    cfg = LabConfig()
    cfg.repeats_per_arm = 2
    drv = SimulatedSiteDriver()
    budget = BudgetGuard(SafetyConfig(max_page_loads=10**6))
    return drv, ExperimentEngine(drv, budget, cfg,
                                 evidence_dir=Path(tempfile.mkdtemp()))


def test_block_intervention_engagement_and_effect():
    drv, eng = _sim_engine()
    exp = build_experiment("/", {"type": "block", "pattern": "chat.js"},
                           "dom.comp_visible.ta-chat", -1, 1.0, ["h1"],
                           "necessity probe")
    res = eng.run(exp)
    assert res.result_class == ResultClass.EFFECT
    d = res.deltas["dom.comp_visible.ta-chat"]
    assert d["delta"] == -1.0


def test_failed_engagement_is_error_not_no_effect():
    drv, eng = _sim_engine()
    exp = build_experiment("/", {"type": "block", "pattern": "nonexistent.js"},
                           "dom.comp_visible.ta-chat", -1, 1.0, ["h1"],
                           "pattern matches nothing on the page")
    res = eng.run(exp)
    assert res.result_class == ResultClass.ERROR


def test_control_and_treatment_share_context():
    drv, eng = _sim_engine()
    exp = build_experiment("/", {"type": "block", "pattern": "header-account.js",
                                 "context": {"viewport": {"width": 390,
                                                          "height": 844}}},
                           "dom.comp_visible.mobile-menu", -1, 1.0, ["h1"],
                           "mobile-context probe")
    res = eng.run(exp)
    assert res.result_class == ResultClass.EFFECT
    # both arms ran at mobile width; only the module differs
    assert res.control.measurements[0]["cond.viewport_width"] == 390
    assert res.treatments[0].measurements[0]["cond.viewport_width"] == 390
    assert res.deltas["dom.comp_visible.mobile-menu"]["delta"] == -1.0


# ---------------------------------------------------------------------------
# Bayesian interpretation + falsification (req. 1)
# ---------------------------------------------------------------------------

def _interp():
    cfg = LabConfig()
    g = _graph()
    return cfg, g, EvidenceInterpreter(cfg, g)


def test_no_effect_refutes_and_spawns_revision_path():
    cfg, g, interp = _interp()
    drv, eng = _sim_engine()
    h = Hypothesis(id="h1", claim="cart.js drives ta-chat",
                   cause="res:cart.js", effect="dom.comp_visible.ta-chat",
                   mechanism="m", prior=0.32, posterior=0.32,
                   predictions=[Prediction(
                       experiment_design={"type": "block",
                                          "pattern": "cart.js"},
                       metric="dom.comp_visible.ta-chat",
                       direction=-1, min_effect=1.0)])
    exp = build_experiment("/", {"type": "block", "pattern": "cart.js"},
                           "dom.comp_visible.ta-chat", -1, 1.0, ["h1"], "t")
    res = eng.run(exp)
    out = interp.interpret(exp, res, {"h1": h})
    assert res.result_class == ResultClass.NO_EFFECT
    # one ambiguous no-effect weakens but does not yet refute (noise
    # tolerance); the replicated second no-effect is decisive
    res2 = eng.run(exp)
    interp.interpret(exp, res2, {"h1": h})
    assert h.posterior < cfg.refute_threshold
    assert h.status == HypStatus.REFUTED
    edge = g.edges.get("res:cart.js=>dom.comp_visible.ta-chat")
    assert edge and edge.kind == EdgeKind.CAUSAL_REFUTED


def test_pinned_boolean_no_effect_is_uninformative():
    cfg, g, interp = _interp()
    drv, eng = _sim_engine()
    # ta-chat-panel is hidden (0) under desktop conditions: a decrease
    # prediction cannot be tested there
    h = Hypothesis(id="h1", claim="x", cause="res:cart.js",
                   effect="dom.comp_visible.ta-chat-panel",
                   mechanism="m", prior=0.32, posterior=0.32,
                   predictions=[Prediction(
                       experiment_design={"type": "block",
                                          "pattern": "cart.js"},
                       metric="dom.comp_visible.ta-chat-panel",
                       direction=-1, min_effect=1.0)])
    exp = build_experiment("/", {"type": "block", "pattern": "cart.js"},
                           "dom.comp_visible.ta-chat-panel", -1, 1.0,
                           ["h1"], "t")
    res = eng.run(exp)
    interp.interpret(exp, res, {"h1": h})
    assert h.posterior == 0.32  # unchanged: floor-pinned, no update


def test_strong_evidence_accelerates_support():
    cfg, g, interp = _interp()
    drv, eng = _sim_engine()
    h = Hypothesis(id="h1", claim="chat.js injects ta-chat",
                   cause="res:chat.js", effect="dom.comp_visible.ta-chat",
                   mechanism="m", prior=0.32, posterior=0.32,
                   predictions=[Prediction(
                       experiment_design={"type": "block",
                                          "pattern": "chat.js"},
                       metric="dom.comp_visible.ta-chat",
                       direction=-1, min_effect=1.0)])
    exp = build_experiment("/", {"type": "block", "pattern": "chat.js"},
                           "dom.comp_visible.ta-chat", -1, 1.0, ["h1"], "t")
    res = eng.run(exp)
    interp.interpret(exp, res, {"h1": h})
    # boolean flip replicated across repeats -> sharp likelihood
    assert h.posterior > 0.5


# ---------------------------------------------------------------------------
# information-gain selection (req. 6, 7)
# ---------------------------------------------------------------------------

def _hyp(i, design, metric, direction, post=0.5):
    return Hypothesis(id=f"h{i}", claim=f"c{i}", cause=f"cause{i}",
                      effect=metric, mechanism="m", posterior=post,
                      prior=post,
                      predictions=[Prediction(experiment_design=design,
                                              metric=metric,
                                              direction=direction,
                                              min_effect=1.0)])


def test_discriminating_experiment_has_higher_ig():
    design_a = {"type": "block", "pattern": "chat.js"}
    design_b = {"type": "block", "pattern": "irrelevant.js"}
    # two hypotheses that DISAGREE about design_a
    h1 = _hyp(1, design_a, "dom.comp_visible.ta-chat", -1, 0.5)
    h2 = _hyp(2, design_a, "dom.comp_visible.ta-chat", 0, 0.5)  # null claim
    # one hypothesis staking design_b only
    h3 = _hyp(3, design_b, "dom.comp_visible.newsletter-form", -1, 0.5)
    pages = {"chat.js": ["/"], "irrelevant.js": ["/"]}

    class C:  # minimal candidate shim
        def __init__(self, design):
            self.design = design
            self.metric = "dom.comp_visible.ta-chat"
            self.staked_hypotheses = []

    from app.experiments.selection import Candidate
    ca = Candidate(page="/", design=design_a,
                   metric="dom.comp_visible.ta-chat", direction=-1,
                   min_effect=1.0)
    cb = Candidate(page="/", design=design_b,
                   metric="dom.comp_visible.newsletter-form", direction=-1,
                   min_effect=1.0)
    ig_a = info_gain(ca, [h1, h2, h3])
    ig_b = info_gain(cb, [h1, h2, h3])
    assert ig_a > ig_b, ("experiment splitting two disagreeing hypotheses "
                         "must carry more expected information")


# ---------------------------------------------------------------------------
# reproducibility packages (req. 11)
# ---------------------------------------------------------------------------

def test_package_rerun_replicates():
    drv, eng = _sim_engine()
    exp = build_experiment("/", {"type": "block", "pattern": "tailwind.css"},
                           "dom.doc_height_px", -1, 300.0, ["h1"], "layout")
    res = eng.run(exp)
    pkg = build_package(Path(tempfile.mkdtemp()) / "pkg", exp, res, drv,
                        "tailwind.css causally affects doc height")
    assert (pkg / "experiment.json").exists()
    assert (pkg / "evidence.json").exists()
    assert (pkg / "rerun.py").exists()
    cfg = LabConfig()
    with SimulatedSiteDriver() as d2:  # fresh driver = independence
        rep = rerun_package(pkg, d2, BudgetGuard(SafetyConfig(
            max_page_loads=10**6)), cfg, repeats=1)
    assert rep["replicated"] is True
    assert rep["original_result_class"] == rep["rerun_result_class"]


# ---------------------------------------------------------------------------

def main():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
