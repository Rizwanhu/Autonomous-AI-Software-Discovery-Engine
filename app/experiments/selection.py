"""Information-gain experiment selection (req. 7) and root-cause
discrimination (req. 6).

Classical Bayesian experiment design: given active hypotheses with current
belief masses, and each hypothesis's predicted outcome distribution over the
result classes {EFFECT, NO_EFFECT, OPPOSITE} for a candidate experiment, the
expected information gain of that experiment is

    IG(E) = H(belief) - sum_o P(o) * H(belief | o)

measured in bits. The engine picks the experiment with the highest IG per
unit cost — i.e. the single safe experiment that is expected to eliminate the
largest number of competing explanations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..types import HypStatus, Prediction, ResultClass

OUTCOMES = [ResultClass.EFFECT, ResultClass.NO_EFFECT, ResultClass.OPPOSITE]

# P(outcome | hypothesis true) when the hypothesis stakes a prediction
P_TRUE_STAKED = {ResultClass.EFFECT: 0.88, ResultClass.NO_EFFECT: 0.08,
                 ResultClass.OPPOSITE: 0.04}
# staking "no effect" (independence claim)
P_TRUE_NULL = {ResultClass.NO_EFFECT: 0.88, ResultClass.EFFECT: 0.06,
               ResultClass.OPPOSITE: 0.06}
# P(outcome | hypothesis false) — predictions of a wrong hypothesis mostly fail
P_FALSE = {ResultClass.EFFECT: 0.38, ResultClass.NO_EFFECT: 0.38,
           ResultClass.OPPOSITE: 0.24}
# hypothesis has no stake in this experiment — outcome shouldn't move it much
P_DONTCARE = {ResultClass.EFFECT: 0.40, ResultClass.NO_EFFECT: 0.50,
              ResultClass.OPPOSITE: 0.10}

COST_UNITS = {"block": 1.0, "viewport": 1.0, "disable_js": 1.0,
              "throttle": 2.2, "delay": 2.6, "mock_api": 1.0}


def design_key(design: dict) -> tuple:
    t = design.get("type")
    if t == "viewport":
        return ("viewport", design.get("width"))
    if t == "throttle":
        return ("throttle", design.get("profile"))
    if t == "delay":
        return ("delay", design.get("pattern"), design.get("delay_ms"))
    return (t, design.get("pattern"))


def prediction_distribution(h, design: dict) -> dict[ResultClass, float] | None:
    """Outcome distribution this hypothesis expects for the given design,
    or None when it has no stake (its belief should barely move)."""
    for p in h.predictions:
        if design_key(p.experiment_design) == design_key(design):
            if p.direction == 0:
                return dict(P_TRUE_NULL)
            base = dict(P_TRUE_STAKED)
            if p.direction < 0:  # expects a decrease -> OPPOSITE means increase
                base = {ResultClass.EFFECT: P_TRUE_STAKED[ResultClass.OPPOSITE],
                        ResultClass.OPPOSITE: P_TRUE_STAKED[ResultClass.EFFECT],
                        ResultClass.NO_EFFECT: P_TRUE_STAKED[ResultClass.NO_EFFECT]}
            return base
    return None


def entropy(p: list[float]) -> float:
    return -sum(x * math.log2(x) for x in p if x > 0)


def belief_vector(hypotheses: list) -> list[float]:
    total = sum(h.posterior for h in hypotheses) or 1.0
    return [h.posterior / total for h in hypotheses]


@dataclass
class Candidate:
    page: str
    design: dict
    metric: str
    direction: int
    min_effect: float
    staked_hypotheses: list = field(default_factory=list)   # (hyp, prediction)
    ig_bits: float = 0.0
    cost: float = 1.0
    discriminates: list = field(default_factory=list)       # hyp id pairs
    rationale: str = ""


def build_candidates(hypotheses: list, pages_seen_resources: dict) -> list[Candidate]:
    """Collect unique (page, design, metric) candidates from every active
    hypothesis prediction, with the pages each resource was observed on."""
    by_key: dict[tuple, Candidate] = {}
    for h in hypotheses:
        if h.status not in (HypStatus.ACTIVE, HypStatus.PROPOSED):
            continue
        for p in h.predictions:
            dk = design_key(p.experiment_design)
            pages = pages_seen_resources.get(_pattern_of(p.experiment_design),
                                             ["/"])
            for page in pages[:1]:  # test on the first page it appears on
                key = (page, dk, p.metric)
                if key not in by_key:
                    by_key[key] = Candidate(
                        page=page, design=p.experiment_design, metric=p.metric,
                        direction=p.direction, min_effect=p.min_effect,
                        cost=COST_UNITS.get(p.experiment_design.get("type"), 1.0))
                by_key[key].staked_hypotheses.append((h, p))
    return list(by_key.values())


def _pattern_of(design: dict) -> str:
    return design.get("pattern", "")


def info_gain(cand: Candidate, hypotheses: list) -> float:
    """Expected entropy reduction over the ACTIVE hypothesis set, in bits."""
    belief = belief_vector(hypotheses)
    h0 = entropy(belief)
    if h0 == 0:
        return 0.0
    # per-hypothesis outcome distributions for this candidate
    dists = []
    for h in hypotheses:
        d = prediction_distribution(h, cand.design)
        dists.append(d)

    p_out = {o: 0.0 for o in OUTCOMES}
    posteriors = {o: [] for o in OUTCOMES}
    for o in OUTCOMES:
        for h, d in zip(hypotheses, dists):
            if d is not None:
                p_o_given_h = d[o] if d is not None else P_DONTCARE[o]
                # Bayes: P(h|o) ∝ P(h) * P(o|h) where "h false" uses P_FALSE
                num = h.posterior * p_o_given_h
                den = h.posterior * p_o_given_h + \
                    (1 - h.posterior) * P_FALSE[o]
                posteriors[o].append(num / den if den > 0 else 0.0)
            else:
                posteriors[o].append(h.posterior)  # no stake: unchanged
        p_out[o] = sum(b * (d[o] if d else P_DONTCARE[o])
                       for b, d in zip(belief, dists))
    total = sum(p_out.values()) or 1.0
    expected_h = 0.0
    for o in OUTCOMES:
        if p_out[o] <= 0:
            continue
        post = posteriors[o]
        s = sum(post) or 1.0
        expected_h += (p_out[o] / total) * entropy([x / s for x in post])
    return h0 - expected_h


def discrimination_report(cand: Candidate, hypotheses: list) -> list[tuple]:
    """Pairs of hypotheses (staked ANYWHERE on this design, not just on the
    candidate's own metric) whose predicted outcome distributions maximally
    disagree — the root-cause distinguishing power (req. 6). A single run of
    this design adjudicates every staked hypothesis on its own metric."""
    pairs = []
    stakes = [(h, p) for h in hypotheses if h.status in (HypStatus.ACTIVE,
                                                        HypStatus.PROPOSED)
              for p in h.predictions
              if design_key(p.experiment_design) == design_key(cand.design)]
    for i, (h1, p1) in enumerate(stakes):
        for h2, p2 in stakes[i + 1:]:
            d1 = prediction_distribution(h1, cand.design)
            d2 = prediction_distribution(h2, cand.design)
            if d1 and d2:
                div = sum(abs(d1[o] - d2[o]) for o in OUTCOMES) / 2
                if div >= 0.4:
                    pairs.append((h1.id, h2.id, round(div, 2),
                                  f"{p1.metric} vs {p2.metric}"))
    return sorted(pairs, key=lambda t: -t[2])[:6]


SUPPORT_THRESHOLD = 0.85  # must match LabConfig.support_threshold


def _review_bonus(cand: Candidate, hypotheses: list) -> float:
    """Decision value beyond entropy: an experiment that can push a staked
    hypothesis across the independent-review threshold unlocks a concrete
    action (review -> package -> discovery), so it earns a bonus. Pure
    information gain would otherwise always prefer fresh exploration over
    confirming near-threshold hypotheses."""
    for h, p in cand.staked_hypotheses:
        if not (0.60 <= h.posterior < SUPPORT_THRESHOLD):
            continue
        if design_key(p.experiment_design) == design_key(cand.design):
            return 0.05
    return 0.0


def rank_experiments(hypotheses: list, pages_seen_resources: dict,
                     baselines: dict | None = None
                     ) -> list[Candidate]:
    """Rank candidates by expected information gain per unit cost.

    `baselines` maps (page, metric) -> observed value under STANDARD
    conditions; a boolean metric sitting at floor/ceiling there makes the
    experiment nearly uninformative (no room to move), so it is heavily
    deprioritised unless the design's context changes that baseline."""
    cands = build_candidates(hypotheses, pages_seen_resources)
    active = [h for h in hypotheses
              if h.status in (HypStatus.ACTIVE, HypStatus.PROPOSED)]
    for c in cands:
        c.ig_bits = round(info_gain(c, active), 4)
        bonus = _review_bonus(c, hypotheses)
        if bonus:
            c.ig_bits = round(c.ig_bits + bonus, 4)
        c.discriminates = discrimination_report(c, hypotheses)
        notes = []
        if bonus:
            notes.append("near-threshold confirmation: review unlock")
        if baselines:
            base = baselines.get((c.page, c.metric))
            ctx = c.design.get("context") or {}
            if base is not None and c.metric.startswith("dom.comp_visible") \
                    and base in (0.0, 1.0) and not ctx:
                c.ig_bits = round(c.ig_bits * 0.15, 4)
                notes.append(f"floor/ceiling risk (baseline={base})")
        c.rationale = (
            f"expected information gain {c.ig_bits:.2f} bits over "
            f"{len(active)} active hypotheses; stakes: "
            + ", ".join(f"{h.id}({p.metric},dir={p.direction:+d})"
                        for h, p in c.staked_hypotheses[:4])
            + (f"; NOTE: {'; '.join(notes)}" if notes else ""))
    cands.sort(key=lambda c: (-(c.ig_bits / c.cost), c.cost))
    return cands


def select_next_experiment(hypotheses: list, pages_seen_resources: dict,
                           exclude_ids: set | None = None) -> Candidate | None:
    ranked = rank_experiments(hypotheses, pages_seen_resources)
    for c in ranked:
        if not exclude_ids or (c.page, design_key(c.design), c.metric) \
                not in exclude_ids:
            return c
    return None
