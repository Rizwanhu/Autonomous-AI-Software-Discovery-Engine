"""Evidence interpretation, falsification and merit (req. 1, 9).

The EvidenceInterpreter updates every staked hypothesis after each
experiment using the same likelihood model the selector used for planning
(no post-hoc fiddling), then applies the falsification-first rules:

    posterior < refute_threshold  -> REFUTED  (highest scientific merit!)
    posterior > support_threshold -> send to the independent review board
    otherwise                     -> remains ACTIVE, more evidence needed

The MeritLedger is the reward signal: it pays for DISPROOF, pays for
discoveries that survive independent verification, and pays nothing for
re-confirming what is already known.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..experiments.selection import prediction_distribution
from ..types import (Experiment, ExperimentResult, HypStatus, Hypothesis,
                     MeritEntry, ResultClass)
from ..main.graph import CausalGraph, EdgeKind

P_FALSE = {ResultClass.EFFECT: 0.38, ResultClass.NO_EFFECT: 0.38,
           ResultClass.OPPOSITE: 0.24}


def _likelihood(outcome: ResultClass, stake, sharp: bool = False) -> float:
    """P(outcome | hypothesis TRUE), mirroring the selector's planning model.

    A staked prediction treats 'moved as predicted' as success; a null
    (independence) prediction treats NO_EFFECT as success. When arm repeats
    agree exactly (sharp=True) the measurement is near-deterministic and
    likelihoods tighten accordingly."""
    if stake.direction == 0:
        p = {ResultClass.NO_EFFECT: 0.97, ResultClass.EFFECT: 0.015,
             ResultClass.OPPOSITE: 0.015} if sharp else \
            {ResultClass.NO_EFFECT: 0.88, ResultClass.EFFECT: 0.06,
             ResultClass.OPPOSITE: 0.06}
    else:
        p = {ResultClass.EFFECT: 0.97, ResultClass.NO_EFFECT: 0.02,
             ResultClass.OPPOSITE: 0.01} if sharp else \
            {ResultClass.EFFECT: 0.88, ResultClass.NO_EFFECT: 0.08,
             ResultClass.OPPOSITE: 0.04}
    return p[outcome]


@dataclass
class Interpretation:
    experiment_id: str
    verdicts: dict = field(default_factory=dict)     # hyp_id -> note
    refuted: list[str] = field(default_factory=list)
    promoted_candidates: list[str] = field(default_factory=list)


class EvidenceInterpreter:
    """Independent agent: sees experiment + hypotheses, updates beliefs."""

    def __init__(self, config, graph: CausalGraph):
        self.config = config
        self.graph = graph

    def interpret(self, experiment: Experiment, result: ExperimentResult,
                  hypotheses: dict[str, Hypothesis]) -> Interpretation:
        interp = Interpretation(experiment_id=experiment.id)
        if result.result_class == ResultClass.ERROR:
            for hid in experiment.hypothesis_ids:
                if hid in hypotheses:
                    interp.verdicts[hid] = (
                        "experiment ERROR (intervention failed to engage) — "
                        "no belief update, evidence discarded")
            return interp

        design = _design_of(experiment)
        for h in hypotheses.values():
            if h.status not in (HypStatus.ACTIVE, HypStatus.PROPOSED):
                continue
            stake = None
            for p in h.predictions:
                from ..experiments.selection import design_key
                if design_key(p.experiment_design) == design_key(design):
                    stake = p
                    break
            if stake is None:
                continue  # no stake, no update
            # evaluate the outcome on THIS hypothesis's own metric+direction
            outcome = self._outcome_for(result, stake)
            if outcome == ResultClass.ERROR:
                interp.verdicts[h.id] = (
                    f"metric {stake.metric} unmeasurable in this experiment — "
                    "no update")
                continue
            if outcome == ResultClass.NO_EFFECT and \
                    self._pinned_against(result, stake):
                interp.verdicts[h.id] = (
                    f"{stake.metric} pinned against the predicted direction "
                    f"in the control arm — NO_EFFECT here is uninformative, "
                    "no update")
                continue
            sharp = self._strong_evidence(result, stake)
            d = _likelihood(outcome, stake, sharp)
            num = h.posterior * d
            den = num + (1 - h.posterior) * P_FALSE[outcome]
            old = h.posterior
            h.posterior = round(num / den, 4) if den > 0 else h.posterior
            if h.posterior > old:
                h.evidence_for.append(experiment.id)
            else:
                h.evidence_against.append(experiment.id)
            interp.verdicts[h.id] = (
                f"its own metric {stake.metric} (dir {stake.direction:+d}, "
                f"min {stake.min_effect}) -> observed "
                f"{outcome.value}; posterior {old:.3f} -> {h.posterior:.3f}")

            if h.posterior < self.config.refute_threshold:
                h.status = HypStatus.REFUTED
                interp.refuted.append(h.id)
                self._record_refutation(h, experiment, result)
            elif h.posterior >= self.config.support_threshold:
                interp.promoted_candidates.append(h.id)

        # update graph candidate edges with fresh evidence either way
        for h in hypotheses.values():
            if h.status in (HypStatus.ACTIVE, HypStatus.PROPOSED) and \
                    experiment.id in h.evidence_for + h.evidence_against:
                try:
                    self.graph.add_candidate(h.cause, h.effect, h.id,
                                             note=f"tested by {experiment.id}")
                except Exception:
                    pass
        return interp

    @staticmethod
    def _pinned_against(result: ExperimentResult, stake) -> bool:
        """True when the metric is boolean, constant in the control arm, and
        pinned in the direction the hypothesis needs it to move: the
        treatment cannot push it further, so NO_EFFECT carries no evidence.

        A decrease prediction at value 1, or an increase prediction at 0,
        still has head-room and IS informative."""
        metric = stake.metric
        if not metric.startswith("dom.comp"):
            return False
        vals = [m[metric] for m in result.control.measurements
                if metric in m]
        if not vals or not all(v in (0.0, 1.0) for v in vals):
            return False
        v = vals[0]
        if stake.direction < 0 and v == 0.0:
            return True   # wants a decrease, already at floor
        if stake.direction > 0 and v == 1.0:
            return True   # wants an increase, already at ceiling
        if stake.direction == 0 and v == 0.0:
            return True   # null claim: can't separate 'independent' from 'pinned'
        return False

    @staticmethod
    def _strong_evidence(result: ExperimentResult, stake) -> bool:
        """Internally-replicated evidence: an experiment already runs
        repeat loads per arm. If EVERY control/treatment repeat pair moves
        beyond the minimum effect in the predicted direction — or the metric
        is boolean and flips identically in all repeats — the measurement is
        strong enough to justify sharp likelihoods."""
        metric = stake.metric
        cvals = [m[metric] for m in result.control.measurements
                 if metric in m]
        tvals = [m[metric] for m in result.treatments[0].measurements
                 if metric in m] if result.treatments else []
        if not cvals or not tvals:
            return False
        both = cvals + tvals
        if all(v in (0.0, 1.0) for v in both):
            return len(set(cvals)) == 1 and len(set(tvals)) == 1 and \
                cvals[0] != tvals[0]
        pairs = min(len(cvals), len(tvals))
        return all((t - c) * stake.direction >= stake.min_effect
                   for c, t in zip(cvals, tvals)) and pairs > 0

    @staticmethod
    def _outcome_for(result: ExperimentResult, stake) -> ResultClass:
        d = (result.deltas or {}).get(stake.metric)
        if d is None:
            return ResultClass.ERROR
        moved = d["delta"] * stake.direction
        if moved >= stake.min_effect:
            return ResultClass.EFFECT
        if moved <= -stake.min_effect:
            return ResultClass.OPPOSITE
        return ResultClass.NO_EFFECT

    def _record_refutation(self, h: Hypothesis, experiment: Experiment,
                           result: ExperimentResult):
        try:
            self.graph.refute(h.cause, h.effect, [experiment.id],
                              note=f"refuted via {experiment.id} "
                                   f"({result.result_class.value}); "
                                   f"hypothesis {h.id} preserved as refuted")
        except Exception:
            pass


def _design_of(experiment: Experiment) -> dict:
    return experiment.arms[1].intervention if len(experiment.arms) > 1 else {}


class MeritLedger:
    """The reward function. Disproof is the highest-paying act."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: list[MeritEntry] = []
        # cumulative across sessions: scientific merit is a career, not a run
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.entries = [MeritEntry(points=e["points"],
                                            reason=e["reason"], ref=e["ref"],
                                            timestamp=e.get("at", ""))
                                 for e in data.get("entries", [])]
            except Exception:
                pass

    def award(self, points: float, reason: str, ref: str):
        self.entries.append(MeritEntry(points=points, reason=reason, ref=ref))

    def award_refutation(self, h: Hypothesis):
        self.award(5.0, f"REFUTED own hypothesis {h.id}: '{h.claim[:90]}' — "
                        "disproof is the core scientific act", h.id)

    def award_severe_test_survived(self, h: Hypothesis, experiment_id: str):
        self.award(2.0, f"hypothesis {h.id} survived a designed-to-fail test "
                        f"{experiment_id}", h.id)

    def award_discovery(self, statement: str, discovery_id: str):
        self.award(8.0, f"discovery survived independent verification: "
                        f"{statement[:90]}", discovery_id)

    def award_discrimination(self, experiment_id: str, n: int):
        self.award(3.0, f"experiment {experiment_id} discriminated between "
                        f"{n} competing explanations", experiment_id)

    def total(self) -> float:
        return sum(e.points for e in self.entries)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"total": self.total(),
             "entries": [{"points": e.points, "reason": e.reason,
                          "ref": e.ref, "at": e.timestamp}
                         for e in self.entries]}, indent=2),
            encoding="utf-8")

    def summary(self) -> str:
        refuted = sum(1 for e in self.entries if "REFUTED" in e.reason)
        disc = sum(1 for e in self.entries if "discriminated" in e.reason)
        return (f"merit={self.total():.1f} (refutations={refuted}, "
                f"discriminating-experiments={disc}, entries={len(self.entries)})")
