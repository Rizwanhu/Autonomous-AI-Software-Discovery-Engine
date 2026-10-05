"""Counterfactual analysis (req. 5).

For each experimentally supported causal relationship the engine emits a
structured what-if report. The rule enforced here: every counterfactual
statement is labelled SIMULATED and must cite the measured intervention
evidence it extrapolates from; everything else is labelled OBSERVED.
"""
from __future__ import annotations

from typing import Optional


def build_counterfactual(edge, experiments: dict, delta: Optional[dict] = None
                         ) -> dict:
    """edge: a causal graph Edge of kind CAUSAL_SUPPORTED.
    experiments: {exp_id: (Experiment, ExperimentResult)} providing evidence.
    delta: the measured treatment-vs-control delta for the effect metric."""
    ev_ids = edge.evidence or []
    exp_summaries = []
    for eid in ev_ids:
        found = experiments.get(eid)
        if found:
            exp, res = found
            d = (res.deltas or {}).get(exp.target_metric, {})
            exp_summaries.append({
                "experiment": eid,
                "intervention": exp.arms[1].intervention
                if len(exp.arms) > 1 else {},
                "page": exp.page,
                "result_class": res.result_class.value,
                "control": d.get("control"),
                "treatment": d.get("treatment"),
                "delta": d.get("delta"),
            })

    measured = delta or (exp_summaries[0]["delta"] if exp_summaries else None)

    return {
        "relationship": {"cause": edge.cause, "effect": edge.effect,
                         "measured_effect": measured,
                         "evidence_experiments": ev_ids},
        "observed": {
            "kind": "OBSERVED",
            "statement":
                f"Under the control condition, {edge.effect} took its "
                f"baseline value; under the intervention on {edge.cause}, "
                f"the measured change was {measured}. This was directly "
                f"observed in {len(ev_ids)} controlled experiment(s).",
        },
        "counterfactual": {
            "kind": "SIMULATED",
            "statement":
                f"HAD {edge.cause} been altered on a page where it was left "
                f"untouched, {edge.effect} would be expected to change by "
                f"approximately {measured} (extrapolated from measured "
                f"effect size, NOT observed on that page).",
            "caveats": [
                "extrapolation assumes the same browser/device/network class "
                "as the experiment context",
                "site deploys can change the relationship at any time; "
                "re-run the evidence experiments to re-check",
            ],
        },
        "validation_experiment": {
            "kind": "PROPOSED",
            "statement":
                f"Apply the same intervention to a NEW page/condition and "
                f"measure {edge.effect}; a replication within tolerance "
                f"supports generalisation, a failure bounds the scope.",
            "design": exp_summaries[0]["intervention"] if exp_summaries
            else None,
        },
        "evidence_chain": exp_summaries,
    }
