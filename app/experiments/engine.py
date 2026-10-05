"""Controlled experiment engine (req. 3).

Each experiment is a two-arm controlled trial on our own instrumented browser:

    control   : page loaded under standard conditions
    treatment : identical conditions + exactly ONE intervention

For every run we record Control -> Variable Changed -> Expected Effect ->
Actual Effect -> Evidence. The engine also verifies that the intervention
actually ENGAGED (e.g. the blocked resource really failed, the delayed
resource really took longer) — an experiment whose intervention silently
failed is reported as ERROR, never as NO_EFFECT.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import LabConfig
from ..observe.budget import BudgetGuard, BudgetExhausted
from ..observe.instrument import flatten_observation
from ..types import (Arm, ArmResult, Experiment, ExperimentResult,
                     ResultClass, new_id)

STANDARD_CONDITIONS = {"viewport": {"width": 1440, "height": 900},
                       "throttle": "none", "label": "standard"}

MAX_DELAY_MS = 6000


class UnsafeDesignError(Exception):
    pass


def validate_design(design: dict, page_url_path: str) -> None:
    t = design.get("type")
    if t == "block":
        p = design.get("pattern", "")
        if not p or len(p) < 4:
            raise UnsafeDesignError("block pattern too broad")
        if p in page_url_path:
            raise UnsafeDesignError("refusing to block the document itself")
    elif t == "delay":
        if design.get("delay_ms", 0) > MAX_DELAY_MS:
            raise UnsafeDesignError("delay exceeds cap")
    elif t in ("viewport", "throttle", "disable_js", "mock_api"):
        pass
    else:
        raise UnsafeDesignError(f"unknown intervention type {t!r}")


def _conditions_with(design: dict, base: dict) -> dict:
    """Interventions that are context-level (viewport/throttle/js) modify the
    arm's conditions; routing-level ones (block/delay/mock) stay as routes.
    A design may carry an explicit `context` (e.g. run a block experiment at
    mobile viewport) which is merged last."""
    cond = dict(base)
    ctx = design.get("context") or {}
    cond.update(ctx)
    t = design.get("type")
    if t == "viewport":
        cond["viewport"] = {"width": design["width"],
                            "height": design.get("height", 900)}
    elif t == "throttle":
        cond["throttle"] = design.get("profile", "slow3g")
    elif t == "disable_js":
        cond["js_disabled"] = True
    return cond


def _routing_steps(design: dict) -> list[dict]:
    t = design.get("type")
    if t in ("block", "delay", "mock_api"):
        return [design]
    return []


class ExperimentEngine:
    def __init__(self, driver, budget: BudgetGuard, config: LabConfig,
                 evidence_dir: Path):
        self.driver = driver
        self.budget = budget
        self.config = config
        self.evidence_dir = evidence_dir
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.log: list[dict] = []

    # ------------------------------------------------------------------
    def run(self, experiment: Experiment) -> ExperimentResult:
        base = dict(STANDARD_CONDITIONS)
        design = (experiment.arms[1].intervention
                  if len(experiment.arms) > 1 else {})
        validate_design(design, experiment.page)

        # a design's `context` (e.g. mobile viewport) applies to BOTH arms:
        # the control must differ from the treatment by the intervention
        # alone, never by the context it runs in
        shared = {**base, **(design.get("context") or {})}

        exp_dir = self.evidence_dir / experiment.id
        exp_dir.mkdir(parents=True, exist_ok=True)

        control = self._run_arm(experiment, "control", {}, shared, exp_dir)
        treatment = self._run_arm(experiment, design.get("type", "treatment"),
                                  design, shared, exp_dir)

        deltas = self._deltas(control, treatment)
        engaged = self._intervention_engaged(design, treatment)

        result = ExperimentResult(experiment_id=experiment.id,
                                  control=control, treatments=[treatment],
                                  deltas=deltas)
        if not engaged:
            result.result_class = ResultClass.ERROR
            result.effect_size = 0.0
        else:
            result.result_class, result.effect_size = self._classify(
                experiment, deltas)

        experiment.status = "run"
        self._write_package_data(exp_dir, experiment, result)
        self.log.append({"experiment_id": experiment.id, "title": experiment.title,
                         "result_class": result.result_class.value,
                         "effect_size": round(result.effect_size, 3),
                         "engaged": engaged, "ts": time.strftime("%H:%M:%S")})
        return result

    # ------------------------------------------------------------------
    def _run_arm(self, experiment: Experiment, arm_name: str, design: dict,
                 base: dict, exp_dir: Path) -> ArmResult:
        steps = _routing_steps(design)
        cond = _conditions_with(design, base)
        cond = {k: v for k, v in cond.items() if k != "label"}
        cond["arm"] = arm_name
        repeats = self.config.repeats_per_arm
        ar = ArmResult(arm=arm_name)
        for i in range(repeats):
            shot = str(exp_dir / f"{arm_name}-r{i}.png")
            try:
                obs = self.driver.load(experiment.page, cond, steps,
                                       screenshot_path=shot)
                ar.measurements.append(flatten_observation(obs))
                ar.observations.append(obs.to_dict())
                ar.screenshots.append(shot)
                ar.logs.append({"repeat": i,
                                "console": obs.console,
                                "resources": [(r.basename, r.status,
                                               r.duration_ms, r.error)
                                              for r in obs.resources]})
            except BudgetExhausted:
                raise
            except Exception as e:
                ar.logs.append({"repeat": i, "error": str(e)[:300]})
        return ar

    @staticmethod
    def _mean(metrics: list[dict], key: str) -> float | None:
        vals = [m[key] for m in metrics if key in m]
        return sum(vals) / len(vals) if vals else None

    def _deltas(self, control: ArmResult, treatment: ArmResult) -> dict:
        keys = set()
        for m in list(control.measurements) + list(treatment.measurements):
            keys.update(m.keys())
        out = {}
        for k in sorted(keys):
            c = self._mean(control.measurements, k)
            t = self._mean(treatment.measurements, k)
            if c is not None and t is not None:
                out[k] = {"control": round(c, 3), "treatment": round(t, 3),
                          "delta": round(t - c, 3)}
        return out

    @staticmethod
    def _classify(experiment: Experiment, deltas: dict
                  ) -> tuple[ResultClass, float]:
        metric = experiment.target_metric
        d = deltas.get(metric)
        if not d:
            return ResultClass.ERROR, 0.0
        delta, thresh = d["delta"], experiment.min_effect
        direction = experiment.expected_direction
        moved = delta * direction
        if moved >= thresh:
            return ResultClass.EFFECT, abs(delta)
        if moved <= -thresh:
            return ResultClass.OPPOSITE, -abs(delta)
        return ResultClass.NO_EFFECT, 0.0

    @staticmethod
    def _intervention_engaged(design: dict, treatment: ArmResult) -> bool:
        t = design.get("type")
        if t in (None, "viewport", "throttle", "disable_js"):
            return bool(treatment.measurements)  # context-level: trusted
        pattern = design.get("pattern", "")
        for log in treatment.logs:
            for name, status, dur, err in log.get("resources", []):
                if t == "block" and pattern in name and (err or
                                                         (status or 0) >= 400):
                    return True
                if t == "delay" and pattern in name and dur is not None:
                    if dur >= design.get("delay_ms", 0) * 0.6:
                        return True
                if t == "mock_api" and pattern in name:
                    return True
        return False

    def _write_package_data(self, exp_dir: Path, experiment: Experiment,
                            result: ExperimentResult) -> None:
        (exp_dir / "result.json").write_text(
            json.dumps(result.to_dict(), indent=2, default=str),
            encoding="utf-8")
        (exp_dir / "experiment.json").write_text(
            json.dumps({
                "id": experiment.id, "title": experiment.title,
                "page": experiment.page,
                "arms": [{"name": a.name, "intervention": a.intervention}
                         for a in experiment.arms],
                "target_metric": experiment.target_metric,
                "expected_direction": experiment.expected_direction,
                "min_effect": experiment.min_effect,
                "hypothesis_ids": experiment.hypothesis_ids,
                "design_rationale": experiment.design_rationale,
            }, indent=2), encoding="utf-8")


def build_experiment(page: str, design: dict, metric: str, direction: int,
                     min_effect: float, hypothesis_ids: list[str],
                     rationale: str, discriminates: list[str] | None = None
                     ) -> Experiment:
    arms = [Arm(name="control", intervention={}),
            Arm(name=design.get("type", "treatment"), intervention=design)]
    return Experiment(
        id=new_id("exp"), title=f"{design.get('type')}:{design.get('pattern', '')}"
        f" -> {metric} on {page}", page=page, arms=arms,
        target_metric=metric, expected_direction=direction,
        min_effect=min_effect, hypothesis_ids=list(hypothesis_ids),
        design_rationale=rationale,
        discriminates=list(discriminates or []))
