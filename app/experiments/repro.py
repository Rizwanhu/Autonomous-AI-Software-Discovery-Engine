"""Discovery reproducibility (req. 11).

Every significant discovery is emitted as a self-contained experiment
package:

    package/
      experiment.json      machine-readable definition (arms, metric, thresholds)
      environment.json     browser/driver/OS + throttle profiles
      evidence.json        control->changed->expected->actual->evidence
      screenshots/         control & treatment captures
      VERIFICATION.md      independent re-run results (written by verifier)
      rerun.py             one-command reproduction

`rerun_package` re-executes the definition against a FRESH driver and
reports whether the result class replicates. The IndependentVerifier uses
exactly this path — it never reads the original outcome before running.
"""
from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path

from ..types import (Arm, ArmResult, Experiment, ExperimentResult,
                     ResultClass)
from .engine import ExperimentEngine, STANDARD_CONDITIONS, build_experiment


def _environment(driver) -> dict:
    env = {"python": sys.version.split()[0],
           "platform": platform.platform(),
           "driver": type(driver).__name__}
    try:
        import playwright
        env["playwright"] = getattr(playwright, "__version__", "unknown")
        b = getattr(driver, "_browser", None)
        env["chromium"] = b.version if b else "n/a"
    except Exception:
        pass
    from .engine import MAX_DELAY_MS  # noqa: F401  (documented cap)
    from ..observe.driver import THROTTLE_PROFILES
    env["throttle_profiles"] = {k: list(v) for k, v in
                                THROTTLE_PROFILES.items()}
    return env


def build_package(out_dir: Path, experiment: Experiment,
                  result: ExperimentResult, driver,
                  statement: str) -> Path:
    pkg = Path(out_dir)
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "screenshots").mkdir(exist_ok=True)

    exp_json = {
        "id": experiment.id,
        "statement_under_test": statement,
        "page": experiment.page,
        "arms": [{"name": a.name, "intervention": a.intervention}
                 for a in experiment.arms],
        "target_metric": experiment.target_metric,
        "expected_direction": experiment.expected_direction,
        "min_effect": experiment.min_effect,
        "standard_conditions": STANDARD_CONDITIONS,
        "hypothesis_ids": experiment.hypothesis_ids,
        "design_rationale": experiment.design_rationale,
    }
    (pkg / "experiment.json").write_text(json.dumps(exp_json, indent=2),
                                         encoding="utf-8")
    (pkg / "environment.json").write_text(
        json.dumps(_environment(driver), indent=2), encoding="utf-8")

    evidence = {
        "record": "Control -> Variable Changed -> Expected Effect -> "
                  "Actual Effect -> Evidence",
        "control": _arm_summary(result.control, experiment.target_metric),
        "treatment": [_arm_summary(t, experiment.target_metric)
                      for t in result.treatments],
        "expected_effect": f"{experiment.expected_direction:+d} x >= "
                           f"{experiment.min_effect} on "
                           f"{experiment.target_metric}",
        "actual_effect": result.deltas.get(experiment.target_metric, {}),
        "result_class": result.result_class.value,
        "deltas_all_metrics": result.deltas,
        "screenshots": result.control.screenshots +
                       [s for t in result.treatments for s in t.screenshots],
    }
    (pkg / "evidence.json").write_text(json.dumps(evidence, indent=2,
                                                  default=str),
                                        encoding="utf-8")

    for shot in evidence["screenshots"]:
        if shot and Path(shot).exists():
            try:
                shutil.copy(shot, pkg / "screenshots" / Path(shot).name)
            except OSError:
                pass

    rerun = pkg / "rerun.py"
    rerun.write_text(_RERUN_TEMPLATE.format(pkg=pkg), encoding="utf-8")
    return pkg


def _arm_summary(arm: ArmResult, metric: str) -> dict:
    vals = [m[metric] for m in arm.measurements if metric in m]
    return {"arm": arm.arm, "metric": metric,
            "repeats": len(arm.measurements),
            "values": vals,
            "mean": round(sum(vals) / len(vals), 3) if vals else None,
            "logs": arm.logs}


def load_experiment_from_package(pkg_dir: Path) -> Experiment:
    d = json.loads((pkg_dir / "experiment.json").read_text(encoding="utf-8"))
    arms = [Arm(name=a["name"], intervention=a["intervention"])
            for a in d["arms"]]
    return Experiment(id=d["id"] + "-rerun", title="re-run " + d["id"],
                      page=d["page"], arms=arms,
                      target_metric=d["target_metric"],
                      expected_direction=d["expected_direction"],
                      min_effect=d["min_effect"],
                      hypothesis_ids=d.get("hypothesis_ids", []))


def rerun_package(pkg_dir: Path, driver, budget, config,
                  repeats: int | None = None) -> dict:
    """Re-execute the packaged experiment on a fresh engine. Returns the
    replicated result class + deltas and whether it matches the original."""
    pkg_dir = Path(pkg_dir)
    exp = load_experiment_from_package(pkg_dir)
    orig = json.loads((pkg_dir / "evidence.json").read_text(encoding="utf-8"))

    cfg = config
    if repeats:
        import dataclasses
        cfg = dataclasses.replace(config, repeats_per_arm=repeats)

    engine = ExperimentEngine(driver, budget, cfg,
                              evidence_dir=pkg_dir / "rerun-evidence")
    result = engine.run(exp)
    replicated = result.result_class.value == orig.get("result_class")
    out = {
        "package": str(pkg_dir),
        "original_result_class": orig.get("result_class"),
        "rerun_result_class": result.result_class.value,
        "replicated": replicated,
        "rerun_delta": (result.deltas or {}).get(exp.target_metric),
        "original_delta": orig.get("actual_effect"),
    }
    (pkg_dir / "rerun-result.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


_RUN_TEMPLATE_RESERVED = None


_RERUN_TEMPLATE = '''#!/usr/bin/env python3
"""One-command reproduction of this discovery package.

    python rerun.py            (from the package directory, with the
                                causal-lab project on sys.path)

Re-executes experiment.json against the live target with a fresh browser
and compares the outcome with evidence.json.
"""
import sys
from pathlib import Path

ROOT = Path(r"{pkg}").resolve().parents[2]
sys.path.insert(0, str(ROOT))

from causal_lab.config import LabConfig, SafetyConfig
from causal_lab.observe.budget import BudgetGuard
from causal_lab.observe.driver import PlaywrightDriver
from causal_lab.experiments.repro import rerun_package

if __name__ == "__main__":
    cfg = LabConfig(pages=["/"])
    guard = BudgetGuard(SafetyConfig(max_page_loads=12))
    with PlaywrightDriver(cfg.base_url, guard) as drv:
        report = rerun_package(Path(r"{pkg}"), drv, guard, cfg, repeats=2)
    print("replicated:", report["replicated"])
    print("original  :", report["original_result_class"],
          report["original_delta"])
    print("re-run    :", report["rerun_result_class"], report["rerun_delta"])
    sys.exit(0 if report["replicated"] else 1)
'''
