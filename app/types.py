"""Core scientific data model.

Everything the engine produces is one of these record types, serialisable to
JSON, so any conclusion can be traced back to raw evidence.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------

def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


# ---------------------------------------------------------------------------
# Factors — the observable variables of the causal model (req. 2)
# ---------------------------------------------------------------------------

class FactorKind(str, Enum):
    PAGE = "page"
    DOM_COMPONENT = "dom_component"
    JS_MODULE = "js_module"
    CSS_RESOURCE = "css_resource"
    NETWORK_REQUEST = "network_request"
    THIRD_PARTY = "third_party"
    BROWSER_CONDITION = "browser_condition"   # viewport, device, UA
    LOAD_SEQUENCE = "load_sequence"
    USER_ACTION = "user_action"
    RUNTIME_ERROR = "runtime_error"
    PERF_METRIC = "perf_metric"
    A11Y = "accessibility"
    VISUAL_STATE = "visual_state"


@dataclass
class Factor:
    """A node of the causal graph: one observable, nameable thing."""
    key: str                      # e.g. "js:chat.js", "cond:viewport.width"
    kind: FactorKind
    label: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "Factor":
        d = dict(d)
        d["kind"] = FactorKind(d["kind"])
        return Factor(**d)


# ---------------------------------------------------------------------------
# Observations — passive measurements of the real site (no intervention)
# ---------------------------------------------------------------------------

@dataclass
class ResourceRecord:
    url: str
    kind: str                      # document | script | stylesheet | font | image | xhr | other
    status: Optional[int]          # HTTP status or None if blocked/failed
    size_bytes: int
    duration_ms: Optional[float]
    third_party: bool
    error: Optional[str] = None

    @property
    def basename(self) -> str:
        tail = self.url.split("?")[0].rstrip("/").split("/")[-1]
        return tail or self.url


@dataclass
class Observation:
    """One passive measurement of a page under stated conditions."""
    id: str
    page: str                                  # "/" or "/services"
    conditions: dict                           # viewport, throttle, js, cache...
    resources: list[ResourceRecord] = field(default_factory=list)
    dom: dict = field(default_factory=dict)    # node counts, component visibility
    console: list[dict] = field(default_factory=list)
    performance: dict = field(default_factory=dict)   # fcp/lcp/load ms
    a11y: dict = field(default_factory=dict)
    visual_hash: Optional[str] = None
    screenshot: Optional[str] = None           # path to PNG
    timestamp: str = field(default_factory=now_iso)

    # -- derived metrics used by experiments -------------------------------
    def metric(self, name: str) -> float:
        """Flat metric accessor: searches dom/performance/a11y/conditions."""
        for source in (self.performance, self.dom, self.a11y):
            if name in source:
                v = source[name]
                return float(v) if isinstance(v, bool) is False else float(bool(v))
        if name.startswith("cond."):
            return float(self.conditions.get(name[5:], 0) or 0)
        if name.startswith("res."):
            attr = name[4:]
            return float(sum(1 for r in self.resources
                             if attr == "count"
                             or (attr == "failed" and (r.status or 0) >= 400)
                             or (attr == "third_party" and r.third_party)))
        raise KeyError(name)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resources"] = [asdict(r) for r in self.resources]
        return d


# ---------------------------------------------------------------------------
# Hypotheses — falsifiable causal claims with prior/posterior mass (req. 1,4)
# ---------------------------------------------------------------------------

class HypStatus(str, Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    SUPPORTED = "supported"          # survived severe tests + review
    REFUTED = "refuted"              # preserved, never deleted (req. 9)
    SUPERSEDED = "superseded"        # replaced by a revision
    UNRESOLVED = "unresolved"        # budget ran out before a verdict


@dataclass
class Prediction:
    """What a hypothesis expects an experiment to show."""
    experiment_design: dict          # the intervention spec this prediction is about
    metric: str
    direction: int                   # +1 increase, -1 decrease, 0 no change
    min_effect: float                # effect size that counts as EVIDENCE
    rationale: str = ""


@dataclass
class Hypothesis:
    id: str
    claim: str                                   # "X causes Y under conditions C"
    cause: str                                   # factor key
    effect: str                                  # factor key / metric name
    mechanism: str                               # proposed how
    origin: str = "anomaly"                      # anomaly | correlation | revision | screening
    parent_id: Optional[str] = None              # hypothesis this revises
    status: HypStatus = HypStatus.PROPOSED
    prior: float = 0.5
    posterior: float = 0.5
    predictions: list[Prediction] = field(default_factory=list)
    evidence_for: list[str] = field(default_factory=list)    # experiment ids
    evidence_against: list[str] = field(default_factory=list)
    falsification_criteria: str = ""
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["predictions"] = [asdict(p) for p in self.predictions]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Hypothesis":
        d = dict(d)
        d["status"] = HypStatus(d["status"])
        d["predictions"] = [Prediction(**p) for p in d.get("predictions", [])]
        return Hypothesis(**d)


# ---------------------------------------------------------------------------
# Experiments — controlled interventions (req. 3)
# ---------------------------------------------------------------------------

@dataclass
class Arm:
    name: str                        # "control" or descriptive treatment name
    intervention: dict               # intervention spec applied ({} for control)
    repeats: int = 1
    conditions: dict = field(default_factory=dict)   # base conditions for all arms


@dataclass
class ArmResult:
    arm: str
    measurements: list[dict] = field(default_factory=list)   # metric dicts per repeat
    screenshots: list[str] = field(default_factory=list)
    logs: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)   # serialized Observation


class ResultClass(str, Enum):
    EFFECT = "effect"                # metric moved as predicted
    OPPOSITE = "opposite"            # metric moved significantly the other way
    NO_EFFECT = "no_effect"          # |delta| below threshold
    ERROR = "error"                  # experiment itself failed to run


@dataclass
class Experiment:
    id: str
    title: str
    page: str
    arms: list[Arm]
    target_metric: str
    expected_direction: int          # predicted sign of (treatment - control)
    min_effect: float                # threshold for a meaningful effect
    hypothesis_ids: list[str] = field(default_factory=list)
    design_rationale: str = ""       # why this experiment, what it discriminates
    discriminates: list[str] = field(default_factory=list)  # hypothesis ids it separates
    status: str = "designed"         # designed | run | failed
    created_at: str = field(default_factory=now_iso)


@dataclass
class ExperimentResult:
    experiment_id: str
    control: ArmResult
    treatments: list[ArmResult]
    deltas: dict = field(default_factory=dict)     # arm -> {metric: delta}
    result_class: ResultClass = ResultClass.NO_EFFECT
    effect_size: float = 0.0
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["result_class"] = self.result_class.value
        return d


# ---------------------------------------------------------------------------
# Discoveries — conclusions that survived independent verification (req. 11)
# ---------------------------------------------------------------------------

@dataclass
class Discovery:
    id: str
    statement: str                    # "Blocking X causally changes Y (evidence E...)"
    hypothesis_id: str
    experiment_ids: list[str]
    counterfactual: Optional[dict] = None
    review: Optional[dict] = None     # review-board record
    package_path: Optional[str] = None
    verified: bool = False
    created_at: str = field(default_factory=now_iso)


# ---------------------------------------------------------------------------
# Merit ledger — the reward signal: disproof scores highest (req. 1)
# ---------------------------------------------------------------------------

@dataclass
class MeritEntry:
    points: float
    reason: str
    ref: str                          # hypothesis / experiment / discovery id
    timestamp: str = field(default_factory=now_iso)


# ---------------------------------------------------------------------------
# Deserialisation helpers (run resumption)
# ---------------------------------------------------------------------------

def observation_from_dict(d: dict) -> Observation:
    resources = [ResourceRecord(**r) for r in d.get("resources", [])]
    return Observation(
        id=d["id"], page=d["page"], conditions=d.get("conditions", {}),
        resources=resources, dom=d.get("dom", {}),
        console=d.get("console", []), performance=d.get("performance", {}),
        a11y=d.get("a11y", {}), visual_hash=d.get("visual_hash"),
        screenshot=d.get("screenshot"), timestamp=d.get("timestamp", ""))


def _arm_from_dict(d: dict) -> Arm:
    return Arm(name=d["name"], intervention=d.get("intervention", {}),
               repeats=d.get("repeats", 1),
               conditions=d.get("conditions", {}))


def _arm_result_from_dict(d: dict) -> ArmResult:
    return ArmResult(arm=d["arm"],
                     measurements=d.get("measurements", []),
                     screenshots=d.get("screenshots", []),
                     logs=d.get("logs", []),
                     observations=d.get("observations", []))


def experiment_from_dict(d: dict) -> Experiment:
    return Experiment(
        id=d["id"], title=d.get("title", ""), page=d["page"],
        arms=[_arm_from_dict(a) for a in d.get("arms", [])],
        target_metric=d["target_metric"],
        expected_direction=d["expected_direction"],
        min_effect=d["min_effect"],
        hypothesis_ids=d.get("hypothesis_ids", []),
        design_rationale=d.get("design_rationale", ""),
        discriminates=d.get("discriminates", []),
        status=d.get("status", "run"))


def result_from_dict(d: dict) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=d["experiment_id"],
        control=_arm_result_from_dict(d["control"]),
        treatments=[_arm_result_from_dict(t) for t in d.get("treatments", [])],
        deltas=d.get("deltas", {}),
        result_class=ResultClass(d.get("result_class", "no_effect")),
        effect_size=d.get("effect_size", 0.0),
        timestamp=d.get("timestamp", ""))
