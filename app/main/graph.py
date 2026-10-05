"""The causal dependency model — a versioned graph whose edges strictly
separate *observed correlation* from *experimentally supported causation*
(req. 2) and which never deletes a refuted relationship (req. 9).

Edge lifecycle:

    CORRELATION        observational co-variation only; can NEVER be promoted
                       without intervention evidence (enforced in promote()).
    CAUSAL_CANDIDATE   an active hypothesis is testing this edge.
    CAUSAL_SUPPORTED   controlled intervention + independent verification.
    CAUSAL_REFUTED     intervention showed no effect — preserved forever,
                       together with the contradictory evidence.

Every mutation is journalled: data/graph/ledger.jsonl is append-only and
data/graph/vNNN.json holds full snapshots so any historical version can be
reconstructed and audited.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from ..types import Factor, FactorKind, stable_hash


class EdgeKind(str, Enum):
    CORRELATION = "correlation"
    CAUSAL_CANDIDATE = "causal_candidate"
    CAUSAL_SUPPORTED = "causal_supported"
    CAUSAL_REFUTED = "causal_refuted"


@dataclass
class Edge:
    cause: str
    effect: str
    kind: EdgeKind
    strength: float = 0.0          # |corr| or effect size
    support: int = 0               # number of supporting observations
    evidence: list = field(default_factory=list)     # experiment ids
    hypothesis_id: Optional[str] = None
    note: str = ""
    history: list = field(default_factory=list)      # audit trail

    @property
    def id(self) -> str:
        return f"{self.cause}=>{self.effect}"

    def to_dict(self) -> dict:
        d = {"cause": self.cause, "effect": self.effect,
             "kind": self.kind.value, "strength": round(self.strength, 4),
             "support": self.support, "evidence": self.evidence,
             "hypothesis_id": self.hypothesis_id, "note": self.note,
             "history": self.history}
        return d


class GraphIntegrityError(Exception):
    pass


class CausalGraph:
    def __init__(self, root: Path):
        self.root = root
        self.graph_dir = root / "graph"
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.factors: dict[str, Factor] = {}
        self.edges: dict[str, Edge] = {}
        self.version = 0
        self._load_latest()

    # -- persistence --------------------------------------------------------
    def _load_latest(self) -> None:
        ledger = self.graph_dir / "ledger.jsonl"
        if ledger.exists():
            lines = [json.loads(l) for l in ledger.read_text(
                encoding="utf-8").splitlines() if l.strip()]
            if lines:
                self.version = lines[-1]["version"]
        snapshots = sorted(self.graph_dir.glob("v*.json"))
        if snapshots:
            data = json.loads(snapshots[-1].read_text(encoding="utf-8"))
            self.version = data["version"]
            self.factors = {k: Factor.from_dict(v)
                            for k, v in data["factors"].items()}
            self.edges = {e["cause"] + "=>" + e["effect"]: Edge(
                cause=e["cause"], effect=e["effect"],
                kind=EdgeKind(e["kind"]), strength=e["strength"],
                support=e["support"], evidence=e["evidence"],
                hypothesis_id=e["hypothesis_id"], note=e["note"],
                history=e["history"]) for e in data["edges"]}

    def save_version(self, reason: str) -> int:
        self.version += 1
        payload = {
            "version": self.version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": reason,
            "factors": {k: f.to_dict() for k, f in self.factors.items()},
            "edges": [e.to_dict() for e in self.edges.values()],
        }
        snap = self.graph_dir / f"v{self.version:03d}.json"
        snap.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with open(self.graph_dir / "ledger.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"version": self.version, "reason": reason,
                                 "edges": len(self.edges),
                                 "factors": len(self.factors),
                                 "timestamp": payload["timestamp"]}) + "\n")
        return self.version

    # -- factors ------------------------------------------------------------
    def ensure_factor(self, key: str, kind: FactorKind, label: str,
                      meta: dict | None = None) -> Factor:
        if key not in self.factors:
            self.factors[key] = Factor(key=key, kind=kind, label=label,
                                       meta=meta or {})
        return self.factors[key]

    # -- edges ---------------------------------------------------------------
    def _upsert(self, edge: Edge, event: str) -> None:
        edge.history.append({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime()),
                             "event": event,
                             "kind": edge.kind.value,
                             "note": edge.note[:200]})
        self.edges[edge.id] = edge

    def add_correlation(self, cause: str, effect: str, strength: float,
                        support: int = 1, note: str = "") -> Edge:
        eid = f"{cause}=>{effect}"
        existing = self.edges.get(eid)
        if existing and existing.kind == EdgeKind.CORRELATION:
            # weighted update of observational strength
            n = existing.support
            existing.strength = (existing.strength * n + strength) / (n + 1)
            existing.support += 1
            self._upsert(existing, "correlation_reinforced")
            return existing
        if existing and existing.kind != EdgeKind.CORRELATION:
            return existing  # never downgrade an intervention-backed edge
        edge = Edge(cause=cause, effect=effect, kind=EdgeKind.CORRELATION,
                    strength=strength, support=support, note=note)
        self._upsert(edge, "correlation_recorded")
        return edge

    def add_candidate(self, cause: str, effect: str, hypothesis_id: str,
                      note: str = "") -> Edge:
        eid = f"{cause}=>{effect}"
        existing = self.edges.get(eid)
        if existing and existing.kind in (EdgeKind.CAUSAL_SUPPORTED,
                                          EdgeKind.CAUSAL_REFUTED):
            # already settled by experiment; a new hypothesis may only
            # supersede via revise()
            return existing
        edge = existing or Edge(cause=cause, effect=effect,
                                kind=EdgeKind.CAUSAL_CANDIDATE)
        edge.kind = EdgeKind.CAUSAL_CANDIDATE
        edge.hypothesis_id = hypothesis_id
        edge.note = note or edge.note
        self._upsert(edge, "candidate_registered")
        return edge

    def promote(self, cause: str, effect: str, experiment_ids: list[str],
                effect_size: float, note: str = "") -> Edge:
        """CORRELATION/CANDIDATE -> CAUSAL_SUPPORTED.

        Refuses to promote without intervention evidence: this is where the
        correlation/causation line is actually enforced in code."""
        if not experiment_ids:
            raise GraphIntegrityError(
                f"refusing to promote {cause}=>{effect} to causal without "
                "intervention evidence")
        eid = f"{cause}=>{effect}"
        edge = self.edges.get(eid)
        if edge and edge.kind == EdgeKind.CAUSAL_REFUTED:
            raise GraphIntegrityError(
                f"{eid} was experimentally refuted; promoting it requires a "
                "revising hypothesis to re-test it first (see revive())")
        edge = edge or Edge(cause=cause, effect=effect,
                            kind=EdgeKind.CAUSAL_CANDIDATE)
        edge.kind = EdgeKind.CAUSAL_SUPPORTED
        edge.evidence = list(dict.fromkeys(edge.evidence + experiment_ids))
        edge.strength = effect_size
        edge.note = note or edge.note
        self._upsert(edge, "promoted_to_causal")
        return edge

    def refute(self, cause: str, effect: str, experiment_ids: list[str],
               note: str = "") -> Edge:
        """Mark an edge experimentally refuted. The edge is PRESERVED with its
        contradictory evidence — scientific memory, not deletion (req. 9)."""
        eid = f"{cause}=>{effect}"
        edge = self.edges.get(eid) or Edge(cause=cause, effect=effect,
                                           kind=EdgeKind.CAUSAL_CANDIDATE)
        edge.kind = EdgeKind.CAUSAL_REFUTED
        edge.evidence = list(dict.fromkeys(edge.evidence + experiment_ids))
        edge.note = note or edge.note
        self._upsert(edge, "refuted_by_experiment")
        return edge

    def revive(self, cause: str, effect: str, hypothesis_id: str,
               note: str) -> Edge:
        """A refuted edge may only return to candidacy through an explicit
        revising hypothesis that will be re-tested."""
        eid = f"{cause}=>{effect}"
        edge = self.edges.get(eid)
        if not edge or edge.kind != EdgeKind.CAUSAL_REFUTED:
            raise GraphIntegrityError("revive() applies to refuted edges only")
        edge.kind = EdgeKind.CAUSAL_CANDIDATE
        edge.hypothesis_id = hypothesis_id
        edge.note = f"REVISED: {note}"
        self._upsert(edge, "revived_for_retest")
        return edge

    # -- queries --------------------------------------------------------------
    def correlations_of(self, factor: str) -> list[Edge]:
        return [e for e in self.edges.values()
                if e.kind == EdgeKind.CORRELATION
                and factor in (e.cause, e.effect)]

    def causal_parents(self, factor: str) -> list[Edge]:
        return [e for e in self.edges.values()
                if e.kind == EdgeKind.CAUSAL_SUPPORTED and e.effect == factor]

    def supported_edges(self) -> list[Edge]:
        return [e for e in self.edges.values()
                if e.kind == EdgeKind.CAUSAL_SUPPORTED]

    def refuted_edges(self) -> list[Edge]:
        return [e for e in self.edges.values()
                if e.kind == EdgeKind.CAUSAL_REFUTED]

    def stats(self) -> dict:
        kinds = {}
        for e in self.edges.values():
            kinds[e.kind.value] = kinds.get(e.kind.value, 0) + 1
        return {"factors": len(self.factors), "edges": len(self.edges),
                "by_kind": kinds, "version": self.version}

    def to_mermaid(self) -> str:
        style = {EdgeKind.CORRELATION: "corr",
                 EdgeKind.CAUSAL_CANDIDATE: "cand",
                 EdgeKind.CAUSAL_SUPPORTED: "causal",
                 EdgeKind.CAUSAL_REFUTED: "refuted"}
        lines = ["graph LR"]
        for e in self.edges.values():
            arrow = {"corr": "-.->", "cand": "-->", "causal": "==>",
                     "refuted": "--x"}[style[e.kind]]
            lines.append(f"  {self._nid(e.cause)} {arrow}|{style[e.kind]}| "
                         f"{self._nid(e.effect)}")
        lines.append("  classDef causal stroke:#0a7,stroke-width:3px;")
        lines.append("  classDef refuted stroke:#c22,stroke-dasharray:4;")
        lines.append("  classDef corr stroke:#888,stroke-dasharray:2;")
        return "\n".join(lines)

    @staticmethod
    def _nid(key: str) -> str:
        return "N" + stable_hash(key)
