# Causal Lab — Autonomous AI Software Science & Causal Discovery Engine

A research-grade autonomous AI system that audits **any public website** by
running a falsification-first scientific investigation into its observable
behaviour. Originally built around [TechAbout.com](https://www.techabout.com)
(the TechAbout brand site — services, portfolio, careers), the engine is now
fully site-agnostic: point it at any origin with `--url` and it discovers
the site's pages, resources, components and conditions by itself.

This is not a testing tool and not a vulnerability scanner. It is a
*scientific instrument* that:

1. **observes** the site passively under varied conditions,
2. **generates falsifiable causal hypotheses** (from anomalies, mined
   correlations, and assumption-free screening probes),
3. **designs and runs controlled experiments** in its own instrumented
   browser,
4. **rewards itself for disproving its own hypotheses**,
5. **revises** refuted hypotheses and re-tests them,
6. **independently verifies** surviving findings (red team + blind
   replication), and
7. **writes a full audit report and an improvement plan** at the end of
   every run.

---

## 1. How to run the project

### 1.1 Prerequisites

- Python 3.11+
- A Chromium browser for Playwright:

```bash
pip install -r requirements.txt
playwright install chromium
```

### 1.2 Basic commands

Everything runs through one CLI, `main.py`:

```bash
# audit the default target (TechAbout.com)
python main.py discover

# audit ANY public website — pages are auto-discovered from its nav
python main.py discover --url https://example.com

# choose specific pages and bound the run
python main.py discover --url https://www.techabout.com --pages / /services /careers

# bound the investigation (defaults: 140 page loads, unlimited experiments)
python main.py discover --max-loads 100 --max-experiments 12

# continue/resume a previous investigation (state is preserved)
python main.py discover --url https://example.com

# watch the browser work (debugging)
python main.py discover --headed

# offline validation against the simulated ground-truth site (no network)
python main.py discover --sim

# reproduce any verified discovery later
python main.py verify --package data/packages/D001-res-tailwind-css-dom-doc-height-px

# unit tests (fully offline, ~30 s)
python tests/test_engine.py
```

> **Windows + Git Bash note:** prefix commands with `MSYS_NO_PATHCONV=1` so
> that page paths like `/services` are not rewritten to `C:/Program Files/...`
> by the shell.

### 1.3 What happens when you run `discover`

```
PHASE 1  Observation     each page is loaded under 4 passive conditions
                          (desktop / mobile viewport / slow-3G / JS-off),
                          capturing resources, DOM components, console,
                          performance, accessibility, screenshots
PHASE 2a Correlation     pairwise metric co-variation is mined across
                          conditions and recorded as CORRELATION edges
                          (never causal by itself)
PHASE 2b Hypothesis      anomaly-driven competing explanations +
                          assumption-free screening probes, each with
                          concrete predictions & falsification criteria
PHASE 3  The loop         while budget remains:
                            select the experiment with the highest expected
                            information gain -> run control+treatment arms ->
                            Bayesian belief update -> refute / confirm /
                            revise -> anything crossing the support
                            threshold goes to review
PHASE 4  Review           RedTeamFalsifier attacks the claim, then an
                          IndependentVerifier blind re-runs the packaged
                          experiment; only survivors become Discoveries
PHASE 5  Reports          investigation.md + summary.md + reproducibility
                          packages + versioned graph + state for resumption
```

### 1.4 Where the outputs live

Per target. TechAbout uses `data/`; any other site gets its own isolated
directory `data-<host>/` (override with `--data-dir`):

```
data/                         (or data-<host>/)
  reports/
    investigation.md            full scientific record (the paper)
    summary.md                  audit summary & improvement plan (the action list)
  packages/D00X-<slug>/        one reproducibility package per verified discovery
    experiment.json              machine-readable experiment definition
    evidence.json                control -> variable -> expected -> actual -> evidence
    environment.json             browser/driver/OS + throttle profiles
    screenshots/                 control & treatment captures
    rerun.py                     one-command reproduction
    VERIFICATION.md              blind verifier's transcript
  graph/
    vNNN.json                    full causal-graph snapshots (append-only history)
    ledger.jsonl                 version ledger with reasons
  observations/                 raw passive observations (JSON + screenshots)
  evidence/<exp-id>/            per-experiment artefacts
  state.json                    resumable run state
  merit.json                    the scientific merit ledger
```

### 1.5 Command reference

| command | flags | meaning |
|---|---|---|
| `discover` | `--url` | target origin; default TechAbout.com, any site works |
| | `--pages` | page paths; default: brand pages for TechAbout, auto-discovered nav pages otherwise |
| | `--data-dir` | override the state directory |
| | `--max-loads` | hard cap on page loads (default 140) |
| | `--max-experiments` | cap on controlled experiments |
| | `--headed` | show the browser window |
| | `--sim` | offline ground-truth validation run |
| | `--discover-pages` | verify pages respond 200 before auditing |
| `verify` | `--package`, `--repeats` | re-run a discovery package and compare |
| `report` | — | reports regenerate at the end of every `discover` |

---

## 2. The two end-of-run reports

### `reports/investigation.md` — the scientific record

The full paper: loop status, every verified discovery with its counterfactual
analysis (explicitly labelled OBSERVED vs SIMULATED), the complete refuted-
hypothesis table (preserved forever, with contradicting evidence), the
controlled-experiment ledger (Control → Variable Changed → Expected →
Actual → Evidence for every experiment), the Mermaid causal graph, detected
anomalies, and an honesty statement.

### `reports/summary.md` — the audit summary & improvement plan

The action-oriented document, written automatically at the end of every run:

1. **Site health issues observed** — failed resources (with page+status),
   runtime/console errors, accessibility gaps (images missing `alt`),
   performance envelope (fastest → slowest navigation and under what
   condition), third-party exposure and page weight, plus every anomaly
   flagged during observation.
2. **Verified causal findings and what they mean for the site** — each
   discovery is translated into practical advice (e.g. a script on the
   DOMContentLoaded critical path → consider async/defer/preload; a layout
   that collapses without a stylesheet → inline critical CSS; a JS-injected
   component → provide a no-JS fallback).
3. **Open questions** — every unresolved hypothesis with its current
   posterior and the *recommended next experiment* to run, ranked by how
   close it is to a verdict.
4. **Beliefs the audit ruled out** — refuted hypotheses, preserved with the
   evidence that killed them.
5. **Experiment quality review (engine self-critique)** — experiments that
   failed to engage and why, floor/ceiling-pinned no-effect runs that carry
   no evidential weight, hypotheses with no evidence yet, and known
   limitations (user-action-gated components need interaction probes).
6. **Recommended next actions** — for the site owner (fix list) and for the
   investigation (budget, pages, re-verify after site changes).

---

## 3. How the science works

### 3.1 The causal model

Factors are observable things — JS modules (`res:chat.js`), CSS builds
(`res:tailwind.css`), third-party resources (`res:css2`), browser conditions
(`cond:viewport_width`, `cond:throttle`, `cond:js_disabled`), performance
metrics, DOM-component presence/visibility, error counts, accessibility
counts. Components are **discovered dynamically**: every element carrying an
`id` is probed, so nothing is hardcoded per site.

Graph edges are strictly typed, and the typing is enforced *in code*:

| kind | meaning | how it can change |
|---|---|---|
| `correlation` | observational co-variation only | `promote()` **refuses** to make it causal without intervention evidence |
| `causal_candidate` | an active hypothesis is testing it | becomes supported or refuted only via experiments |
| `causal_supported` | intervention evidence + independent verification | may be re-opened only by explicit revision |
| `causal_refuted` | intervention showed no effect | **preserved forever** with its contradictory evidence; may only return via `revive()` under a revising hypothesis |

### 3.2 Controlled experiments

Every experiment is a two-arm trial inside our own instrumented Chromium:

```
control   : page under standard conditions (desktop, unthrottled, JS on)
treatment : identical context + exactly ONE intervention
```

Interventions: `block` a resource, `delay` it (load-order manipulation),
`viewport` emulation, network `throttle` (CDP), `disable_js`, `mock_api`
responses, cached vs uncached contexts. A design may carry a `context`
(e.g. run a block experiment *at mobile viewport*); the control arm shares
the context so the arms differ by the intervention alone.

Two hygiene rules the engine enforces on itself:

- **Engagement check** — the blocked request must really have failed, the
  delayed resource really taken longer; otherwise the result is `ERROR`,
  never `NO_EFFECT`.
- **Floor/ceiling guard** — a NO_EFFECT on a boolean metric pinned against
  the predicted direction in the control arm is uninformative and produces
  no belief update.

### 3.3 Belief updating and falsification

Each hypothesis predicts, per candidate design, an outcome distribution over
{EFFECT, NO_EFFECT, OPPOSITE} on *its own* metric. After each experiment,
posterior odds are updated with the same likelihoods used for planning
(no post-hoc fiddling):

- a correct prediction with internally-replicated repeats (every control/
  treatment pair moved beyond the minimum effect) counts as *sharp*
  evidence (P ≈ 0.97);
- an ambiguous single result counts as ordinary evidence (P ≈ 0.88);
- posterior < 0.08 → **REFUTED** (merit +5 — the highest scientific act);
- posterior > 0.85 → sent to the independent review board;
- refuted hypotheses automatically spawn a *revised* hypothesis (independence
  or reversed-direction claim) that must survive its own re-test.

### 3.4 Experiment selection (expected information gain)

Classical Bayesian experiment design over the active hypothesis set:

    IG(E) = H(belief) − Σ_outcome P(outcome) · H(belief | outcome)

in bits, divided by cost (throttle/delay designs cost more), plus a
**review-unlock bonus** for experiments that can push a near-threshold
hypothesis across the verification gate (pure entropy would otherwise
always prefer fresh exploration). `discrimination_report` lists hypothesis
pairs whose predictions maximally disagree on a design — that is the
root-cause distinguishing power.

### 3.5 Multi-agent review

Six role-separated agents with deliberately restricted views:

| agent | sees | decides |
|---|---|---|
| HypothesisScientist | observations, graph | generates falsifiable claims |
| ExperimentDesigner | active hypotheses | max-information-gain designs |
| Executor | designs | runs the controlled trial |
| EvidenceInterpreter | experiment + stakes | Bayesian updates |
| RedTeamFalsifier | claim + evidence only | enumerates confounds (engagement, scope, floor/ceiling, near-threshold deltas, direction) |
| IndependentVerifier | the package only | blind re-runs it **before** reading the original outcome, then compares |

A discovery is recorded only when the red team's critiques are addressed
**and** the blind re-run replicates the result class.

### 3.6 Reproducibility & resumption

Every discovery ships a self-contained package (definition, environment,
evidence, screenshots, `rerun.py`, blind `VERIFICATION.md`). Runs resume:
hypotheses keep their posteriors, experiments keep their evidence,
observations are not re-crawled — the investigation continues across
sessions until the hypothesis space is exhausted.

---

## 4. Project structure & code walkthrough

```
techabout-causal-lab/
├── main.py                     orchestrator + CLI (see below)
├── requirements.txt
├── README.md / SAFETY.md
├── tests/test_engine.py        offline unit tests (sim driver, no network)
├── causal_lab/
│   ├── types.py                the scientific data model
│   ├── config.py               target, budgets, thresholds, conditions
│   ├── reports.py              investigation.md + summary.md generators
│   ├── observe/                "what can we see?"
│   │   ├── driver.py             PlaywrightDriver: real-browser loads with
│   │   │                         routing-level interventions, CDP throttling,
│   │   │                         console/network capture, screenshots; also a
│   │   │                         JS-off fallback DOM parser
│   │   ├── instrument.py         measurement JS + flatten_observation(): the
│   │   │                         single source of truth for metric names
│   │   ├── simulated.py          synthetic site with SEEDED ground-truth
│   │   │                         causality (the engine's own falsification)
│   │   ├── sitemap.py            page/nav discovery for any target
│   │   └── budget.py             rate limiter + hard load caps (safety cage)
│   ├── causal/                 "what could explain it?"
│   │   ├── graph.py              versioned causal graph; promote()/refute()/
│   │   │                         revive() enforce the correlation↔causation
│   │   │                         line and preserve refuted edges
│   │   └── hypotheses.py         anomaly detection, correlation mining,
│   │                             screening-probe generation (data-driven
│   │                             viewport/JS probes), revision-after-refutation
│   ├── experiments/            "how do we test it?"
│   │   ├── engine.py             two-arm trials, engagement checks, context
│   │   │                         sharing, per-experiment evidence dirs
│   │   ├── selection.py          Bayesian info-gain ranking, discrimination
│   │   │                         report, review-unlock bonus
│   │   └── repro.py              package builder + blind re-run machinery
│   └── reasoning/              "what did we learn?"
│       ├── falsify.py             EvidenceInterpreter (Bayes updates,
│       │                          floor guard, sharp evidence) + MeritLedger
│       ├── counterfactual.py      OBSERVED vs SIMULATED what-if reports
│       └── review.py              RedTeamFalsifier + IndependentVerifier
└── data/ (data-<host>/)        all run artefacts (see §1.4)
```

### Key modules in detail

**`types.py`** — every record in the system: `Observation` (one passive
measurement under stated conditions), `ResourceRecord`, `Hypothesis` (claim,
cause/effect factor keys, mechanism, prior/posterior, `Prediction`s,
falsification criteria, full evidence lists), `Experiment`/`Arm`/
`ExperimentResult` (deltas, result class), `Discovery`, `MeritEntry`, plus
deserialisers used for session resumption.

**`observe/driver.py`** — the only component that touches the real world.
`load(page, conditions, interventions, screenshot)` creates a fresh context
(= uncached), applies context-level interventions (viewport/throttle/js) to
the context and routing-level ones (block/delay/mock) via `page.route`,
measures, and returns an `Observation`. Politeness is structural: every
load passes `BudgetGuard.spend_load()`.

**`causal/graph.py`** — the scientific memory. `add_correlation`,
`add_candidate`, `promote` (requires evidence, refuses on refuted edges),
`refute` (preserves), `revive` (explicit re-opening only). Every mutation
appends to the edge's audit trail; `save_version()` writes full snapshots
plus an append-only ledger.

**`causal/hypotheses.py`** — three generative channels: **anomaly**
(failed resources, console errors, font/load failures, perf cliffs, missing
components → competing explanation sets), **correlation** (mined, recorded
as non-causal edges), and **screening** (round-robin mix of component /
layout / font / condition / timing probes, each checked for
falsifiability-as-designed before emission — a metric that cannot move
under any creatable condition yields no hypothesis).

**`experiments/selection.py`** — `rank_experiments()` computes IG per
candidate, applies the cost divisor, floor/ceiling deprioritisation for
boolean metrics pinned under standard conditions, and the review-unlock
bonus; `discrimination_report()` finds the hypothesis pairs a design best
separates.

**`reasoning/review.py`** — the gate. The verifier consumes *only* the
package, runs it on the driver, and reads the original outcome afterwards;
its transcript is written into `VERIFICATION.md` inside the package.

**`main.py`** — the `Investigation` orchestrator: observe → mine →
hypothesize → the falsification loop (selection, execution, interpretation,
refutation/revision, discovery reviews) → finalize (reports, packages,
state). The CLI wraps it with `--url/--pages/--data-dir` targeting so any
site can be audited with per-host state isolation.

---

## 5. Validating the engine itself

The engine is falsifiable about its own discovery ability.
`observe/simulated.py` implements a synthetic TechAbout-like site whose
ground-truth causal structure is *known* (chat widget injected by `chat.js`,
layout dependence on `tailwind.css`, a 768px responsive breakpoint,
deferred-script timing, third-party font dependency, throttle scaling,
JS-injection). Running

```bash
python main.py discover --sim --max-experiments 60
```

recovers **9/9 seeded relationships**, refutes the false ones, and prints
the recovery matrix. If the machinery were merely correlational this check
would fail. `tests/test_engine.py` adds 12 offline unit tests covering
graph integrity rules, engagement checks, context sharing, Bayesian
refutation, the floor guard, information-gain ordering, and package
re-runs.

## 6. Safety

See [SAFETY.md](SAFETY.md). In short: GET-only toward the origin, all
manipulation client-side in our own browser, navigations rate-limited
(≥1.5 s apart) and hard-capped, bounded designs (never block the document,
delays ≤ 6 s), honest reporting (SIMULATED counterfactuals labelled,
correlations never presented as causal, refutations preserved). If you
audit a site you do not own, keep these guards in place and stay within its
terms of service.

## 7. What the first TechAbout.com investigation found

77 controlled experiments across 5 resumed sessions, 24 refuted hypotheses
preserved, 6 independently verified discoveries:

| discovery | measured effect (treatment − control) |
|---|---|
| `res:tailwind.css` → layout height | blocking the stylesheet makes the homepage **+3812px taller** — unstyled content overflows its containers |
| `cond:throttle` → navigation time | slow-3G throttling adds **~+8.5–8.8s** to goto |
| `res:cart.js` → DOMContentLoaded | delaying the deferred script +4s delays DCL by **~+3.6–4.0s** |
| `res:chat.js` → DOMContentLoaded | **~+3.8–4.0s** |
| `res:forms.js` → DOMContentLoaded | **~+3.5–4.0s** |
| `res:header-account.js` → DOMContentLoaded | **~+2.6–3.3s** |

And equally informative refuted beliefs: the `#ta-chat` widget markup is
**server-rendered** (survives with JavaScript fully disabled); no script
module gates any component's visibility; the mobile menu is user-action-
gated, not breakpoint-gated. A smoke audit of an unrelated site
(`books.toscrape.com`) confirmed the generic path: it auto-discovered that
site's resources, refuted a false CSS dependency, and found that blocking
its `styles.css` inflates page height by +4800px.
