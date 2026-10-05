"""Causal Lab — an autonomous scientific-investigation engine for the publicly
observable TechAbout.com website.

The system runs a falsification-first scientific loop:

    Observation -> Hypothesis -> Experiment Design -> Controlled Experiment
                -> Evidence -> Falsification -> Revised Hypothesis
                -> Independent Verification -> Discovery

Nothing in this package attacks or burdens the target server: every
"manipulation" happens inside our own browser context (request routing,
emulation, throttling) against publicly observable pages.
"""

__version__ = "1.0.0"
