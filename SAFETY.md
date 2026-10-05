# Safety & Ethics

The Causal Lab investigates a **publicly observable production website**.
The following rules are enforced in code, not just documented:

1. **Read-only toward the origin.** The engine only ever performs ordinary
   GET navigations of public pages. It never submits forms, never
   authenticates, never POSTs, and never touches admin or private surfaces.

2. **All manipulation is client-side.** "Interventions" (blocking, delaying,
   mocking, throttling, viewport emulation) happen inside the lab's own
   Chromium context via request routing and CDP. The origin cannot observe
   a difference from an ordinary page load.

3. **Rate limiting.** `BudgetGuard` enforces a minimum interval between
   navigations (default 1.5 s) and a hard cap on total page loads per run
   (default 140) plus a wall-clock budget (default 45 min). Every page load
   in the codebase goes through it.

4. **Bounded experiment designs.** `validate_design` refuses to block the
   document itself, rejects overly broad block patterns, and caps delay
   magnitudes (≤ 6 s).

5. **No load testing.** Two repeats per arm by default; this is a handful of
   page loads per experiment — comparable to a single human tester — not a
   traffic generator.

6. **Honest reporting.** Counterfactual statements are labelled SIMULATED
   and must cite measured evidence; correlations are never presented as
   causal; refuted hypotheses are preserved and reported.

If you fork this engine to investigate a site you do not own, keep these
guards in place and stay within the site's terms of service and applicable
computer-misuse law.
