"""Gating and assignment, shared by every filter that needs them. [B4]

Stubbed in phase 1 although only B3's gating is used, so that adding a B4 filter really is
"add one file" rather than "add a filter plus the plumbing it needs". The alternative -
letting each filter grow its own copy of gating - is how three filters end up with three
subtly different chi-square thresholds.

Who uses what:
  * gating     - Bernoulli (B2), and every B4 filter.
  * assignment - GNN (single best hard assignment).
  * murty      - JPDA (marginal association probabilities) and PMBM (global hypotheses).
"""
