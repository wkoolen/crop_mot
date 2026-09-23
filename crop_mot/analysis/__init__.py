"""Evaluation, validation and plotting. The only place allowed to read ground truth. [B3]

Everything here runs AFTER a filter has finished and reads from the run folder. That
ordering is what keeps the truth/filter separation honest: by the time this code opens
truth.jsonl, the estimates it is judging are already on disk and cannot be influenced.

B3 lives here in two halves:
  * `analytic` + `events`  - the closed-form reference the filter is checked against;
  * `montecarlo`           - the empirical check that the modelling assumptions hold on
                             average, not just on one realisation.
"""
