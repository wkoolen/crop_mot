"""Ground truth: where the plants are and where the robot walked. [B1]

Everything in this subpackage is TRUTH. It is written to truth.jsonl for evaluation and is
never handed to a filter - the only thing that crosses from here into the filter's world is
the reported pose inside a `Scan`.
"""
