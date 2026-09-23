"""Pipeline entry points: config -> run folder. [B1/B2]

The pipeline is strictly one-directional:

    config + seed -> truth -> sensor model -> detections.jsonl
                                                   |
                                                   v
                                          filter (predict/update/extract)
                                                   |
                                                   v
                                          estimates_<filter>.jsonl -> analysis/plots

Each arrow crosses a file on disk, not a function call. That is what lets you re-run a
filter on last week's detections, run six filters on the identical data, and trace any plot
back to the config that produced it.
"""
