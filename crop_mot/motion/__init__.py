"""Target dynamics. [B2/B4]

Separate from the sensor because the thesis separates them: the motion model is what the
prediction step uses, the measurement model is what the update step uses. For static plants
the motion model is nearly trivial, but it is still an injected object so that a moving-
target variant is a new file rather than an edit to every filter.
"""
