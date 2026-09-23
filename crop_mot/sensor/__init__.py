"""The black-box detector: geometry, noise, misses and clutter. [B1]

Vision is deliberately NOT modelled. The detector is a stub with three knobs - detection
probability p_D, clutter rate lambda_FA, and Gaussian measurement noise R - because the
thesis contribution is the estimation and data association, not the perception.

Note the split between two ideas that are easy to conflate:
  * `models.MeasurementModel` - given that a detection happened, what does z look like?
  * `sensor_model.SensorModel` - does a detection happen at all, and how much clutter?
The filter needs both, and gets its own instances configured from `filter.assumed_sensor`,
which may deliberately differ from the truth.
"""
