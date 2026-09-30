"""Clairvoyant per-scan counts: what the detector saw versus what was really there. [B1]

EVALUATION ONLY. Everything here is derived from labels.jsonl, which a filter never reads,
so these counts know things no filter can: how many plants were inside the FOV, which
detections were clutter, and why a visible plant went undetected. Nothing is stored in
the run folder; labels.jsonl stays the single source of truth and the counts are
recomputed on demand, so they cannot drift out of step with it.

Serves: [B1] sanity-checking a scenario (empirical p_D against the configured one, clutter
mean against lambda_FA, detections per plant once multiplicity is on, how much of the
clutter came from weeds); the `counts` plot.
"""

from __future__ import annotations

from dataclasses import dataclass

from crop_mot.types import ScanLabels


@dataclass(frozen=True)
class ScanCounts:
    """One scan's truth-side tally. [B1]

    Identities that hold for every scan:
        n_total   = n_object_detections + n_clutter
        n_clutter = n_weed + n_transient
        n_visible = n_detected_objects + n_missed + n_truncated

    Attributes:
        k: scan index.
        n_visible: plants inside the FOV (seen from the TRUE pose).
        n_detected_objects: distinct plants with at least one detection.
        n_object_detections: detections originating from a plant. Equal to
            n_detected_objects when multiplicity is "single"; larger otherwise.
        n_clutter: detections with no origin: n_weed + n_transient.
        n_weed: clutter detections that came from a weed (decision D15).
        n_transient: the Poisson clutter, drawn fresh every scan (compare with lambda_FA).
        n_visible_weeds: weeds inside the FOV (seen from the TRUE pose).
        n_missed: visible plants that produced no detection at all (the p_D coin flip, or
            a Poisson count of zero).
        n_truncated: visible plants whose detections all fell outside the FOV (D4).
        n_total: all detections the filter received this scan.
    """

    k: int
    n_visible: int
    n_detected_objects: int
    n_object_detections: int
    n_clutter: int
    n_weed: int
    n_transient: int
    n_visible_weeds: int
    n_missed: int
    n_truncated: int
    n_total: int


def weed_origin(label: ScanLabels) -> tuple[int | None, ...]:
    """The weed each detection came from, or None: one entry per detection. [B1, D15]

    `ScanLabels.weed_origin` is left empty for a scenario without weeds; this expands it
    to one None per detection, so callers can zip it with `origin` without the empty
    form silently truncating the zip.
    """
    return label.weed_origin or (None,) * len(label.origin)


def scan_counts(labels: list[ScanLabels]) -> list[ScanCounts]:
    """Tally each scan's labels. [B1]

    Args:
        labels: the run's labels, as read by `crop_mot.sensor.record.read_labels`.

    Returns:
        One ScanCounts per scan, in the same order.
    """
    counts = []
    for label in labels:
        n_clutter = sum(1 for origin in label.origin if origin is None)
        n_weed = sum(1 for weed in weed_origin(label) if weed is not None)
        n_visible = len(label.visible_ids)
        n_detected = len(label.detected_ids)
        n_truncated = len(label.truncated_ids)
        counts.append(ScanCounts(
            k=label.k,
            n_visible=n_visible,
            n_detected_objects=n_detected,
            n_object_detections=len(label.origin) - n_clutter,
            n_clutter=n_clutter,
            n_weed=n_weed,
            n_transient=n_clutter - n_weed,
            n_visible_weeds=len(label.visible_weed_ids),
            n_missed=n_visible - n_detected - n_truncated,
            n_truncated=n_truncated,
            n_total=len(label.origin),
        ))
    return counts


def summarise(counts: list[ScanCounts]) -> dict[str, float]:
    """Totals and the empirical rates they imply. [B1]

    Args:
        counts: the output of `scan_counts`.

    Returns:
        A dict holding the totals of every count field, plus:
          * "p_D_empirical": detected plant-scans / visible plant-scans, the detection
            probability INCLUDING the FOV-edge loss (compare with the configured p_D);
          * "clutter_per_scan": mean clutter detections per scan, weeds included;
          * "transient_per_scan": mean Poisson clutter per scan (compare with lambda_FA);
          * "weed_per_scan": mean weed detections per scan;
          * "weed_p_D_empirical": weed detections / visible weed-scans, including the
            FOV-edge loss (compare with weed_detection's p_D);
          * "detections_per_detected_object": 1.0 for multiplicity "single".
        Rates whose denominator is zero are NaN.
    """
    fields = ("n_visible", "n_detected_objects", "n_object_detections", "n_clutter",
              "n_weed", "n_transient", "n_visible_weeds", "n_missed", "n_truncated",
              "n_total")
    totals = {name: sum(getattr(c, name) for c in counts) for name in fields}

    def ratio(num: float, den: float) -> float:
        return num / den if den else float("nan")

    return {
        "n_scans": len(counts),
        **totals,
        "p_D_empirical": ratio(totals["n_detected_objects"], totals["n_visible"]),
        "clutter_per_scan": ratio(totals["n_clutter"], len(counts)),
        "transient_per_scan": ratio(totals["n_transient"], len(counts)),
        "weed_per_scan": ratio(totals["n_weed"], len(counts)),
        "weed_p_D_empirical": ratio(totals["n_weed"], totals["n_visible_weeds"]),
        "detections_per_detected_object": ratio(totals["n_object_detections"],
                                                totals["n_detected_objects"]),
    }


def format_summary(counts: list[ScanCounts]) -> str:
    """A short plain-text table of `summarise`, for printing after `simulate`. [B1]

    The clutter is split into weeds and transient only when a weed was ever in view, so
    the table of a scenario without weeds is unchanged.
    """
    s = summarise(counts)
    lines = [
        f"Clairvoyant counts over {s['n_scans']} scans (truth-side; filters never see this)",
        f"  plant-scans in FOV       {s['n_visible']:6d}",
        f"    detected               {s['n_detected_objects']:6d}"
        f"   empirical p_D {s['p_D_empirical']:.3f}",
        f"    missed (no detection)  {s['n_missed']:6d}",
        f"    lost at FOV edge       {s['n_truncated']:6d}",
        f"  detections received      {s['n_total']:6d}",
        f"    from plants            {s['n_object_detections']:6d}"
        f"   per detected plant {s['detections_per_detected_object']:.2f}",
        f"    clutter                {s['n_clutter']:6d}"
        f"   per scan {s['clutter_per_scan']:.2f}",
    ]
    if s["n_visible_weeds"] > 0:
        lines += [
            f"      transient (Poisson)  {s['n_transient']:6d}"
            f"   per scan {s['transient_per_scan']:.2f}",
            f"      from weeds           {s['n_weed']:6d}"
            f"   per scan {s['weed_per_scan']:.2f}",
            f"  weed-scans in FOV        {s['n_visible_weeds']:6d}"
            f"   reported as plants {s['weed_p_D_empirical']:.3f}",
        ]
    return "\n".join(lines)
