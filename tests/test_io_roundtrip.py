"""Serialisation round-trips exactly. [B1/B2]

Unglamorous but load-bearing: every stage of the pipeline crosses a file, so a lossy
round-trip would quietly corrupt results everywhere at once. Float formatting is the usual
culprit - a writer that formats to six decimals makes `test_same_seed_reproduces_detections`
pass while changing the data.
"""

from __future__ import annotations


def test_scan_roundtrip_is_exact(tmp_path) -> None:
    """Scan -> detections.jsonl -> Scan preserves every value bit for bit. [B1]

    Check k, t, the full pose, and every measurement vector. Use exact equality, not
    approximate: JSON can represent a float64 exactly with repr, so anything less is a
    writer that is throwing precision away.
    """
    raise NotImplementedError


def test_track_estimate_roundtrip_is_exact(tmp_path) -> None:
    """TrackEstimate -> estimates.jsonl -> TrackEstimate preserves r, mean and cov. [B2]

    r especially: B3 compares it against a closed form at near machine precision, so a
    lossy round-trip would make the cross-check fail for a reason unrelated to the
    mathematics.
    """
    raise NotImplementedError


def test_scan_labels_roundtrip_preserves_none_origins(tmp_path) -> None:
    """A clutter detection's origin survives as None, not as a sentinel. [B1/B3]

    JSON has null, so this should be natural - but a writer that coerces to int would turn
    clutter into object id 0, and B3's n_clutter_gated would then be silently wrong in a
    way no other test would catch.
    """
    raise NotImplementedError


def test_empty_scan_roundtrips(tmp_path) -> None:
    """A scan with zero detections survives the round trip as a scan, not a missing line. [B1]

    Scans with no detections are common - the robot passes a gap in the row - and they are
    exactly the scans that drive r downward in B2. Dropping them would shorten the
    trajectory and desynchronise it from the event sequence.
    """
    raise NotImplementedError
