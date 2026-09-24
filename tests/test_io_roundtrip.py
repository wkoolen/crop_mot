"""Serialisation round-trips exactly. [B1/B2]

Unglamorous but load-bearing: every stage of the pipeline crosses a file, so a lossy
round-trip would quietly corrupt results everywhere at once. Float formatting is the usual
culprit - a writer that formats to six decimals makes `test_same_seed_reproduces_detections`
pass while changing the data.
"""

from __future__ import annotations

import numpy as np

from crop_mot.analysis.estimates_log import ScanEstimates, read_estimates, write_estimates
from crop_mot.sensor.record import read_detections, read_labels, write_detections, write_labels
from crop_mot.types import Detection, Pose2D, Scan, ScanLabels, TrackEstimate


def test_scan_roundtrip_is_exact(tmp_path) -> None:
    """Scan -> detections.jsonl -> Scan preserves every value bit for bit. [B1]

    Check k, t, the full pose, and every measurement vector. Use exact equality, not
    approximate: JSON can represent a float64 exactly with repr, so anything less is a
    writer that is throwing precision away.
    """
    rng = np.random.default_rng(0)
    scans = []
    for k in range(4):
        # Awkward floats on purpose: 0.1 and 1/3 have no short decimal form.
        pose = Pose2D(x=0.1 * k, y=1.0 / 3.0 + k, theta=float(rng.normal()))
        detections = tuple(Detection(z=rng.normal(size=2) * 10.0**k) for _ in range(k + 1))
        scans.append(Scan(k=k, t=0.25 * k + 1e-13, pose=pose, detections=detections))

    path = tmp_path / "detections.jsonl"
    write_detections(path, scans)
    back = read_detections(path)

    assert len(back) == len(scans)
    for original, restored in zip(scans, back):
        assert restored.k == original.k
        assert restored.t == original.t
        assert restored.pose == original.pose
        assert len(restored.detections) == len(original.detections)
        for d_orig, d_back in zip(original.detections, restored.detections):
            assert d_back.z.dtype == np.float64
            assert np.array_equal(d_back.z, d_orig.z)


def test_track_estimate_roundtrip_is_exact(tmp_path) -> None:
    """TrackEstimate -> estimates.jsonl -> TrackEstimate preserves r, mean and cov. [B2]

    r especially: B3 compares it against a closed form at near machine precision, so a
    lossy round-trip would make the cross-check fail for a reason unrelated to the
    mathematics.
    """
    rng = np.random.default_rng(1)
    A = rng.normal(size=(2, 2))
    estimate = TrackEstimate(track_id=7, r=1.0 / 3.0, mean=rng.normal(size=2), cov=A @ A.T)
    records = [
        ScanEstimates(k=0, estimates=()),
        ScanEstimates(k=1, estimates=(estimate,), diagnostics={"n_gated": 2.0}),
    ]

    path = tmp_path / "estimates_test.jsonl"
    write_estimates(path, records)
    back = read_estimates(path)

    assert [rec.k for rec in back] == [0, 1]
    assert back[0].estimates == ()
    assert back[1].diagnostics == {"n_gated": 2.0}
    restored = back[1].estimates[0]
    assert restored.track_id == estimate.track_id
    assert restored.r == estimate.r
    assert np.array_equal(restored.mean, estimate.mean)
    assert np.array_equal(restored.cov, estimate.cov)


def test_scan_labels_roundtrip_preserves_none_origins(tmp_path) -> None:
    """A clutter detection's origin survives as None, not as a sentinel. [B1/B3]

    JSON has null, so this should be natural - but a writer that coerces to int would turn
    clutter into object id 0, and B3's n_clutter_gated would then be silently wrong in a
    way no other test would catch.
    """
    labels = [
        ScanLabels(k=0, origin=(3, None, 0, None), visible_ids=(0, 3, 4), detected_ids=(0, 3)),
        ScanLabels(k=1, origin=(), visible_ids=(), detected_ids=()),
    ]

    path = tmp_path / "labels.jsonl"
    write_labels(path, labels)
    back = read_labels(path)

    assert back == labels
    assert back[0].origin[1] is None and back[0].origin[3] is None
    assert back[0].origin[2] == 0 and back[0].origin[2] is not None


def test_empty_scan_roundtrips(tmp_path) -> None:
    """A scan with zero detections survives the round trip as a scan, not a missing line. [B1]

    Scans with no detections are common - the robot passes a gap in the row - and they are
    exactly the scans that drive r downward in B2. Dropping them would shorten the
    trajectory and desynchronise it from the event sequence.
    """
    pose = Pose2D(x=0.0, y=0.0, theta=0.5)
    scans = [
        Scan(k=0, t=0.0, pose=pose, detections=(Detection(z=np.array([1.0, 2.0])),)),
        Scan(k=1, t=0.25, pose=pose, detections=()),
        Scan(k=2, t=0.5, pose=pose, detections=()),
    ]

    path = tmp_path / "detections.jsonl"
    write_detections(path, scans)
    back = read_detections(path)

    assert [scan.k for scan in back] == [0, 1, 2]
    assert back[1].detections == () and back[2].detections == ()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
