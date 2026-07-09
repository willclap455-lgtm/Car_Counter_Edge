"""Phase D tests: VehicleTracker unique-id counting on a synthetic clip.

Simulates a short clip (20 FPS, 1280x720) in which three cars cross the
frame one after another, driving the tracker with per-frame detections.
"""

import numpy as np

from src.detector import Detections
from src.tracker import VehicleTracker


def _car_box(x: float, y: float = 400.0, w: float = 160.0, h: float = 90.0) -> list[float]:
    return [x, y, x + w, y + h]


def _dets(boxes: list[list[float]], class_id: int = 2, conf: float = 0.9) -> Detections:
    if not boxes:
        return Detections()
    n = len(boxes)
    return Detections(
        xyxy=np.asarray(boxes, dtype=np.float32),
        confidence=np.full((n,), conf, dtype=np.float32),
        class_id=np.full((n,), class_id, dtype=np.int32),
    )


def _drive_clip(tracker: VehicleTracker) -> list[dict[int, list[float]]]:
    """Three cars cross left-to-right, staggered in time; returns per-frame tracks."""
    frames = []
    speed = 12.0  # px/frame
    # (start_frame, end_frame, x at start)
    cars = [(0, 60, -100.0), (30, 90, -100.0), (60, 120, -100.0)]
    per_frame_tracks: list[dict[int, list[float]]] = []
    for f in range(130):
        boxes = []
        for start, end, x0 in cars:
            if start <= f < end:
                boxes.append(_car_box(x0 + speed * (f - start) + 150))
        frames.append(boxes)
        tracked = tracker.update(_dets(boxes))
        per_frame_tracks.append(
            {int(t): tracked.xyxy[i].tolist() for i, t in enumerate(tracked.tracker_id)}
        )
    return per_frame_tracks


def test_total_count_is_3_at_end_of_clip():
    tracker = VehicleTracker(frame_rate=20)
    _drive_clip(tracker)
    assert tracker.total_count == 3, f"expected 3 unique vehicles, got {tracker.total_count}"


def test_id_constant_for_single_car_across_frames():
    tracker = VehicleTracker(frame_rate=20)
    ids_seen = set()
    for f in range(50):
        boxes = [_car_box(50.0 + 10.0 * f)]
        tracked = tracker.update(_dets(boxes))
        if len(tracked):
            ids_seen.update(int(t) for t in tracked.tracker_id)
    assert len(ids_seen) == 1, f"single car got multiple ids: {ids_seen}"
    assert tracker.total_count == 1


def test_new_ids_reported_once():
    tracker = VehicleTracker(frame_rate=20)
    all_new = []
    for f in range(30):
        boxes = [_car_box(100.0 + 8.0 * f)]
        if f >= 10:
            boxes.append(_car_box(300.0 + 8.0 * (f - 10), y=150.0))
        tracked = tracker.update(_dets(boxes))
        all_new.extend(tracked.new_ids)
    assert len(all_new) == len(set(all_new)) == 2
    assert tracker.total_count == 2


def test_empty_frames_do_not_crash_or_count():
    tracker = VehicleTracker(frame_rate=20)
    for _ in range(10):
        tracked = tracker.update(Detections())
        assert len(tracked) == 0
    assert tracker.total_count == 0


def test_reset_clears_counts():
    tracker = VehicleTracker(frame_rate=20)
    for f in range(10):
        tracker.update(_dets([_car_box(100.0 + 10.0 * f)]))
    assert tracker.total_count == 1
    tracker.reset()
    assert tracker.total_count == 0
