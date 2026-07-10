"""ByteTrack-based vehicle tracker, unique-ID counter, and direction of
travel estimation (Phases D + direction feature).

VehicleTracker wraps supervision.ByteTrack. Each call to ``update`` feeds the
current frame's detections to the tracker and records every tracker id ever
seen in ``seen_ids``; ``total_count`` is the number of unique ids seen since
start.

Direction of travel: each track's centroid history is recorded. A newly seen
vehicle stays "pending" until it has moved at least ``min_displacement_px``
(or ``max_pending_frames`` have elapsed); then a VehicleEvent is emitted with
its direction — the dominant axis of net displacement in frame coordinates:
UP / DOWN / LEFT / RIGHT (UNKNOWN if it never moved enough). Events drive the
debug snapshot and the per-vehicle database row.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from hailo_apps.python.core.common.hailo_logger import get_logger
from src.detector import Detections

logger = get_logger(__name__)

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "UNKNOWN")

# Frames of absence after which a track's history is purged.
HISTORY_PURGE_FRAMES = 600


def classify_direction(dx: float, dy: float, min_displacement_px: float) -> str:
    """Dominant-axis direction in image coordinates (y grows downward)."""
    if math.hypot(dx, dy) < min_displacement_px:
        return "UNKNOWN"
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


def travel_angle(dx: float, dy: float, min_displacement_px: float) -> float | None:
    """Angle of travel in degrees on the UP/DOWN/LEFT/RIGHT plane.

    0 = UP, 90 = RIGHT, 180 = DOWN, 270 = LEFT (clockwise). Image
    coordinates have y growing downward, hence the -dy. Returns None when
    the displacement is too small to be meaningful (UNKNOWN direction).
    """
    if math.hypot(dx, dy) < min_displacement_px:
        return None
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class VehicleEvent:
    """Emitted once per unique vehicle, when its direction is resolved."""

    vehicle_id: int
    direction: str
    timestamp: float  # unix epoch seconds
    xyxy: np.ndarray  # last known box [x1, y1, x2, y2]
    class_id: int
    angle: float | None = None  # degrees: 0=UP, 90=RIGHT, 180=DOWN, 270=LEFT


@dataclass
class TrackedObjects:
    """Per-frame tracking output."""

    xyxy: np.ndarray
    tracker_id: np.ndarray
    class_id: np.ndarray
    confidence: np.ndarray
    new_ids: list[int]
    events: list[VehicleEvent] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tracker_id)


@dataclass
class _TrackState:
    """Motion history for one tracker id."""

    centroids: deque  # of (cx, cy)
    last_box: np.ndarray
    last_class: int
    last_seen_frame: int


class VehicleTracker:
    """Assigns persistent ids to vehicle detections, counts unique vehicles,
    and estimates each vehicle's direction of travel.

    Args:
        frame_rate: Ingest frame rate (ByteTrack uses it to scale its
            lost-track buffer).
        track_activation_threshold: Detection confidence needed to start or
            keep a track active.
        lost_track_buffer: Frames a lost track is kept alive for re-matching.
        minimum_matching_threshold: IoU threshold for detection-track matching.
        minimum_consecutive_frames: Frames a track must persist before it is
            confirmed (and counted). The default of 3 keeps single-frame
            detector flickers and ghost boxes from ever becoming counts.
        min_displacement_px: Net centroid displacement required to resolve a
            vehicle's direction of travel.
        max_pending_frames: Frames after which a pending vehicle's direction
            is resolved with whatever history exists (UNKNOWN if it never
            moved enough).
        duplicate_iou_threshold: A brand-new track whose box overlaps an
            existing active track by at least this IoU is treated as a
            duplicate id for the same vehicle and never counted.

    Note on lost-track persistence: supervision scales the buffer as
    ``max_time_lost = frame_rate / 30 * lost_track_buffer`` frames. The
    default lost_track_buffer=90 at 20 FPS keeps a lost id alive for 60
    frames (~3 s), so a vehicle that flickers out of detection briefly
    re-matches its old id instead of being double counted. State per lost
    track is a few hundred bytes, so this cannot bog the system down.
    """

    def __init__(
        self,
        frame_rate: int = 20,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 90,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 3,
        min_displacement_px: float = 20.0,
        max_pending_frames: int = 60,
        duplicate_iou_threshold: float = 0.55,
    ) -> None:
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
        self.min_displacement_px = float(min_displacement_px)
        self.max_pending_frames = int(max_pending_frames)
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        self.seen_ids: set[int] = set()
        self.duplicate_ids: set[int] = set()
        self._frame_idx = 0
        self._states: dict[int, _TrackState] = {}
        self._pending: dict[int, int] = {}  # id -> frame when first seen

    @property
    def total_count(self) -> int:
        """Number of unique tracker ids ever seen."""
        return len(self.seen_ids)

    def update(self, detections: Detections) -> TrackedObjects:
        """Feed one frame's detections; returns tracks with persistent ids
        plus any VehicleEvents whose direction resolved this frame."""
        self._frame_idx += 1
        sv_dets = sv.Detections(
            xyxy=detections.xyxy.reshape(-1, 4).astype(np.float32),
            confidence=detections.confidence.astype(np.float32),
            class_id=detections.class_id.astype(np.int32),
        )
        tracked = self._tracker.update_with_detections(sv_dets)

        tracker_ids = (
            tracked.tracker_id if tracked.tracker_id is not None else np.zeros((0,), dtype=np.int32)
        )
        class_ids = (
            tracked.class_id if tracked.class_id is not None else np.zeros((0,), dtype=np.int32)
        )

        current_boxes = {int(t): tracked.xyxy[i] for i, t in enumerate(tracker_ids)}
        new_ids: list[int] = []
        for i, raw_tid in enumerate(tracker_ids):
            tid = int(raw_tid)
            box = tracked.xyxy[i]
            cx, cy = float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0
            state = self._states.get(tid)
            if state is None:
                state = _TrackState(
                    centroids=deque(maxlen=2 * self.max_pending_frames),
                    last_box=box,
                    last_class=int(class_ids[i]),
                    last_seen_frame=self._frame_idx,
                )
                self._states[tid] = state
            state.centroids.append((cx, cy))
            state.last_box = box
            state.last_class = int(class_ids[i])
            state.last_seen_frame = self._frame_idx

            if tid in self.seen_ids or tid in self.duplicate_ids:
                continue
            if self._is_duplicate_of_counted_track(tid, box, current_boxes):
                self.duplicate_ids.add(tid)
                logger.info(
                    "Track %d looks like a duplicate id of an existing vehicle; not counting", tid
                )
                continue
            self.seen_ids.add(tid)
            new_ids.append(tid)
            self._pending[tid] = self._frame_idx

        if new_ids:
            logger.info("New vehicle ids %s (total_count=%d)", new_ids, self.total_count)

        events = self._resolve_pending()
        self._purge_stale_states()

        return TrackedObjects(
            xyxy=tracked.xyxy,
            tracker_id=np.asarray(tracker_ids, dtype=np.int64),
            class_id=class_ids,
            confidence=(
                tracked.confidence
                if tracked.confidence is not None
                else np.zeros((0,), dtype=np.float32)
            ),
            new_ids=new_ids,
            events=events,
        )

    def _is_duplicate_of_counted_track(
        self, tid: int, box: np.ndarray, current_boxes: dict[int, np.ndarray]
    ) -> bool:
        """True if a brand-new track's box overlaps an already-counted,
        currently visible track enough to be the same physical vehicle."""
        for other_tid, other_box in current_boxes.items():
            if other_tid == tid or other_tid not in self.seen_ids:
                continue
            if _box_iou(box, other_box) >= self.duplicate_iou_threshold:
                return True
        return False

    def _displacement(self, state: _TrackState) -> tuple[float, float]:
        first = state.centroids[0]
        last = state.centroids[-1]
        return last[0] - first[0], last[1] - first[1]

    def _resolve_pending(self) -> list[VehicleEvent]:
        """Emit an event for each pending vehicle whose direction is known."""
        events: list[VehicleEvent] = []
        for tid in list(self._pending):
            state = self._states[tid]
            dx, dy = self._displacement(state)
            moved_enough = math.hypot(dx, dy) >= self.min_displacement_px
            timed_out = self._frame_idx - self._pending[tid] >= self.max_pending_frames
            if not (moved_enough or timed_out):
                continue
            direction = classify_direction(dx, dy, self.min_displacement_px)
            angle = travel_angle(dx, dy, self.min_displacement_px)
            events.append(
                VehicleEvent(
                    vehicle_id=tid,
                    direction=direction,
                    timestamp=time.time(),
                    xyxy=np.asarray(state.last_box, dtype=np.float32),
                    class_id=state.last_class,
                    angle=angle,
                )
            )
            del self._pending[tid]
            logger.info(
                "Vehicle %d direction resolved: %s angle=%s (dx=%.0f dy=%.0f)",
                tid,
                direction,
                f"{angle:.1f}" if angle is not None else "None",
                dx,
                dy,
            )
        return events

    def _purge_stale_states(self) -> None:
        """Drop motion history (and duplicate flags) for long-gone tracks."""
        for tid in list(self._states):
            if tid in self._pending:
                continue
            if self._frame_idx - self._states[tid].last_seen_frame > HISTORY_PURGE_FRAMES:
                del self._states[tid]
                # ByteTrack ids are never reused, so the flag is done its job.
                self.duplicate_ids.discard(tid)

    def reset(self) -> None:
        """Reset tracker state, the unique-id set, and motion histories."""
        self._tracker.reset()
        self.seen_ids.clear()
        self.duplicate_ids.clear()
        self._states.clear()
        self._pending.clear()
        self._frame_idx = 0
