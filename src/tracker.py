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


@dataclass
class VehicleEvent:
    """Emitted once per unique vehicle, when its direction is resolved."""

    vehicle_id: int
    direction: str
    timestamp: float  # unix epoch seconds
    xyxy: np.ndarray  # last known box [x1, y1, x2, y2]
    class_id: int


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
            confirmed (and counted).
        min_displacement_px: Net centroid displacement required to resolve a
            vehicle's direction of travel.
        max_pending_frames: Frames after which a pending vehicle's direction
            is resolved with whatever history exists (UNKNOWN if it never
            moved enough).
    """

    def __init__(
        self,
        frame_rate: int = 20,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
        min_displacement_px: float = 20.0,
        max_pending_frames: int = 60,
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
        self.seen_ids: set[int] = set()
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

            if tid not in self.seen_ids:
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
            events.append(
                VehicleEvent(
                    vehicle_id=tid,
                    direction=direction,
                    timestamp=time.time(),
                    xyxy=np.asarray(state.last_box, dtype=np.float32),
                    class_id=state.last_class,
                )
            )
            del self._pending[tid]
            logger.info(
                "Vehicle %d direction resolved: %s (dx=%.0f dy=%.0f)", tid, direction, dx, dy
            )
        return events

    def _purge_stale_states(self) -> None:
        """Drop motion history for tracks not seen in a long time."""
        for tid in list(self._states):
            if tid in self._pending:
                continue
            if self._frame_idx - self._states[tid].last_seen_frame > HISTORY_PURGE_FRAMES:
                del self._states[tid]

    def reset(self) -> None:
        """Reset tracker state, the unique-id set, and motion histories."""
        self._tracker.reset()
        self.seen_ids.clear()
        self._states.clear()
        self._pending.clear()
        self._frame_idx = 0
