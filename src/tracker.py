"""ByteTrack-based vehicle tracker and unique-ID counter (Phase D).

VehicleTracker wraps supervision.ByteTrack. Each call to ``update`` feeds the
current frame's detections to the tracker and records every tracker id ever
seen in ``seen_ids``; ``total_count`` is the number of unique ids seen since
start. New ids are reported so callers can save debug snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from hailo_apps.python.core.common.hailo_logger import get_logger
from src.detector import Detections

logger = get_logger(__name__)


@dataclass
class TrackedObjects:
    """Per-frame tracking output."""

    xyxy: np.ndarray
    tracker_id: np.ndarray
    class_id: np.ndarray
    confidence: np.ndarray
    new_ids: list[int]

    def __len__(self) -> int:
        return len(self.tracker_id)


class VehicleTracker:
    """Assigns persistent ids to vehicle detections and counts unique vehicles.

    Args:
        frame_rate: Ingest frame rate (ByteTrack uses it to scale its
            lost-track buffer).
        track_activation_threshold: Detection confidence needed to start or
            keep a track active.
        lost_track_buffer: Frames a lost track is kept alive for re-matching.
        minimum_matching_threshold: IoU threshold for detection-track matching.
        minimum_consecutive_frames: Frames a track must persist before it is
            confirmed (and counted).
    """

    def __init__(
        self,
        frame_rate: int = 20,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
        self.seen_ids: set[int] = set()

    @property
    def total_count(self) -> int:
        """Number of unique tracker ids ever seen."""
        return len(self.seen_ids)

    def update(self, detections: Detections) -> TrackedObjects:
        """Feed one frame's detections; returns tracks with persistent ids."""
        sv_dets = sv.Detections(
            xyxy=detections.xyxy.reshape(-1, 4).astype(np.float32),
            confidence=detections.confidence.astype(np.float32),
            class_id=detections.class_id.astype(np.int32),
        )
        tracked = self._tracker.update_with_detections(sv_dets)

        tracker_ids = (
            tracked.tracker_id if tracked.tracker_id is not None else np.zeros((0,), dtype=np.int32)
        )
        new_ids = [int(tid) for tid in tracker_ids if int(tid) not in self.seen_ids]
        self.seen_ids.update(new_ids)
        if new_ids:
            logger.info("New vehicle ids %s (total_count=%d)", new_ids, self.total_count)

        return TrackedObjects(
            xyxy=tracked.xyxy,
            tracker_id=np.asarray(tracker_ids, dtype=np.int64),
            class_id=(
                tracked.class_id if tracked.class_id is not None else np.zeros((0,), dtype=np.int32)
            ),
            confidence=(
                tracked.confidence
                if tracked.confidence is not None
                else np.zeros((0,), dtype=np.float32)
            ),
            new_ids=new_ids,
        )

    def reset(self) -> None:
        """Reset tracker state and the unique-id set."""
        self._tracker.reset()
        self.seen_ids.clear()
