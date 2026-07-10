"""Vehicle counting pipeline (Phase F + G).

Composes: RTSPSource -> HailoYoloDetector -> VehicleTracker
          -> (debug snapshots + logger + CountRecorder -> Postgres)

Each newly detected vehicle emits a VehicleEvent once its direction of
travel (UP / DOWN / LEFT / RIGHT, in frame coordinates) resolves:
- an annotated snapshot is saved as <vehicle_id>-<timestamp>-<direction>.jpg
- a (vehicle_id, timestamp, direction) row is inserted into vehicle_events

Hardening (Phase G):
- RTSP auto-reconnect with exponential backoff on SourceLost.
- Hailo detector reset (release + reacquire VDevice) on inference exceptions.
- SIGTERM/SIGINT set a flag; the main loop exits cleanly and the final
  total_count row is flushed to Postgres.

Usage::

    python3 -m src.main --rtsp <uri> --hef <yolov8m.hef> \
        --dsn postgresql://user:pass@host:5432/vehicle_counts

``--mock-detector`` replaces the Hailo detector with a deterministic
simulator (for development hosts without a Hailo device).
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import cv2
import numpy as np

from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging
from src.detector import COCO_VEHICLE_NAMES, Detections, HailoYoloDetector
from src.ingest import RTSPSource, SourceLost
from src.persistence import CountRecorder
from src.tracker import TrackedObjects, VehicleEvent, VehicleTracker

logger = get_logger(__name__)

RECONNECT_BACKOFF_S = (1, 2, 5, 10, 30)
DETECTOR_RESET_BACKOFF_S = 5


class MockDetector:
    """Deterministic detector simulator for hosts without a Hailo device.

    Emits one synthetic "car" crossing the frame left-to-right every
    ``spawn_period`` seconds, ignoring frame content. Lets the full
    pipeline (tracking, counting, snapshots, persistence) run end-to-end.
    """

    def __init__(self, spawn_period: float = 15.0, crossing_time: float = 8.0) -> None:
        self.spawn_period = spawn_period
        self.crossing_time = crossing_time
        self._t0 = time.monotonic()

    def detect(self, frame: np.ndarray) -> Detections:
        h, w = frame.shape[:2]
        t = time.monotonic() - self._t0
        boxes, scores, classes = [], [], []
        # Cars spawn at t = k * spawn_period and cross for crossing_time.
        first_active = max(0, int((t - self.crossing_time) // self.spawn_period))
        for k in range(first_active, int(t // self.spawn_period) + 1):
            age = t - k * self.spawn_period
            if 0 <= age <= self.crossing_time:
                progress = age / self.crossing_time
                x = progress * (w - 100)
                y = h * 0.5 + (k % 3 - 1) * h * 0.15
                boxes.append([x, y, x + 100, y + 60])
                scores.append(0.9)
                classes.append(2)
        if not boxes:
            return Detections()
        return Detections(
            xyxy=np.asarray(boxes, dtype=np.float32),
            confidence=np.asarray(scores, dtype=np.float32),
            class_id=np.asarray(classes, dtype=np.int32),
        )

    def close(self) -> None:
        pass


class VehicleCounterApp:
    """End-to-end vehicle counting application."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.running = True
        self.debug_dir = Path(args.debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = VehicleTracker(frame_rate=int(args.target_fps))
        self.detector = self._create_detector()
        self.recorder = CountRecorder(
            args.dsn,
            get_count=lambda: self.tracker.total_count,
            interval_minutes=args.interval_minutes,
        )
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame) -> None:
        logger.info("Signal %s received, shutting down...", sig)
        self.running = False

    def _create_detector(self):
        if self.args.mock_detector:
            logger.warning("Using MOCK detector (no Hailo inference)")
            return MockDetector()
        return HailoYoloDetector(self.args.hef, score_threshold=self.args.score_threshold)

    def _reset_detector(self) -> None:
        """Phase G: release and reacquire the Hailo device after a failure."""
        logger.warning("Resetting detector...")
        try:
            self.detector.close()
        except Exception as e:
            logger.warning("Error closing detector during reset: %s", e)
        time.sleep(DETECTOR_RESET_BACKOFF_S)
        self.detector = self._create_detector()
        logger.info("Detector reset complete")

    def _handle_vehicle_event(
        self, frame: np.ndarray, tracked: TrackedObjects, event: VehicleEvent
    ) -> None:
        """New vehicle with resolved direction: save snapshot + DB row.

        Snapshot filename: <vehicle_id>-<timestamp>-<direction>.jpg
        """
        annotated = frame.copy()
        # Event vehicle in red, other active tracks in green.
        for i, tid in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            is_event = int(tid) == event.vehicle_id
            color = (0, 0, 255) if is_event else (0, 200, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            name = COCO_VEHICLE_NAMES.get(int(tracked.class_id[i]), "?")
            label = f"{name} #{int(tid)}" + (f" {event.direction}" if is_event else "")
            cv2.putText(
                annotated,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        # The event vehicle may already have left the active track list;
        # make sure its last known box is drawn regardless.
        if int(event.vehicle_id) not in [int(t) for t in tracked.tracker_id]:
            x1, y1, x2, y2 = event.xyxy.astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(event.timestamp))
        path = self.debug_dir / f"{event.vehicle_id}-{ts_str}-{event.direction}.jpg"
        cv2.imwrite(str(path), annotated)
        logger.info("Saved snapshot %s", path)

        self.recorder.record_vehicle(event.vehicle_id, event.direction, ts=event.timestamp)

    def _process_stream(self, source: RTSPSource, deadline: float | None) -> None:
        """Consume frames until shutdown, duration deadline, or SourceLost."""
        fps_count = 0
        fps_window_start: float | None = None  # set at first frame (skip connect latency)
        for frame in source.frames():
            if fps_window_start is None:
                fps_window_start = time.monotonic()
            if not self.running:
                break
            if deadline is not None and time.monotonic() >= deadline:
                self.running = False
                break

            try:
                detections = self.detector.detect(frame)
            except Exception as e:
                logger.error("Inference failed: %s", e)
                self._reset_detector()
                continue

            tracked = self.tracker.update(detections)
            for event in tracked.events:
                self._handle_vehicle_event(frame, tracked, event)

            fps_count += 1
            now = time.monotonic()
            if now - fps_window_start >= 5.0:
                fps = fps_count / (now - fps_window_start)
                logger.info(
                    "pipeline FPS: %.2f | active tracks: %d | total_count: %d",
                    fps,
                    len(tracked),
                    self.tracker.total_count,
                )
                fps_count = 0
                fps_window_start = now

    def run(self) -> int:
        """Main loop with RTSP auto-reconnect. Returns exit code."""
        deadline = time.monotonic() + self.args.duration if self.args.duration else None
        self.recorder.start()
        reconnect_attempt = 0
        try:
            while self.running:
                source = RTSPSource(self.args.rtsp, target_fps=self.args.target_fps)
                try:
                    source.start()
                    reconnect_attempt = 0
                    self._process_stream(source, deadline)
                except SourceLost as e:
                    if not self.running:
                        break
                    backoff = RECONNECT_BACKOFF_S[
                        min(reconnect_attempt, len(RECONNECT_BACKOFF_S) - 1)
                    ]
                    reconnect_attempt += 1
                    logger.warning(
                        "RTSP lost (%s); reconnect attempt %d in %ds",
                        e,
                        reconnect_attempt,
                        backoff,
                    )
                    # Sleep in small steps so signals still stop us promptly.
                    end = time.monotonic() + backoff
                    while self.running and time.monotonic() < end:
                        time.sleep(0.2)
                finally:
                    source.stop()
        finally:
            logger.info("Final total_count: %d", self.tracker.total_count)
            self.recorder.stop(flush=True)  # SIGTERM flush requirement
            try:
                self.detector.close()
            except Exception as e:
                logger.warning("Detector close error: %s", e)
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hailo vehicle counting pipeline")
    parser.add_argument("--rtsp", required=True, help="RTSP stream URI")
    parser.add_argument("--hef", default=None, help="Path to yolov8m HEF")
    parser.add_argument("--dsn", required=True, help="Postgres DSN")
    parser.add_argument("--target-fps", type=float, default=20.0, help="Ingest FPS")
    parser.add_argument(
        "--interval-minutes", type=float, default=5.0, help="Persist interval (minutes)"
    )
    parser.add_argument("--debug-dir", default="debug", help="Snapshot directory")
    parser.add_argument(
        "--score-threshold", type=float, default=0.3, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Stop after N seconds (for tests)"
    )
    parser.add_argument(
        "--mock-detector",
        action="store_true",
        help="Use a simulated detector (development without Hailo hardware)",
    )
    args = parser.parse_args(argv)
    if not args.mock_detector and not args.hef:
        parser.error("--hef is required unless --mock-detector is set")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    init_logging(level="INFO")
    app = VehicleCounterApp(args)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
