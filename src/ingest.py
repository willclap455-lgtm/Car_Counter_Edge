"""RTSP ingestion for the vehicle counting app (Phase B).

Provides RTSPSource, which reads an RTSP stream with OpenCV (GStreamer
backend when available, FFMPEG otherwise) and re-times the frames to a
fixed output rate (default 20 FPS): frames are dropped when the camera is
faster and duplicated when it is slower, so consumers see a steady clock.

A capture thread feeds a bounded queue; the consumer-facing generator
paces itself with a monotonic deadline schedule. If no frame arrives for
``lost_timeout`` seconds (or the connection dies), SourceLost is raised.

Test mode (Phase B test)::

    python3 -m src.ingest --rtsp <uri> --duration 60
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from collections.abc import Iterator

import numpy as np

# Force TCP for the FFMPEG backend before cv2 is imported.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2

from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)


class SourceLost(Exception):
    """Raised when the RTSP source stops delivering frames."""


def _gstreamer_available() -> bool:
    return "GStreamer:                   YES" in cv2.getBuildInformation()


class RTSPSource:
    """Reads an RTSP stream and yields frames at a fixed target FPS.

    Args:
        uri: RTSP URI (rtsp://user:pass@host:port/path).
        target_fps: Output frame rate maintained via drop/duplicate.
        queue_size: Bounded capture queue size (oldest frames dropped).
        lost_timeout: Seconds without a fresh frame (after the first one)
            before SourceLost is raised.
        connect_timeout: Seconds allowed for the first frame to arrive
            (RTSP negotiation + decoder probing can take several seconds).
    """

    def __init__(
        self,
        uri: str,
        target_fps: float = 20.0,
        queue_size: int = 8,
        lost_timeout: float = 10.0,
        connect_timeout: float = 30.0,
    ) -> None:
        self.uri = uri
        self.target_fps = float(target_fps)
        self.lost_timeout = float(lost_timeout)
        self.connect_timeout = float(connect_timeout)
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._capture_dead = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self.frames_captured = 0
        self.frames_emitted = 0

    def _open_capture(self) -> cv2.VideoCapture:
        if _gstreamer_available():
            pipeline = (
                f'rtspsrc location="{self.uri}" protocols=tcp latency=200 ! '
                "rtph264depay ! h264parse ! avdec_h264 ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=2 sync=false"
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                logger.info("RTSP opened via GStreamer backend")
                return cap
            cap.release()
            logger.warning("GStreamer open failed, falling back to FFMPEG")
        cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            logger.info("RTSP opened via FFMPEG backend (TCP transport)")
        return cap

    def start(self) -> RTSPSource:
        """Open the stream and start the capture thread."""
        self._cap = self._open_capture()
        if not self._cap.isOpened():
            raise SourceLost(f"Cannot open RTSP source: {self.uri}")
        self._stop.clear()
        self._capture_dead.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        consecutive_failures = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 25:
                    logger.error(
                        "Capture failed %d times in a row, marking source dead",
                        consecutive_failures,
                    )
                    break
                time.sleep(0.02)
                continue
            consecutive_failures = 0
            self.frames_captured += 1
            # Bounded queue: drop the oldest frame when full (camera faster
            # than consumer), keeping latency low.
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    pass
        self._capture_dead.set()

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames at exactly ``target_fps``.

        Duplicates the last frame when the camera is slow, drops frames when
        it is fast. Raises SourceLost when the source dies or stalls beyond
        ``lost_timeout``.
        """
        if self._thread is None:
            self.start()
        period = 1.0 / self.target_fps
        last_frame: np.ndarray | None = None
        last_fresh_ts = time.monotonic()
        next_deadline = time.monotonic() + period

        while not self._stop.is_set():
            # Drain the queue down to the newest frame (drop policy).
            fresh = None
            while True:
                try:
                    fresh = self._queue.get_nowait()
                except queue.Empty:
                    break
            if fresh is None and last_frame is None:
                # Block for the very first frame.
                try:
                    fresh = self._queue.get(timeout=0.25)
                except queue.Empty:
                    fresh = None
            if fresh is not None:
                last_frame = fresh
                last_fresh_ts = time.monotonic()

            now = time.monotonic()
            if self._capture_dead.is_set() and self._queue.empty():
                raise SourceLost(f"RTSP source lost: {self.uri}")
            # Allow a longer grace period before the very first frame.
            stall_limit = self.lost_timeout if last_frame is not None else self.connect_timeout
            if now - last_fresh_ts > stall_limit:
                raise SourceLost(f"RTSP source stalled >{stall_limit:.0f}s: {self.uri}")

            if last_frame is not None:
                sleep_for = next_deadline - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                # Deadline schedule (not sleep(period)) so pacing errors do
                # not accumulate and the long-run rate stays at target_fps.
                next_deadline += period
                if time.monotonic() - next_deadline > 1.0:
                    next_deadline = time.monotonic() + period
                self.frames_emitted += 1
                yield last_frame
            else:
                time.sleep(0.005)

    def stop(self) -> None:
        """Stop the capture thread and release the stream."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> RTSPSource:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _run_fps_test(uri: str, duration: float, target_fps: float) -> int:
    """Phase B test: print measured FPS every second; PASS if min >= target."""
    source = RTSPSource(uri, target_fps=target_fps)
    per_second: list[float] = []
    count = 0
    window_start: float | None = None
    test_start: float | None = None
    try:
        with source:
            for _ in source.frames():
                now = time.monotonic()
                if test_start is None:
                    # Start the clock at the first frame so RTSP connect
                    # latency is not counted against the FPS measurement.
                    test_start = window_start = now
                    continue
                count += 1
                if now - window_start >= 1.0:
                    fps = count / (now - window_start)
                    per_second.append(fps)
                    print(f"[{now - test_start:6.1f}s] ingest FPS: {fps:.2f}")
                    count = 0
                    window_start = now
                if now - test_start >= duration:
                    break
    except SourceLost as e:
        print(f"SourceLost raised: {e}")
        raise
    if not per_second:
        print("FAIL: no FPS samples collected")
        return 1
    min_fps = min(per_second)
    avg_fps = sum(per_second) / len(per_second)
    # 5% tolerance absorbs scheduler jitter on 1-second windows.
    passed = min_fps >= target_fps * 0.95
    print(
        f"\nSamples: {len(per_second)}  min: {min_fps:.2f}  avg: {avg_fps:.2f}  "
        f"target: {target_fps:.0f}  -> {'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="RTSP ingestion FPS test (Phase B)")
    parser.add_argument("--rtsp", required=True, help="RTSP URI")
    parser.add_argument("--duration", type=float, default=60.0, help="Test seconds")
    parser.add_argument("--target-fps", type=float, default=20.0, help="Target FPS")
    args = parser.parse_args()
    return _run_fps_test(args.rtsp, args.duration, args.target_fps)


if __name__ == "__main__":
    raise SystemExit(main())
