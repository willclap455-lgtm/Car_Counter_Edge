"""Hailo YOLOv8 vehicle detector (Phase C).

HailoYoloDetector loads a yolov8m HEF (with on-chip NMS postprocess) onto a
shared VDevice via the hailo-apps HailoInfer wrapper and exposes a
synchronous ``detect(frame)`` that returns detections filtered to the
requested COCO classes (car=2, motorcycle=3, bus=5, truck=7).

Pre/postprocessing (letterbox + NMS-output decoding) are pure functions so
they can be unit-tested without Hailo hardware.

On-device self test (Phase C test)::

    python3 -m src.detector --hef /usr/local/hailo/resources/models/hailo10h/yolov8m.hef \
        --image <test.jpg> [--runs 50]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)

VEHICLE_CLASSES: frozenset[int] = frozenset({2, 3, 5, 7})
COCO_VEHICLE_NAMES: dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Detections:
    """Detection results in original-frame pixel coordinates."""

    xyxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    confidence: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    class_id: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))

    def __len__(self) -> int:
        return len(self.xyxy)


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def class_agnostic_nms(detections: Detections, iou_threshold: float = 0.65) -> Detections:
    """Suppress overlapping boxes across classes, keeping the highest score.

    The HEF's on-chip NMS runs per class, so one vehicle detected as e.g.
    both "car" and "truck" survives with two overlapping boxes — which the
    tracker then double-counts. This pass removes such duplicates.
    """
    n = len(detections)
    if n <= 1:
        return detections
    order = np.argsort(-detections.confidence)
    keep: list[int] = []
    for idx in order:
        if all(_iou(detections.xyxy[idx], detections.xyxy[k]) < iou_threshold for k in keep):
            keep.append(int(idx))
    if len(keep) == n:
        return detections
    keep_arr = np.asarray(sorted(keep), dtype=np.int64)
    return Detections(
        xyxy=detections.xyxy[keep_arr],
        confidence=detections.confidence[keep_arr],
        class_id=detections.class_id[keep_arr],
    )


def letterbox(frame: np.ndarray, model_w: int, model_h: int) -> tuple[np.ndarray, float, int, int]:
    """
    Resize with unchanged aspect ratio and pad to (model_h, model_w).

    Args:
        frame: Input image (H, W, 3) uint8.
        model_w: Model input width.
        model_h: Model input height.

    Returns:
        (padded_image, scale, x_offset, y_offset) — everything needed to map
        model-space boxes back to frame coordinates.
    """
    img_h, img_w = frame.shape[:2]
    scale = min(model_w / img_w, model_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((model_h, model_w, 3), 114, dtype=np.uint8)
    x_off = (model_w - new_w) // 2
    y_off = (model_h - new_h) // 2
    padded[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return padded, scale, x_off, y_off


def decode_nms_output(
    nms_result: list[np.ndarray] | np.ndarray,
    frame_shape: tuple[int, int],
    model_w: int,
    model_h: int,
    scale: float,
    x_off: int,
    y_off: int,
    classes: frozenset[int] | set[int],
    score_threshold: float,
) -> Detections:
    """
    Decode Hailo on-chip NMS output into frame-space detections.

    The NMS-by-class output is one array per COCO class, each row being
    ``[ymin, xmin, ymax, xmax, score]`` normalized to the model input.

    Args:
        nms_result: Per-class detection arrays from the HEF NMS node.
        frame_shape: (height, width) of the original frame.
        model_w: Model input width.
        model_h: Model input height.
        scale: Letterbox scale factor.
        x_off: Letterbox horizontal padding offset.
        y_off: Letterbox vertical padding offset.
        classes: COCO class ids to keep.
        score_threshold: Minimum confidence.

    Returns:
        Detections in original-frame pixel coordinates.
    """
    img_h, img_w = frame_shape
    boxes: list[list[float]] = []
    scores: list[float] = []
    class_ids: list[int] = []

    for class_id, class_dets in enumerate(nms_result):
        if class_id not in classes or len(class_dets) == 0:
            continue
        for det in class_dets:
            score = float(det[4])
            if score < score_threshold:
                continue
            ymin, xmin, ymax, xmax = (float(v) for v in det[:4])
            # Normalized model space -> letterboxed pixels -> frame pixels.
            x1 = (xmin * model_w - x_off) / scale
            y1 = (ymin * model_h - y_off) / scale
            x2 = (xmax * model_w - x_off) / scale
            y2 = (ymax * model_h - y_off) / scale
            x1, x2 = max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))
            y1, y2 = max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(class_id)

    if not boxes:
        return Detections()
    return Detections(
        xyxy=np.asarray(boxes, dtype=np.float32),
        confidence=np.asarray(scores, dtype=np.float32),
        class_id=np.asarray(class_ids, dtype=np.int32),
    )


class HailoYoloDetector:
    """YOLOv8 detector running on a Hailo device (shared VDevice).

    Args:
        hef_path: Path to the compiled yolov8m HEF (with NMS postprocess).
        classes: COCO class ids to keep (default vehicles {2, 3, 5, 7}).
        score_threshold: Minimum detection confidence.
    """

    def __init__(
        self,
        hef_path: str,
        classes: set[int] | frozenset[int] = VEHICLE_CLASSES,
        score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.65,
    ) -> None:
        # Imported lazily: pulls in hailo_platform, which needs libhailort.
        from hailo_apps.python.core.common.hailo_inference import HailoInfer

        self.hef_path = str(hef_path)
        self.classes = frozenset(classes)
        self.score_threshold = float(score_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self._infer = HailoInfer(self.hef_path, batch_size=1)
        self.input_h, self.input_w, _ = self._infer.get_input_shape()
        logger.info(
            "HailoYoloDetector ready: %s (input %dx%d, classes %s)",
            self.hef_path,
            self.input_w,
            self.input_h,
            sorted(self.classes),
        )

    def detect(self, frame: np.ndarray, timeout_ms: int = 5000) -> Detections:
        """Run inference on one BGR frame and return filtered detections."""
        preprocessed, scale, x_off, y_off = letterbox(frame, self.input_w, self.input_h)
        # HEFs from the model zoo expect RGB input; OpenCV frames are BGR.
        preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

        results: list = []

        def on_done(completion_info, bindings_list) -> None:
            if completion_info.exception:
                logger.error("Inference error: %s", completion_info.exception)
                return
            for bindings in bindings_list:
                results.append(bindings.output().get_buffer())

        job = self._infer.run([preprocessed], on_done)
        job.wait(timeout_ms)
        if not results:
            return Detections()
        detections = decode_nms_output(
            results[0],
            frame.shape[:2],
            self.input_w,
            self.input_h,
            scale,
            x_off,
            y_off,
            self.classes,
            self.score_threshold,
        )
        # On-chip NMS is per class; drop cross-class duplicates of one vehicle.
        return class_agnostic_nms(detections, self.nms_iou_threshold)

    def close(self) -> None:
        """Release the configured model and VDevice resources."""
        self._infer.close()

    def __enter__(self) -> HailoYoloDetector:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _run_device_test(hef: str, image_path: str | None, runs: int) -> int:
    """Phase C on-device test: non-empty detections and latency <= 50 ms."""
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"FAIL: cannot read image {image_path}")
            return 1
    else:
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        print("WARNING: no --image given; using random noise (detections may be empty)")

    with HailoYoloDetector(hef) as det:
        det.detect(frame)  # Warm-up (not timed)
        latencies = []
        last = Detections()
        for _ in range(runs):
            t0 = time.perf_counter()
            last = det.detect(frame)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    lat = np.asarray(latencies)
    print(f"runs: {runs}  median: {np.median(lat):.1f}ms  p95: {np.percentile(lat, 95):.1f}ms")
    print(f"detections on test frame: {len(last)}")
    for i in range(len(last)):
        cid = int(last.class_id[i])
        print(
            f"  {COCO_VEHICLE_NAMES.get(cid, cid)}: conf={last.confidence[i]:.2f} "
            f"box={last.xyxy[i].astype(int).tolist()}"
        )
    ok_latency = float(np.median(lat)) <= 50.0
    ok_dets = len(last) > 0 if image_path else True
    print(f"latency <= 50ms: {'PASS' if ok_latency else 'FAIL'}")
    if image_path:
        print(f"non-empty detections: {'PASS' if ok_dets else 'FAIL'}")
    return 0 if (ok_latency and ok_dets) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Hailo YOLOv8 detector self-test (Phase C)")
    parser.add_argument("--hef", required=True, help="Path to yolov8m HEF")
    parser.add_argument("--image", default=None, help="Test image with vehicles")
    parser.add_argument("--runs", type=int, default=50, help="Timed inference runs")
    args = parser.parse_args()
    return _run_device_test(args.hef, args.image, args.runs)


if __name__ == "__main__":
    raise SystemExit(main())
