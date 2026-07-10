"""Phase C unit tests: letterbox + NMS decode (no Hailo device required)."""

import numpy as np
import pytest

from src.detector import VEHICLE_CLASSES, Detections, decode_nms_output, letterbox

MODEL_W = MODEL_H = 640


def _make_nms_result(entries):
    """Build an 80-class NMS-by-class result. entries: [(class_id, ymin, xmin, ymax, xmax, score)]."""
    result = [np.zeros((0, 5), dtype=np.float32) for _ in range(80)]
    for class_id, ymin, xmin, ymax, xmax, score in entries:
        row = np.array([[ymin, xmin, ymax, xmax, score]], dtype=np.float32)
        result[class_id] = np.concatenate([result[class_id], row])
    return result


def test_letterbox_geometry_wide_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    padded, scale, x_off, y_off = letterbox(frame, MODEL_W, MODEL_H)
    assert padded.shape == (MODEL_H, MODEL_W, 3)
    assert scale == pytest.approx(0.5)
    assert x_off == 0
    assert y_off == (640 - 360) // 2


def test_letterbox_geometry_tall_frame():
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    padded, scale, x_off, y_off = letterbox(frame, MODEL_W, MODEL_H)
    assert padded.shape == (MODEL_H, MODEL_W, 3)
    assert scale == pytest.approx(0.5)
    assert y_off == 0
    assert x_off == (640 - 360) // 2


def test_decode_maps_boxes_back_to_frame_coords():
    frame_h, frame_w = 720, 1280
    _, scale, x_off, y_off = letterbox(np.zeros((frame_h, frame_w, 3), np.uint8), MODEL_W, MODEL_H)

    # A car occupying frame pixels (100, 100)-(300, 250): forward-map into
    # normalized model space, then check decode round-trips it.
    fx1, fy1, fx2, fy2 = 100.0, 100.0, 300.0, 250.0
    xmin = (fx1 * scale + x_off) / MODEL_W
    xmax = (fx2 * scale + x_off) / MODEL_W
    ymin = (fy1 * scale + y_off) / MODEL_H
    ymax = (fy2 * scale + y_off) / MODEL_H

    nms = _make_nms_result([(2, ymin, xmin, ymax, xmax, 0.9)])
    dets = decode_nms_output(
        nms, (frame_h, frame_w), MODEL_W, MODEL_H, scale, x_off, y_off, VEHICLE_CLASSES, 0.3
    )
    assert len(dets) == 1
    assert dets.class_id[0] == 2
    assert dets.confidence[0] == pytest.approx(0.9, abs=1e-6)
    np.testing.assert_allclose(dets.xyxy[0], [fx1, fy1, fx2, fy2], atol=0.5)


def test_decode_filters_non_vehicle_classes():
    nms = _make_nms_result(
        [
            (0, 0.1, 0.1, 0.3, 0.3, 0.95),  # person -> filtered
            (2, 0.1, 0.1, 0.3, 0.3, 0.90),  # car -> kept
            (3, 0.4, 0.4, 0.6, 0.6, 0.80),  # motorcycle -> kept
            (5, 0.2, 0.5, 0.5, 0.9, 0.85),  # bus -> kept
            (7, 0.6, 0.1, 0.9, 0.4, 0.70),  # truck -> kept
            (9, 0.6, 0.6, 0.9, 0.9, 0.99),  # traffic light -> filtered
        ]
    )
    dets = decode_nms_output(nms, (640, 640), MODEL_W, MODEL_H, 1.0, 0, 0, VEHICLE_CLASSES, 0.3)
    assert len(dets) == 4
    assert set(dets.class_id.tolist()) == {2, 3, 5, 7}


def test_decode_applies_score_threshold():
    nms = _make_nms_result(
        [
            (2, 0.1, 0.1, 0.3, 0.3, 0.29),
            (2, 0.4, 0.4, 0.6, 0.6, 0.31),
        ]
    )
    dets = decode_nms_output(nms, (640, 640), MODEL_W, MODEL_H, 1.0, 0, 0, VEHICLE_CLASSES, 0.3)
    assert len(dets) == 1
    assert dets.confidence[0] == pytest.approx(0.31, abs=1e-6)


def test_decode_empty_input():
    nms = [np.zeros((0, 5), dtype=np.float32) for _ in range(80)]
    dets = decode_nms_output(nms, (640, 640), MODEL_W, MODEL_H, 1.0, 0, 0, VEHICLE_CLASSES, 0.3)
    assert isinstance(dets, Detections)
    assert len(dets) == 0
