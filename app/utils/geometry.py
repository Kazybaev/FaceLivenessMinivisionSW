from __future__ import annotations

from app.core.models import BoundingBox


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x2, a.y2
    bx1, by1, bx2, by2 = b.x, b.y, b.x2, b.y2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def is_detection_near_face(obj_box: BoundingBox, face_box: BoundingBox, margin: int = 60) -> bool:
    expanded = BoundingBox(
        x=max(0, face_box.x - margin),
        y=max(0, face_box.y - margin),
        width=face_box.width + 2 * margin,
        height=face_box.height + 2 * margin,
    )
    return bbox_iou(obj_box, expanded) > 0.0
