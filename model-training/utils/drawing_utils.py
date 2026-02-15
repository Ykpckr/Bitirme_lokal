import cv2

def draw_detections(frame, detections, color=(0, 255, 0)):
    """
    Draw YOLO detections: [x1,y1,x2,y2,cls,conf]
    """
    for det in detections:
        x1, y1, x2, y2, cls, conf = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            frame,
            f"{int(cls)} {conf:.2f}",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def bbox_to_yolo_format(width, height, box):
    """Convert [x1,y1,x2,y2] to YOLO normalized (x_center, y_center, w, h)."""
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2 / width
    yc = (y1 + y2) / 2 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return xc, yc, w, h


def add_text_overlay(frame, text, pos=(20, 20)):
    cv2.putText(
        frame, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 255, 255), 2, cv2.LINE_AA
    )
    return frame

