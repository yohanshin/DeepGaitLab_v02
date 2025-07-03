import cv2
import numpy as np

def intersect(area1, area2):
    # Check if two rectangles (areas) intersect
    x1_min, y1_min, x1_max, y1_max = area1
    x2_min, y2_min, x2_max, y2_max = area2

    return not (x1_max < x2_min or x1_min > x2_max or
                y1_max < y2_min or y1_min > y2_max)

def visualize_bbox(bboxes, img, color=(0, 255, 0), thickness=2, font_scale=0.6, bbox_id=0):
    img_copy = img.copy()
    
    if bboxes.dtype != np.int32:
        bboxes = bboxes.astype(np.int32)

    placed_text_areas = []

    for i, bbox in enumerate(bboxes):
        x1, y1, w, h = bbox
        x2 = int(x1 + w)
        y2 = int(y1 + h)

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)

        display_text = f"Bbox ID: {bbox_id} | {x1} {y1} {w} {h}"

        (text_width, text_height), baseline = cv2.getTextSize(
            display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # Initially place text above the bounding box
        text_x, text_y = x1, y1 - 5

        # Shift text down if it goes off the top of the image
        if text_y - text_height - baseline < 0:
            text_y = y1 + text_height + baseline + 5

        # Dynamically shift text down to avoid overlap with existing texts
        text_area = (text_x, text_y - text_height - baseline, text_x + text_width, text_y)

        while any(intersect(text_area, existing_area) for existing_area in placed_text_areas):
            text_y += text_height + baseline + 5
            if text_y > img_copy.shape[0]:  # If beyond bottom boundary, reset near top
                text_y = text_height + baseline + 5
                text_x += 5
            text_area = (text_x, text_y - text_height - baseline, text_x + text_width, text_y)

        placed_text_areas.append(text_area)

        # Draw semi-transparent background rectangle
        overlay = img_copy.copy()
        cv2.rectangle(overlay,
                      (text_area[0] - 2, text_area[1] - 2),
                      (text_area[2] + 2, text_area[3] + 2),
                      (0, 0, 0), -1)

        alpha = 0.6
        img_copy = cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0)

        # Draw text
        cv2.putText(img_copy, display_text, (text_area[0], text_area[3] - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return img_copy