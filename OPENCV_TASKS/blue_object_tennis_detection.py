"""MRM Taskphase: Blue Object Selection and Tennis Ball Detection using OpenCV.

Controls:
    Left-click inside a blue object: select it for tracking
    R: reset the current selection
    Q: quit

The blue object is detected using its colour and contour, not a circular-shape
assumption. Tennis-ball candidates are found in a separate yellow-green mask.
All positions and distances displayed are image-pixel measurements.
"""

from collections import deque
import math

import cv2
import numpy as np


# HSV limits are starting values. Adjust them if room lighting or the camera
# makes the blue object or tennis ball difficult to detect.
BLUE_LOWER = np.array([90, 70, 40], dtype=np.uint8)
BLUE_UPPER = np.array([135, 255, 255], dtype=np.uint8)

TENNIS_LOWER = np.array([20, 70, 70], dtype=np.uint8)
TENNIS_UPPER = np.array([45, 255, 255], dtype=np.uint8)

MIN_BLUE_AREA = 500       # Ignore blue regions smaller than this (px^2)
MIN_TENNIS_AREA = 300     # Ignore tennis-coloured regions smaller than this (px^2)
MIN_TENNIS_CIRCULARITY = 0.58  # Reject very irregular tennis-coloured regions
MIN_TENNIS_EXTENT = 0.50       # Minimum area / bounding-box area
MIN_TENNIS_ASPECT = 0.70       # Minimum width-to-height ratio for a ball
MAX_TENNIS_ASPECT = 1.40       # Maximum width-to-height ratio for a ball
MAX_TRACK_DISTANCE = 180       # Largest centre jump allowed between frames (pixels)

KERNEL = np.ones((5, 5), dtype=np.uint8)
WINDOW_NAME = "MRM: Blue Object Tracking + Tennis Ball Detection"

# OpenCV uses BGR colour order, not RGB.
BLUE = (255, 80, 0)
SELECTED_YELLOW = (0, 255, 255)
BALL_GREEN = (0, 255, 0)
RELATIVE_LINE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Updated each frame; the mouse callback uses these contours for selection.
current_blue_contours = []
selected_object = {"active": False, "last_center": None}
center_history = deque(maxlen=60)


def contour_center(contour):
    """Calculate a contour's centroid as (x, y), or return None if undefined."""
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    return (center_x, center_y)


def clean_mask(mask):
    """Reduce tiny colour specks and fill small gaps in a binary mask."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    return mask


def draw_label(frame, text, position, color=WHITE, scale=0.55, thickness=2):
    """Draw readable text with a dark outline."""
    x, y = position
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def on_mouse(event, x, y, flags, param):
    """Select a blue contour by clicking inside its visible shape."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for contour in current_blue_contours:
        # pointPolygonTest supports irregular shapes; no circle is assumed.
        if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
            center = contour_center(contour)
            if center is not None:
                selected_object["active"] = True
                selected_object["last_center"] = center
                center_history.clear()
                center_history.append(center)
                print(f"Selected blue object at image centre {center}.")
                return

    print("No blue object selected. Click inside a blue object's detected outline.")


def find_blue_objects(mask):
    """Return blue contour information: area, bounding box, and centre."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    objects = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_BLUE_AREA:
            continue

        center = contour_center(contour)
        if center is None:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        objects.append({
            "contour": contour,
            "area": area,
            "box": (x, y, w, h),
            "center": center,
        })

    return objects


def find_tennis_ball_candidates(mask):
    """Find round yellow-green regions and estimate their centre and radius.

    Shape checks are used only for tennis-ball candidates. Blue objects are
    detected by contour and may have any shape.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    frame_height, frame_width = mask.shape[:2]
    edge_margin = 2

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_TENNIS_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        # Circularity is close to 1 for a circle and lower for irregular shapes.
        circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
        if circularity < MIN_TENNIS_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        # Ignore clipped candidates because their estimated radius is unreliable.
        if (x <= edge_margin or y <= edge_margin or
                x + w >= frame_width - edge_margin or
                y + h >= frame_height - edge_margin):
            continue

        aspect_ratio = w / float(h)
        if not MIN_TENNIS_ASPECT <= aspect_ratio <= MAX_TENNIS_ASPECT:
            continue

        extent = area / float(w * h)
        if extent < MIN_TENNIS_EXTENT:
            continue

        (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
        center = (int(circle_x), int(circle_y))
        candidates.append({
            "contour": contour,
            "area": area,
            "box": (x, y, w, h),
            "center": center,
            "radius": float(radius),
            "circularity": circularity,
        })

    return candidates


def main():
    global current_blue_contours

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open the webcam.")
        print("Check camera permissions, close apps using the camera, or try index 1.")
        return

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Could not read a frame from the webcam.")
                break

            # Mirror the view for easier clicking, like a mirror preview.
            frame = cv2.flip(frame, 1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Separate masks ensure blue-object detection and tennis-ball
            # detection are handled independently.
            blue_mask = clean_mask(cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER))
            tennis_mask = clean_mask(cv2.inRange(hsv, TENNIS_LOWER, TENNIS_UPPER))

            blue_objects = find_blue_objects(blue_mask)
            current_blue_contours = [obj["contour"] for obj in blue_objects]

            # Draw each blue object: contour, bounding box, area, and centre.
            for obj in blue_objects:
                x, y, w, h = obj["box"]
                cx, cy = obj["center"]
                cv2.drawContours(frame, [obj["contour"]], -1, BLUE, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), BLUE, 2)
                cv2.circle(frame, (cx, cy), 4, BLUE, -1)
                draw_label(
                    frame,
                    f"Blue | Area: {int(obj['area'])} px^2",
                    (x, max(y - 8, 20)),
                    BLUE,
                    scale=0.5,
                )

            # Track the selected blue object by nearest-centre matching.
            # This is a simple tracker and can confuse objects that cross paths.
            tracked_object = None
            tracking_found = False

            if selected_object["active"] and selected_object["last_center"] is not None:
                previous_x, previous_y = selected_object["last_center"]

                if blue_objects:
                    def centre_distance(obj):
                        cx, cy = obj["center"]
                        return math.hypot(cx - previous_x, cy - previous_y)

                    nearest = min(blue_objects, key=centre_distance)
                    if centre_distance(nearest) <= MAX_TRACK_DISTANCE:
                        tracked_object = nearest
                        tracking_found = True
                        selected_object["last_center"] = nearest["center"]
                        center_history.append(nearest["center"])

            # Highlight the selected object and show its bounding-box dimensions.
            if tracked_object is not None:
                x, y, w, h = tracked_object["box"]
                cx, cy = tracked_object["center"]
                cv2.drawContours(
                    frame, [tracked_object["contour"]], -1, SELECTED_YELLOW, 3
                )
                cv2.rectangle(frame, (x, y), (x + w, y + h), SELECTED_YELLOW, 3)
                cv2.circle(frame, (cx, cy), 6, SELECTED_YELLOW, -1)
                draw_label(
                    frame,
                    f"Selected | Box: x={x}, y={y}, w={w}, h={h} | Centre: ({cx}, {cy})",
                    (10, 82),
                    SELECTED_YELLOW,
                    scale=0.5,
                )

            # Draw the path followed by the selected blue object's centre.
            if len(center_history) >= 2:
                path_points = np.array(center_history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [path_points], False, SELECTED_YELLOW, 2)

            if selected_object["active"]:
                if tracking_found:
                    status = "Selected blue object: TRACKING"
                else:
                    status = "Selected blue object: temporarily lost"
                draw_label(frame, status, (10, 55), SELECTED_YELLOW, scale=0.6)

            # Tennis-ball detection is independent of blue-object selection.
            tennis_candidates = find_tennis_ball_candidates(tennis_mask)
            for candidate in tennis_candidates:
                cx, cy = candidate["center"]
                radius = max(1, int(round(candidate["radius"])))
                cv2.drawContours(
                    frame, [candidate["contour"]], -1, BALL_GREEN, 2
                )
                cv2.circle(frame, (cx, cy), radius, BALL_GREEN, 2)
                cv2.circle(frame, (cx, cy), 4, BALL_GREEN, -1)
                x, y, w, h = candidate["box"]
                draw_label(
                    frame,
                    f"Ball C:({cx},{cy}) R:{candidate['radius']:.1f}px A:{int(candidate['area'])}px^2",
                    (x, max(y - 8, 20)),
                    BALL_GREEN,
                    scale=0.48,
                )

            # Use the largest accepted tennis-coloured contour as the reference
            # ball if more than one candidate is present.
            tennis_ball = max(tennis_candidates, key=lambda item: item["area"]) if tennis_candidates else None

            # Show the selected blue object's image-plane position relative to
            # the tennis ball, when both are detected in the current frame.
            if tracked_object is not None and tennis_ball is not None:
                blue_cx, blue_cy = tracked_object["center"]
                ball_cx, ball_cy = tennis_ball["center"]
                delta_x = blue_cx - ball_cx
                delta_y = blue_cy - ball_cy
                distance_px = math.hypot(delta_x, delta_y)

                cv2.line(
                    frame,
                    tennis_ball["center"],
                    tracked_object["center"],
                    RELATIVE_LINE,
                    2,
                )
                draw_label(
                    frame,
                    f"Blue relative to ball: dx={delta_x:+d} px, dy={delta_y:+d} px, distance={distance_px:.1f} px",
                    (10, 110),
                    RELATIVE_LINE,
                    scale=0.52,
                )

            draw_label(
                frame,
                "Click a blue object to track | R: reset | Q: quit",
                (10, 27),
                WHITE,
                scale=0.58,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                selected_object["active"] = False
                selected_object["last_center"] = None
                center_history.clear()
                print("Selection reset. Click inside a blue object to track it.")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
