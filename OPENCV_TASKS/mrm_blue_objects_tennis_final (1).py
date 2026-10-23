import math
from pathlib import Path

import cv2
import numpy as np

BLUE_LOWER = np.array([90, 70, 40], dtype=np.uint8)
BLUE_UPPER = np.array([135, 255, 255], dtype=np.uint8)
TENNIS_LOWER = np.array([20, 70, 70], dtype=np.uint8)
TENNIS_UPPER = np.array([45, 255, 255], dtype=np.uint8)

MIN_BLUE_AREA = 500
MIN_TENNIS_AREA = 300
MIN_TENNIS_CIRCULARITY = 0.58
MIN_TENNIS_EXTENT = 0.50
MAX_TRACK_DISTANCE = 100
CHECKERBOARD = (9, 6)
SQUARE_SIZE_CM = 2.5
BALL_DIAMETER_CM = 6.7
KERNEL = np.ones((5, 5), dtype=np.uint8)

BLUE = (255, 100, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

tracks = {}
next_track_id = 1
cm_per_pixel = None


def load_calibration():
    candidates = list(Path(".").glob("*.npz"))
    for path in candidates:
        try:
            data = np.load(path)
            keys = set(data.files)
            camera_key = next((k for k in ("camera_matrix", "cameraMatrix", "mtx", "K") if k in keys), None)
            dist_key = next((k for k in ("dist_coeffs", "distCoeffs", "dist", "distortion_coefficients") if k in keys), None)
            if camera_key and dist_key:
                print(f"Loaded calibration: {path}")
                return data[camera_key], data[dist_key]
        except (OSError, ValueError):
            pass
    print("No compatible .npz calibration file found. Camera distortion correction is disabled.")
    return None, None


def clean_mask(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)


def center_of(contour):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return (int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]))


def find_blue_objects(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_BLUE_AREA:
            continue
        center = center_of(contour)
        if center is None:
            continue
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect).astype(np.int32)
        objects.append({
            "contour": contour,
            "area_px": area,
            "center": center,
            "box_points": points,
            "length_px": max(rect[1]),
            "width_px": min(rect[1])
        })
    return objects


def find_tennis_ball(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    height, width = mask.shape[:2]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_TENNIS_AREA:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0 or x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
            continue
        if not 0.65 <= w / float(h) <= 1.45:
            continue
        if area / float(w * h) < MIN_TENNIS_EXTENT or circularity < MIN_TENNIS_CIRCULARITY:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        candidates.append({
            "contour": contour,
            "center": (int(cx), int(cy)),
            "radius_px": float(radius),
            "area_px": area
        })
    return max(candidates, key=lambda item: item["area_px"]) if candidates else None


def update_tracks(objects):
    global next_track_id
    unmatched = set(tracks.keys())
    matches = []

    pairs = []
    for index, obj in enumerate(objects):
        for track_id, track in tracks.items():
            distance = math.dist(obj["center"], track["center"])
            if distance <= MAX_TRACK_DISTANCE:
                pairs.append((distance, index, track_id))
    used_objects = set()
    used_tracks = set()
    for distance, index, track_id in sorted(pairs):
        if index in used_objects or track_id in used_tracks:
            continue
        obj = objects[index]
        track = tracks[track_id]
        step = math.dist(obj["center"], track["center"])
        track["distance_px"] += step
        track["center"] = obj["center"]
        track["contour"] = obj["contour"]
        track["object"] = obj
        track["missed"] = 0
        track["history"].append(obj["center"])
        obj["id"] = track_id
        used_objects.add(index)
        used_tracks.add(track_id)
        matches.append(obj)

    for track_id in list(tracks):
        if track_id not in used_tracks:
            tracks[track_id]["missed"] += 1
            if tracks[track_id]["missed"] > 12:
                del tracks[track_id]

    for index, obj in enumerate(objects):
        if index in used_objects:
            continue
        track_id = next_track_id
        next_track_id += 1
        tracks[track_id] = {
            "center": obj["center"],
            "contour": obj["contour"],
            "object": obj,
            "distance_px": 0.0,
            "missed": 0,
            "history": [obj["center"]]
        }
        obj["id"] = track_id
        matches.append(obj)
    return matches


def detect_checkerboard_scale(gray):
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if not found:
        return None, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    grid = corners.reshape(CHECKERBOARD[1], CHECKERBOARD[0], 2)
    horizontal = np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2).ravel()
    vertical = np.linalg.norm(grid[1:, :, :] - grid[:-1, :, :], axis=2).ravel()
    px_per_square = float(np.median(np.concatenate((horizontal, vertical))))
    if px_per_square <= 0:
        return None, None
    return SQUARE_SIZE_CM / px_per_square, corners


def draw_text(image, text, position, color=WHITE, scale=0.52):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 3, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    global cm_per_pixel
    camera_matrix, dist_coeffs = load_calibration()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam. Close other camera apps or try camera index 1.")
        return

    cv2.namedWindow("MRM Object Detection")
    cv2.namedWindow("MRM Object Information")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            if camera_matrix is not None and dist_coeffs is not None:
                try:
                    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
                except cv2.error:
                    pass

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            checker_scale, checker_corners = detect_checkerboard_scale(gray)
            if checker_scale is not None:
                cm_per_pixel = checker_scale
                cv2.drawChessboardCorners(frame, CHECKERBOARD, checker_corners, True)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blue_mask = clean_mask(cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER))
            tennis_mask = clean_mask(cv2.inRange(hsv, TENNIS_LOWER, TENNIS_UPPER))
            blue_objects = find_blue_objects(blue_mask)
            tracked_objects = update_tracks(blue_objects)
            ball = find_tennis_ball(tennis_mask)

            info = np.zeros((650, 900, 3), dtype=np.uint8)
            draw_text(info, "MRM BLUE OBJECT TRACKING", (20, 30), YELLOW, 0.7)
            draw_text(info, "Blue centres and relative positions are in image pixels.", (20, 60))
            draw_text(info, "Physical measurements are estimates and need a valid scale.", (20, 85))

            if ball:
                bx, by = ball["center"]
                cv2.drawContours(frame, [ball["contour"]], -1, GREEN, 2)
                cv2.circle(frame, (bx, by), max(1, int(ball["radius_px"])), GREEN, 2)
                cv2.circle(frame, (bx, by), 3, YELLOW, -1)
                if cm_per_pixel:
                    radius_cm = ball["radius_px"] * cm_per_pixel
                    ball_area_cm2 = math.pi * radius_cm ** 2
                    ball_text = f"Ball C:({bx},{by}) R:{ball['radius_px']:.1f}px Area:{ball_area_cm2:.2f}cm2"
                else:
                    ball_text = f"Ball C:({bx},{by}) R:{ball['radius_px']:.1f}px Area:scale unavailable"
                draw_text(frame, ball_text, (10, 78), GREEN, 0.45)
                draw_text(info, f"Tennis ball centre: ({bx}, {by}) px", (20, 125))
                draw_text(info, f"Tennis ball radius: {ball['radius_px']:.1f} px", (20, 150))
                if cm_per_pixel:
                    draw_text(info, f"Estimated tennis-ball radius: {radius_cm:.2f} cm", (20, 175))
                    draw_text(info, f"Estimated projected area: {ball_area_cm2:.2f} cm^2", (20, 200))
                else:
                    draw_text(info, "Ball cm measurements: checkerboard scale unavailable", (20, 175), YELLOW)
            else:
                draw_text(info, "Tennis ball not detected in this frame.", (20, 125), YELLOW)

            y_text = 235
            if not tracked_objects:
                draw_text(info, "No blue objects detected.", (20, y_text))
            for obj in sorted(tracked_objects, key=lambda item: item["id"]):
                track_id = obj["id"]
                cx, cy = obj["center"]
                contour = obj["contour"]
                cv2.drawContours(frame, [contour], -1, BLUE, 2)
                cv2.circle(frame, (cx, cy), 4, YELLOW, -1)
                track = tracks[track_id]
                if len(track["history"]) > 1:
                    points = np.array(track["history"], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], False, (255, 0, 255), 2)

                if cm_per_pixel:
                    length_cm = obj["length_px"] * cm_per_pixel
                    width_cm = obj["width_px"] * cm_per_pixel
                    area_cm2 = obj["area_px"] * cm_per_pixel ** 2
                    distance_cm = track["distance_px"] * cm_per_pixel
                    label = f"ID {track_id} L:{length_cm:.1f} W:{width_cm:.1f} cm"
                    distance_text = f"{distance_cm:.2f} cm"
                else:
                    length_cm = width_cm = area_cm2 = distance_cm = None
                    label = f"ID {track_id} ({cx},{cy})"
                    distance_text = "scale unavailable"

                draw_text(frame, label, (max(5, cx - 40), max(95, cy - 12)), BLUE, 0.42)
                if ball:
                    dx = cx - ball["center"][0]
                    dy = cy - ball["center"][1]
                    cv2.line(frame, ball["center"], (cx, cy), MAGENTA, 1)
                    relative = f"dx={dx:+d}px dy={dy:+d}px"
                    relative_distance = f"{math.hypot(dx, dy) * cm_per_pixel:.2f}cm" if cm_per_pixel else f"{math.hypot(dx, dy):.1f}px"
                else:
                    dx = dy = None
                    relative = "Ball unavailable"
                    relative_distance = "N/A"

                draw_text(info, f"Blue ID {track_id}: centre=({cx},{cy}) px", (20, y_text))
                y_text += 25
                draw_text(info, f"  Relative to ball: {relative}; separation={relative_distance}", (35, y_text))
                y_text += 25
                if cm_per_pixel:
                    draw_text(info, f"  Size: {length_cm:.2f} x {width_cm:.2f} cm; contour area: {area_cm2:.2f} cm^2", (35, y_text))
                    y_text += 25
                    draw_text(info, f"  Distance travelled (estimated): {distance_text}", (35, y_text))
                else:
                    draw_text(info, "  Size/area/distance in cm unavailable: show checkerboard on object plane.", (35, y_text), YELLOW)
                y_text += 35

            scale_text = f"Scale: {cm_per_pixel:.5f} cm/px (checkerboard visible)" if cm_per_pixel else "Scale unavailable: show checkerboard in the same plane as objects"
            draw_text(frame, scale_text, (10, frame.shape[0] - 15), WHITE, 0.48)
            draw_text(frame, "Q: quit | R: reset tracking", (10, 28), WHITE, 0.50)

            cv2.imshow("MRM Object Detection", frame)
            cv2.imshow("MRM Object Information", info)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                tracks.clear()
                next_track_id = 1

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
