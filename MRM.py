
import cv2
import numpy as np

# Assuming the side length of the printed black marker square as 5cm.
MARKER_LENGTH = 0.05

# Open laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

# Required ArUco dictionary families
dictionary_ids = {
    "4X4": cv2.aruco.DICT_4X4_50,
    "5X5": cv2.aruco.DICT_5X5_50,
    "6X6": cv2.aruco.DICT_6X6_50
}

# Creating a detector for each dictionary
detectors = {}

for name, dictionary_id in dictionary_ids.items():
    dictionary = cv2.aruco.getPredefinedDictionary(
        dictionary_id
    )
    parameters = cv2.aruco.DetectorParameters()
    detectors[name] = cv2.aruco.ArucoDetector(
        dictionary, parameters
    )

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam frame.")
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Approximate camera calibration parameters.
    focal_length = float(width)

    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((5, 1), dtype=np.float64)

    for name, detector in detectors.items():

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is None:
            continue

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, corner in enumerate(corners):
            image_points = corner.reshape(4, 2).astype(
                np.float32
            )

            half = MARKER_LENGTH / 2

            # 3D positions of the four marker corners
            object_points = np.array([
                [-half,  half, 0],
                [ half,  half, 0],
                [ half, -half, 0],
                [-half, -half, 0]
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not success or tvec[2, 0] <= 0:
                continue


            marker_id = int(ids[i])

            # Draw 3D coordinate axes
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                distortion,
                rvec,
                tvec,
                MARKER_LENGTH * 0.5
            )

            # Position relative to the camera, in metres
            x, y, z = tvec.flatten()

            # Rotation vector
            rx, ry, rz = rvec.flatten()

            label = f"{name} ID:{marker_id}"

            position_text = (
                f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f} m"
            )

            rotation_text = (
                f"rX:{rx:.2f} rY:{ry:.2f} rZ:{rz:.2f}"
            )

            point = tuple(image_points[0].astype(int))
            text_x = max(5, point[0])
            text_y = max(20, point[1] - 35)

            cv2.putText(
                frame, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 2
            )

            cv2.putText(
                frame, position_text, (text_x, text_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1
            )

            cv2.putText(
                frame, rotation_text, (text_x, text_y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 200, 0), 1
            )

    cv2.imshow("MRM ArUco Pose Estimation", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()