
import cv2
import numpy as np

# Number of INNER corners
CHECKERBOARD = (9, 6)

# Captured 15 different checkerboard views
REQUIRED_IMAGES = 15

# Physical size of one square, in metres
SQUARE_SIZE = 0.025

# Prepare the checkerboard's 3D corner coordinates
objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

# Store corresponding 3D and 2D points
object_points = []
image_points = []

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

print("Show the checkerboard to the webcam.")
print("Press SPACE when the full board is visible.")
print("Press Q to quit.")

image_size = None

while len(object_points) < REQUIRED_IMAGES:
    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    found, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, None
    )

    display = frame.copy()

    if found:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS
                + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        cv2.drawChessboardCorners(
            display, CHECKERBOARD, corners, found
        )

        cv2.putText(
            display,
            "Board detected - press SPACE to capture",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )
    else:
        cv2.putText(
            display,
            "Show the complete checkerboard",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2
        )

    cv2.putText(
        display,
        f"Images captured: {len(object_points)}/{REQUIRED_IMAGES}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 0), 2
    )

    cv2.imshow("Camera Calibration", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32 and found:  # SPACE
        object_points.append(objp.copy())
        image_points.append(corners.copy())
        print(f"Captured image {len(object_points)}")

cap.release()
cv2.destroyAllWindows()

if len(object_points) < 10:
    print("Not enough valid images. Capture at least 10 views.")
    raise SystemExit

# Calculate camera matrix and distortion coefficients
rms, camera_matrix, distortion, rvecs, tvecs = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )
)

# Saving the calibration for ArUco marker detection.
np.savez(
    "camera_calibration.npz",
    camera_matrix=camera_matrix,
    distortion=distortion,
    image_width=image_size[0],
    image_height=image_size[1]
)

print("\nCalibration complete!")
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", distortion)
print("RMS reprojection error:", rms)
print("Saved to camera_calibration.npz")
