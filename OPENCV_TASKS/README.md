# Blue Object Selection and Tennis Ball Detection using OpenCV


## 1. Approach

Each webcam frame is converted from BGR to HSV colour space. HSV thresholds are used to create two separate binary masks: one for blue objects and another for the yellow-green colour of a tennis ball. Keeping the masks separate helps the program treat the blue object and tennis ball as different objects.

The program finds contours in the blue mask. It does not assume that a blue object is circular; contours, bounding boxes, and contour moments are used so that irregular shapes can also be detected. The user selects a blue object by left-clicking inside its detected outline. The program then tracks the selected object between frames by matching the closest contour centre to its previous centre and draws a trail showing its movement.

The tennis-ball mask is processed independently. For each accepted tennis-ball candidate, the program checks its area, circularity, bounding-box proportions, and whether it is clipped by the edge of the frame. It then estimates the centre and radius using the minimum enclosing circle and calculates contour area. When both the selected blue object and a tennis-ball candidate are visible, it draws a line between their centres and displays the blue object's relative image position (`dx`, `dy`) and centre-to-centre distance in pixels.

## 2. Method used

1. Read frames from the live webcam using `cv2.VideoCapture(0)`.
2. Mirror the frame for easier interaction and convert it from BGR to HSV.
3. Use `cv2.inRange()` with separate HSV thresholds to create the blue-object and tennis-ball masks.
4. Apply morphological opening and closing to reduce small specks and fill small gaps in each mask.
5. Find contours with `cv2.findContours()` and reject regions below minimum area thresholds.
6. For blue objects, use `cv2.contourArea()`, `cv2.boundingRect()`, and `cv2.moments()` to calculate area, bounding box, and centre. No circular shape is assumed for blue objects.
7. Select a blue object with a mouse click and track it using nearest-centre matching across frames.
8. For tennis-ball candidates, use the separate yellow-green mask, reject small or strongly irregular regions, check the candidate proportions, and estimate the centre and radius with `cv2.minEnclosingCircle()`. Candidates touching the frame edge are ignored because their radius estimate may be unreliable.
9. If the selected blue object and tennis ball are both detected, calculate `dx`, `dy`, and Euclidean centre-to-centre distance in image pixels, then draw a connecting line.
10. Display the measurements and annotations directly on the camera stream.


## 3. Other possible methods

- **HSV trackbars:** allow colour thresholds to be adjusted live while testing.
- **CSRT or KCF tracker:** can track a selected bounding box after selection; these trackers may be less dependent on colour segmentation, but availability varies by OpenCV installation.
- **Object detection model such as YOLO:** useful when objects must be recognised by class rather than only by colour; it requires a suitable trained model and more setup.
- **Background subtraction:** useful when the target moves against a mostly static background, although it does not identify objects by colour alone.
- **Optical flow:** can estimate image motion between frames and may help track movement, but can drift or fail when the appearance changes.
