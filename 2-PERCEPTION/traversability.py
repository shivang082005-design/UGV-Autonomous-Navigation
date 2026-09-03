import cv2
import numpy as np

# ---------------------------------------
# 1. INPUT VIDEO
# ---------------------------------------

video_path = "1-DATA/SKD-1_Outdoor.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Video opened successfully!")


# ---------------------------------------
# 2. PROCESS VIDEO
# ---------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize
    frame = cv2.resize(frame, (960, 540))

    height, width = frame.shape[:2]

    # ---------------------------------------
    # 3. TAKE ONLY LEFT CAMERA VIEW
    # ---------------------------------------

    camera = frame[:, :int(width * 0.48)]

    cam_height, cam_width = camera.shape[:2]

    # ---------------------------------------
    # 4. LOWER TERRAIN REGION
    # ---------------------------------------

    roi_start = int(cam_height * 0.40)

    terrain = camera[roi_start:cam_height, :]

    # ---------------------------------------
    # 5. HSV CONVERSION
    # ---------------------------------------

    hsv = cv2.cvtColor(terrain, cv2.COLOR_BGR2HSV)

    # ---------------------------------------
    # 6. DETECT GRASS
    # ---------------------------------------

    grass_lower = np.array([30, 25, 20])
    grass_upper = np.array([100, 255, 240])

    grass_mask = cv2.inRange(
        hsv,
        grass_lower,
        grass_upper
    )

    # ---------------------------------------
    # 7. DETECT DIRT / SOIL
    # ---------------------------------------

    dirt_lower = np.array([5, 15, 30])
    dirt_upper = np.array([30, 200, 230])

    dirt_mask = cv2.inRange(
        hsv,
        dirt_lower,
        dirt_upper
    )

    # ---------------------------------------
    # 8. DETECT CONCRETE
    # ---------------------------------------

    concrete_lower = np.array([0, 0, 50])
    concrete_upper = np.array([180, 70, 230])

    concrete_mask = cv2.inRange(
        hsv,
        concrete_lower,
        concrete_upper
    )

    # ---------------------------------------
    # 9. COMBINE GROUND MASKS
    # ---------------------------------------

    mask = cv2.bitwise_or(
        grass_mask,
        dirt_mask
    )

    mask = cv2.bitwise_or(
        mask,
        concrete_mask
    )

    # ---------------------------------------
    # 10. REMOVE SMALL NOISE
    # ---------------------------------------

    kernel = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ---------------------------------------
    # 11. CREATE FULL MASK
    # ---------------------------------------

    full_mask = np.zeros(
        (cam_height, cam_width),
        dtype=np.uint8
    )

    full_mask[
        roi_start:cam_height,
        :
    ] = mask

    # ---------------------------------------
    # 12. CREATE OVERLAY
    # ---------------------------------------

    result = camera.copy()

    overlay = camera.copy()

    overlay[full_mask > 0] = (0, 255, 0)

    result = cv2.addWeighted(
        camera,
        0.65,
        overlay,
        0.35,
        0
    )

    # ---------------------------------------
    # 13. ROI LINE
    # ---------------------------------------

    cv2.line(
        result,
        (0, roi_start),
        (cam_width, roi_start),
        (255, 0, 0),
        2
    )

    # =====================================================
    # 14. LEFT / CENTER / RIGHT ANALYSIS
    # =====================================================

    third = cam_width // 3

    left_mask = mask[:, :third]

    center_mask = mask[:, third:2 * third]

    right_mask = mask[:, 2 * third:]

    # Calculate percentage for each region
    left_score = (
        cv2.countNonZero(left_mask) /
        left_mask.size
    ) * 100

    center_score = (
        cv2.countNonZero(center_mask) /
        center_mask.size
    ) * 100

    right_score = (
        cv2.countNonZero(right_mask) /
        right_mask.size
    ) * 100

    # ---------------------------------------
    # 15. FIND BEST DIRECTION
    # ---------------------------------------

    scores = {
        "LEFT": left_score,
        "CENTER": center_score,
        "RIGHT": right_score
    }

    best_direction = max(
        scores,
        key=scores.get
    )

    # ---------------------------------------
    # 16. DRAW REGION LINES
    # ---------------------------------------

    cv2.line(
        result,
        (third, roi_start),
        (third, cam_height),
        (255, 255, 255),
        2
    )

    cv2.line(
        result,
        (2 * third, roi_start),
        (2 * third, cam_height),
        (255, 255, 255),
        2
    )

    # ---------------------------------------
    # 17. DISPLAY SCORES
    # ---------------------------------------

    cv2.putText(
        result,
        f"LEFT: {left_score:.1f}%",
        (10, roi_start + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"CENTER: {center_score:.1f}%",
        (third + 10, roi_start + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"RIGHT: {right_score:.1f}%",
        (2 * third + 10, roi_start + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ---------------------------------------
    # 18. BEST DIRECTION
    # ---------------------------------------

    cv2.putText(
        result,
        f"Recommended: {best_direction}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # ---------------------------------------
    # 19. OVERALL TRAVERSABILITY
    # ---------------------------------------

    terrain_pixels = mask.size

    traversable_pixels = cv2.countNonZero(mask)

    score = (
        traversable_pixels /
        terrain_pixels
    ) * 100

    cv2.putText(
        result,
        f"Traversability: {score:.1f}%",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ---------------------------------------
    # 20. DISPLAY
    # ---------------------------------------

    cv2.imshow(
        "UGV Traversability",
        result
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------------------
# 21. CLEANUP
# ---------------------------------------

cap.release()

cv2.destroyAllWindows()

print("Traversability analysis completed!")