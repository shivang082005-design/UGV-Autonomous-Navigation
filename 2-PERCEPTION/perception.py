import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# 1. INPUT VIDEO
# ============================================================

video_path = "1-DATA/SKD-1_Outdoor.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Video opened successfully!")


# ============================================================
# 2. LOAD YOLO MODEL
# ============================================================

model = YOLO("yolo11n.pt")


# ============================================================
# 3. COCO OBJECTS THAT CAN BE OBSTACLES
# ============================================================

obstacle_classes = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "horse",
    "sheep",
    "cow",
    "boat"
}


# ============================================================
# 4. HELPER FUNCTION
#    Check whether detection is probably the UGV itself
# ============================================================

def is_ugv_detection(x1, y1, x2, y2, width, height):

    box_width = x2 - x1
    box_height = y2 - y1

    center_x = (x1 + x2) / 2
    bottom_y = y2

    # Normalize values
    center_ratio = center_x / width
    bottom_ratio = bottom_y / height
    box_width_ratio = box_width / width
    box_height_ratio = box_height / height

    # The SKD-1 usually appears near the lower-center
    # of this external camera view.
    near_center = 0.30 < center_ratio < 0.70
    near_bottom = bottom_ratio > 0.45
    reasonably_large = box_width_ratio > 0.12
    reasonably_tall = box_height_ratio > 0.12

    if near_center and near_bottom and reasonably_large and reasonably_tall:
        return True

    return False


# ============================================================
# 5. PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    frame = cv2.resize(frame, (1280, 720))

    height, width = frame.shape[:2]


    # ========================================================
    # 6. TAKE LEFT CAMERA VIEW
    # ========================================================

    camera = frame[:, :int(width * 0.48)]

    cam_height, cam_width = camera.shape[:2]


    # ========================================================
    # 7. TERRAIN ROI
    # ========================================================

    roi_start = int(cam_height * 0.40)

    terrain = camera[roi_start:cam_height, :]


    # ========================================================
    # 8. HSV TERRAIN SEGMENTATION
    # ========================================================

    hsv = cv2.cvtColor(terrain, cv2.COLOR_BGR2HSV)


    # --------------------------------------------------------
    # Grass
    # --------------------------------------------------------

    grass_lower = np.array([35, 50, 40])
    grass_upper = np.array([90, 255, 220])

    grass_mask = cv2.inRange(
        hsv,
        grass_lower,
        grass_upper
    )


    # --------------------------------------------------------
    # Dirt / Soil
    # --------------------------------------------------------

    dirt_lower = np.array([8, 30, 40])
    dirt_upper = np.array([25, 180, 210])

    dirt_mask = cv2.inRange(
        hsv,
        dirt_lower,
        dirt_upper
    )


    # --------------------------------------------------------
    # Concrete
    # --------------------------------------------------------

    concrete_lower = np.array([0, 0, 50])
    concrete_upper = np.array([180, 70, 230])

    concrete_mask = cv2.inRange(
        hsv,
        concrete_lower,
        concrete_upper
    )


    # ========================================================
    # 9. COMBINE TERRAIN MASKS
    # ========================================================

    mask = cv2.bitwise_or(
        grass_mask,
        dirt_mask
    )


    # ========================================================
    # 10. REMOVE NOISE
    # ========================================================

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


    # ========================================================
    # 11. FULL MASK
    # ========================================================

    full_mask = np.zeros(
        (cam_height, cam_width),
        dtype=np.uint8
    )

    full_mask[
        roi_start:cam_height,
        :
    ] = mask


    # ========================================================
    # 13. ROI LINE
    # ========================================================

    # Lines are drawn onto a dedicated "base_display" canvas
    # (a copy of camera) instead of "camera" itself, so the
    # raw camera frame fed into YOLO later stays clean.

    base_display = camera.copy()

    cv2.line(
        base_display,
        (0, roi_start),
        (cam_width, roi_start),
        (255, 0, 0),
        2
    )


    # ========================================================
    # 14. DIVIDE TERRAIN INTO 3 NAVIGATION ZONES
    # ========================================================

    zone_width = cam_width // 3

    left_x1 = 0
    left_x2 = zone_width

    center_x1 = zone_width
    center_x2 = zone_width * 2

    right_x1 = zone_width * 2
    right_x2 = cam_width


    # ========================================================
    # 16. DRAW ZONE LINES
    # ========================================================

    cv2.line(
        base_display,
        (zone_width, roi_start),
        (zone_width, cam_height),
        (255, 255, 255),
        2
    )

    cv2.line(
        base_display,
        (zone_width * 2, roi_start),
        (zone_width * 2, cam_height),
        (255, 255, 255),
        2
    )


    # ========================================================
    # 17. YOLO DETECTION
    # ========================================================

    yolo_results = model(
        camera,
        verbose=False
    )


    obstacle_count = 0

    obstacle_info = []


    # ========================================================
    # 17A. IDENTIFY UGV SELF-DETECTION
    # ========================================================

    # Scan the same YOLO results used for obstacles to find
    # the box that is most likely the SKD-1 itself, so its
    # footprint can be excluded from the terrain mask below.

    ugv_box = None

    for detection in yolo_results[0].boxes:

        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()

        confidence = float(
            detection.conf[0].cpu().numpy()
        )

        class_id = int(
            detection.cls[0].cpu().numpy()
        )

        class_name = model.names[class_id]

        if confidence < 0.60:
            continue

        if class_name not in obstacle_classes:
            continue

        if is_ugv_detection(
            x1,
            y1,
            x2,
            y2,
            cam_width,
            cam_height
        ):

            ugv_box = (
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            )

            break


    # ========================================================
    # 17B. REMOVE UGV FROM MASK
    # ========================================================

    # Zero out the UGV's own bounding box in the terrain mask
    # so the vehicle's own body is never displayed as drivable
    # terrain in the overlay created next.

    if ugv_box is not None:

        ux1, uy1, ux2, uy2 = ugv_box

        ux1 = max(0, ux1)
        uy1 = max(0, uy1)
        ux2 = min(cam_width, ux2)
        uy2 = min(cam_height, uy2)

        full_mask[
            uy1:uy2,
            ux1:ux2
        ] = 0


    # ========================================================
    # 15. CALCULATE TERRAIN SCORE
    # ========================================================

    # Calculated after 17B so the UGV's own footprint is
    # already removed from full_mask and does not skew the
    # LEFT/CENTER/RIGHT terrain percentages.

    # Only count pixels inside the terrain ROI.
    roi_mask = full_mask[roi_start:, :]

    roi_height = roi_mask.shape[0]


    # LEFT
    left_mask = roi_mask[
        :,
        left_x1:left_x2
    ]

    left_score = (
        cv2.countNonZero(left_mask) /
        left_mask.size
    ) * 100


    # CENTER
    center_mask = roi_mask[
        :,
        center_x1:center_x2
    ]

    center_score = (
        cv2.countNonZero(center_mask) /
        center_mask.size
    ) * 100


    # RIGHT
    right_mask = roi_mask[
        :,
        right_x1:right_x2
    ]

    right_score = (
        cv2.countNonZero(right_mask) /
        right_mask.size
    ) * 100


    # ========================================================
    # 12. CREATE DISPLAY
    # ========================================================

    overlay = base_display.copy()

    overlay[full_mask > 0] = (0, 255, 0)

    result = cv2.addWeighted(
        base_display,
        0.65,
        overlay,
        0.35,
        0
    )


    # ========================================================
    # 18. PROCESS YOLO DETECTIONS
    # ========================================================

    for detection in yolo_results[0].boxes:

        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()

        confidence = float(
            detection.conf[0].cpu().numpy()
        )

        class_id = int(
            detection.cls[0].cpu().numpy()
        )

        class_name = model.names[class_id]


        # Ignore low confidence detections
        if confidence < 0.60:
            continue


        # Only consider possible obstacles
        if class_name not in obstacle_classes:
            continue


        # ====================================================
        # IGNORE THE UGV ITSELF
        # ====================================================

        if is_ugv_detection(
            x1,
            y1,
            x2,
            y2,
            cam_width,
            cam_height
        ):

            continue


        # ====================================================
        # REAL OBSTACLE
        # ====================================================

        obstacle_count += 1

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2


        # ----------------------------------------------------
        # Determine zone
        # ----------------------------------------------------

        if center_x < zone_width:

            zone = "LEFT"

        elif center_x < zone_width * 2:

            zone = "CENTER"

        else:

            zone = "RIGHT"


        # ====================================================
        # OBSTACLE PENALTY
        # ====================================================

        # Stronger penalty for objects closer to the bottom
        # because they are potentially closer to the UGV.

        proximity = center_y / cam_height

        penalty = confidence * proximity * 80


        if zone == "LEFT":

            left_score -= penalty

        elif zone == "CENTER":

            center_score -= penalty

        elif zone == "RIGHT":

            right_score -= penalty


        # Keep score inside 0-100
        left_score = max(0, min(100, left_score))
        center_score = max(0, min(100, center_score))
        right_score = max(0, min(100, right_score))


        # ====================================================
        # DRAW OBSTACLE BOX
        # ====================================================

        cv2.rectangle(
            result,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            3
        )


        label = f"{class_name} {confidence:.2f}"


        cv2.putText(
            result,
            label,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


        # Store obstacle information
        obstacle_info.append({
            "class": class_name,
            "confidence": round(confidence, 2),
            "zone": zone,
            "x": int(center_x),
            "y": int(center_y)
        })


    # ========================================================
    # 19. SELECT SAFEST DIRECTION
    # ========================================================

    scores = {
        "LEFT": left_score,
        "CENTER": center_score,
        "RIGHT": right_score
    }


    # --------------------------------------------------------
    # Prefer CENTER when scores are almost equal
    # --------------------------------------------------------

    best_direction = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_direction]

    center_difference = best_score - center_score


    if center_difference <= 5:

        recommended_direction = "CENTER"

    else:

        recommended_direction = best_direction


    # --------------------------------------------------------
    # Calculate decision confidence
    # --------------------------------------------------------

    sorted_scores = sorted(
        scores.values(),
        reverse=True
    )

    best = sorted_scores[0]

    second_best = sorted_scores[1]

    difference = best - second_best


    decision_confidence = min(
        100,
        50 + difference * 5
    )


    # ========================================================
    # ========================================================
    # 20. DISPLAY RECOMMENDATION
    # ========================================================

    cv2.putText(
        result,
        f"Recommended: {recommended_direction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        result,
        f"Decision Confidence: {decision_confidence:.1f}%",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    cv2.putText(
        result,
        f"Obstacles: {obstacle_count}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )    


    # ========================================================
    # 21. DISPLAY ZONE SCORES
    # ========================================================

    cv2.putText(
        result,
        f"LEFT: {left_score:.1f}%",
        (15, roi_start + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        result,
        f"CENTER: {center_score:.1f}%",
        (zone_width + 15, roi_start + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        result,
        f"RIGHT: {right_score:.1f}%",
        (zone_width * 2 + 15, roi_start + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 22. DISPLAY
    # ========================================================

    frame[:, :cam_width] = result
    cv2.imshow(
        "UGV DS-1 Perception",
        result
    )


    # ========================================================
    # 23. QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 24. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("DS-1 perception analysis completed!")