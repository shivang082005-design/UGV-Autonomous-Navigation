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
# 4A. HEURISTIC HAZARD DETECTORS (Phase 2)
#
# Potential Rock / Potential Ditch candidates.
#
# IMPORTANT:
# These are NOT YOLO detections and NOT confirmed hazards.
# yolo11n.pt (COCO) has no rock/ditch/tree/pothole classes.
# This is a heuristic OpenCV shape/edge layer.
# ============================================================

def detect_potential_rock(
    terrain,
    roi_start,
    cam_width,
    zone_width,
    full_mask
):

    hazards = []

    gray = cv2.cvtColor(
        terrain,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        40,
        120
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    terrain_roi_mask = full_mask[
        roi_start:,
        :
    ]

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 400 or area > 12000:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        aspect_ratio = (
            w / float(h)
            if h > 0
            else 0
        )

        # Rocks are roughly blob-shaped,
        # not thin or elongated.
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        circularity = (
            4 * np.pi * area
        ) / (
            perimeter * perimeter
        )

        # Reject very irregular/noisy edge fragments.
        if circularity < 0.25:
            continue

        # A rock should sit inside a patch
        # that was NOT already classified as
        # grass/dirt terrain.
        patch = terrain_roi_mask[
            y:y + h,
            x:x + w
        ]

        if patch.size == 0:
            continue

        non_terrain_ratio = (
            1.0
            -
            (
                cv2.countNonZero(patch)
                / float(patch.size)
            )
        )

        if non_terrain_ratio < 0.5:
            continue

        confidence = round(
            min(
                0.85,
                0.30
                + circularity * 0.3
                + non_terrain_ratio * 0.3
            ),
            2
        )

        full_x = x
        full_y = y + roi_start

        center_x = full_x + w / 2
        center_y = full_y + h / 2

        if center_x < zone_width:
            zone = "LEFT"

        elif center_x < zone_width * 2:
            zone = "CENTER"

        else:
            zone = "RIGHT"

        hazards.append({
            "type": "Potential Rock",
            "confidence": confidence,
            "zone": zone,
            "x": int(center_x),
            "y": int(center_y),
            "width": int(w),
            "height": int(h)
        })

    return hazards


def detect_potential_ditch(
    terrain,
    roi_start,
    cam_width,
    zone_width
):

    hazards = []

    gray = cv2.cvtColor(
        terrain,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (7, 7),
        0
    )

    mean_brightness = np.mean(
        blurred
    )

    dark_threshold = max(
        20,
        mean_brightness * 0.55
    )

    _, dark_mask = cv2.threshold(
        blurred,
        dark_threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = np.ones(
        (9, 9),
        np.uint8
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < 800:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if h == 0:
            continue

        aspect_ratio = w / float(h)

        # Ditches are wide,
        # shallow, horizontal-ish depressions.
        if aspect_ratio < 1.5:
            continue

        fill_ratio = (
            area / float(w * h)
        )

        # Reject noisy/speckled dark regions.
        if fill_ratio < 0.35:
            continue

        confidence = round(
            min(
                0.80,
                0.30
                + fill_ratio * 0.4
                + min(aspect_ratio, 4) * 0.05
            ),
            2
        )

        full_x = x
        full_y = y + roi_start

        center_x = full_x + w / 2
        center_y = full_y + h / 2

        if center_x < zone_width:
            zone = "LEFT"

        elif center_x < zone_width * 2:
            zone = "CENTER"

        else:
            zone = "RIGHT"

        hazards.append({
            "type": "Potential Ditch",
            "confidence": confidence,
            "zone": zone,
            "x": int(center_x),
            "y": int(center_y),
            "width": int(w),
            "height": int(h)
        })

    return hazards


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

    frame = cv2.resize(
        frame,
        (1280, 720)
    )

    height, width = frame.shape[:2]


    # ========================================================
    # 6. TAKE LEFT CAMERA VIEW
    # ========================================================

    camera = frame[
        :,
        :int(width * 0.48)
    ]

    cam_height, cam_width = camera.shape[:2]


    # ========================================================
    # 7. TERRAIN ROI
    # ========================================================

    roi_start = int(
        cam_height * 0.40
    )

    terrain = camera[
        roi_start:cam_height,
        :
    ]


    # ========================================================
    # 8. HSV TERRAIN SEGMENTATION
    # ========================================================

    hsv = cv2.cvtColor(
        terrain,
        cv2.COLOR_BGR2HSV
    )


    # --------------------------------------------------------
    # Grass
    # --------------------------------------------------------

    grass_lower = np.array([
        35,
        50,
        40
    ])

    grass_upper = np.array([
        90,
        255,
        220
    ])

    grass_mask = cv2.inRange(
        hsv,
        grass_lower,
        grass_upper
    )


    # --------------------------------------------------------
    # Dirt / Soil
    # --------------------------------------------------------

    dirt_lower = np.array([
        8,
        30,
        40
    ])

    dirt_upper = np.array([
        25,
        180,
        210
    ])

    dirt_mask = cv2.inRange(
        hsv,
        dirt_lower,
        dirt_upper
    )


    # --------------------------------------------------------
    # Concrete
    # --------------------------------------------------------

    concrete_lower = np.array([
        0,
        0,
        50
    ])

    concrete_upper = np.array([
        180,
        70,
        230
    ])

    concrete_mask = cv2.inRange(
        hsv,
        concrete_lower,
        concrete_upper
    )


    # ========================================================
    # 9. COMBINE TERRAIN MASKS
    # ========================================================

    # Concrete is intentionally NOT included.
    mask = cv2.bitwise_or(
        grass_mask,
        dirt_mask
    )


    # ========================================================
    # 10. REMOVE NOISE
    # ========================================================

    kernel = np.ones(
        (7, 7),
        np.uint8
    )

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
    # 13. ROI LINE / BASE DISPLAY
    # ========================================================

    # Use a separate display canvas so that
    # YOLO receives the clean camera image.

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

    # Find the YOLO box that is most likely
    # to represent the UGV itself.

    ugv_box = None

    for detection in yolo_results[0].boxes:

        x1, y1, x2, y2 = (
            detection.xyxy[0]
            .cpu()
            .numpy()
        )

        confidence = float(
            detection.conf[0]
            .cpu()
            .numpy()
        )

        class_id = int(
            detection.cls[0]
            .cpu()
            .numpy()
        )

        class_name = model.names[
            class_id
        ]

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

    if ugv_box is not None:

        ux1, uy1, ux2, uy2 = ugv_box

        ux1 = max(
            0,
            ux1
        )

        uy1 = max(
            0,
            uy1
        )

        ux2 = min(
            cam_width,
            ux2
        )

        uy2 = min(
            cam_height,
            uy2
        )

        full_mask[
            uy1:uy2,
            ux1:ux2
        ] = 0


    # ========================================================
    # 15. CALCULATE TERRAIN SCORE
    # ========================================================

    # Calculated AFTER UGV removal.

    roi_mask = full_mask[
        roi_start:,
        :
    ]

    roi_height = roi_mask.shape[0]


    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    left_mask = roi_mask[
        :,
        left_x1:left_x2
    ]

    left_score = (
        cv2.countNonZero(
            left_mask
        )
        /
        left_mask.size
    ) * 100


    # --------------------------------------------------------
    # CENTER
    # --------------------------------------------------------

    center_mask = roi_mask[
        :,
        center_x1:center_x2
    ]

    center_score = (
        cv2.countNonZero(
            center_mask
        )
        /
        center_mask.size
    ) * 100


    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    right_mask = roi_mask[
        :,
        right_x1:right_x2
    ]

    right_score = (
        cv2.countNonZero(
            right_mask
        )
        /
        right_mask.size
    ) * 100


    # ========================================================
    # 17C. HAZARD DETECTION
    #     POTENTIAL ROCKS / DITCHES
    # ========================================================

    # Heuristic CV candidates only.
    # These are NOT confirmed AI detections.

    hazard_info = []

    potential_rocks = detect_potential_rock(
        terrain,
        roi_start,
        cam_width,
        zone_width,
        full_mask
    )

    potential_ditches = detect_potential_ditch(
        terrain,
        roi_start,
        cam_width,
        zone_width
    )

    hazard_info.extend(
        potential_rocks
    )

    hazard_info.extend(
        potential_ditches
    )


    # --------------------------------------------------------
    # Conservative hazard penalty per zone
    # --------------------------------------------------------

    hazard_penalty = {
        "LEFT": 0.0,
        "CENTER": 0.0,
        "RIGHT": 0.0
    }

    for hazard in hazard_info:

        zone = hazard["zone"]

        # Potential hazard = lighter penalty.
        hazard_penalty[zone] += (
            hazard["confidence"] * 20
        )


    # --------------------------------------------------------
    # Cap total hazard penalty
    # --------------------------------------------------------

    hazard_penalty["LEFT"] = min(
        hazard_penalty["LEFT"],
        40
    )

    hazard_penalty["CENTER"] = min(
        hazard_penalty["CENTER"],
        40
    )

    hazard_penalty["RIGHT"] = min(
        hazard_penalty["RIGHT"],
        40
    )


    # --------------------------------------------------------
    # Apply hazard penalty
    # --------------------------------------------------------

    left_score -= hazard_penalty[
        "LEFT"
    ]

    center_score -= hazard_penalty[
        "CENTER"
    ]

    right_score -= hazard_penalty[
        "RIGHT"
    ]


    # Keep scores between 0 and 100.

    left_score = max(
        0,
        min(100, left_score)
    )

    center_score = max(
        0,
        min(100, center_score)
    )

    right_score = max(
        0,
        min(100, right_score)
    )


    # ========================================================
    # 12. CREATE DISPLAY
    # ========================================================

    overlay = base_display.copy()

    overlay[
        full_mask > 0
    ] = (0, 255, 0)

    result = cv2.addWeighted(
        base_display,
        0.65,
        overlay,
        0.35,
        0
    )


    # ========================================================
    # 17D. DRAW POTENTIAL HAZARDS
    # ========================================================

    # ORANGE = heuristic hazard candidate
    # GREEN = traversable terrain
    # RED = YOLO obstacle

    HAZARD_COLOR = (
        0,
        165,
        255
    )

    for hazard in hazard_info:

        hx = hazard["x"]
        hy = hazard["y"]

        hw = hazard["width"]
        hh = hazard["height"]

        top_left = (
            int(hx - hw / 2),
            int(hy - hh / 2)
        )

        bottom_right = (
            int(hx + hw / 2),
            int(hy + hh / 2)
        )

        cv2.rectangle(
            result,
            top_left,
            bottom_right,
            HAZARD_COLOR,
            2
        )

        hazard_label = (
            f"{hazard['type']} "
            f"{hazard['confidence']:.2f}"
        )

        cv2.putText(
            result,
            hazard_label,
            (
                top_left[0],
                top_left[1] - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            HAZARD_COLOR,
            2
        )


    # ========================================================
    # 18. PROCESS YOLO DETECTIONS
    # ========================================================

    for detection in yolo_results[0].boxes:

        x1, y1, x2, y2 = (
            detection.xyxy[0]
            .cpu()
            .numpy()
        )

        confidence = float(
            detection.conf[0]
            .cpu()
            .numpy()
        )

        class_id = int(
            detection.cls[0]
            .cpu()
            .numpy()
        )

        class_name = model.names[
            class_id
        ]


        # ----------------------------------------------------
        # Ignore low confidence detections
        # ----------------------------------------------------

        if confidence < 0.60:
            continue


        # ----------------------------------------------------
        # Only consider possible obstacles
        # ----------------------------------------------------

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

        center_x = (
            x1 + x2
        ) / 2

        center_y = (
            y1 + y2
        ) / 2


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

        # Objects closer to the bottom
        # receive a stronger penalty.

        proximity = (
            center_y / cam_height
        )

        penalty = (
            confidence
            * proximity
            * 80
        )


        if zone == "LEFT":

            left_score -= penalty

        elif zone == "CENTER":

            center_score -= penalty

        elif zone == "RIGHT":

            right_score -= penalty


        # Keep scores between 0 and 100.

        left_score = max(
            0,
            min(100, left_score)
        )

        center_score = max(
            0,
            min(100, center_score)
        )

        right_score = max(
            0,
            min(100, right_score)
        )


        # ====================================================
        # DRAW OBSTACLE BOX
        # ====================================================

        cv2.rectangle(
            result,
            (
                int(x1),
                int(y1)
            ),
            (
                int(x2),
                int(y2)
            ),
            (0, 0, 255),
            3
        )


        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            result,
            label,
            (
                int(x1),
                int(y1) - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


        # ----------------------------------------------------
        # Store obstacle information
        # ----------------------------------------------------

        obstacle_info.append({
            "class": class_name,
            "confidence": round(
                confidence,
                2
            ),
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
    # Find best direction
    # --------------------------------------------------------

    best_direction = max(
        scores,
        key=scores.get
    )

    best_score = scores[
        best_direction
    ]

    center_difference = (
        best_score
        -
        center_score
    )


    # --------------------------------------------------------
    # Prefer CENTER when scores are almost equal
    # --------------------------------------------------------

    if center_difference <= 5:

        recommended_direction = "CENTER"

    else:

        recommended_direction = (
            best_direction
        )


    # ========================================================
    # DECISION CONFIDENCE
    # ========================================================

    sorted_scores = sorted(
        scores.values(),
        reverse=True
    )

    best = sorted_scores[0]

    second_best = sorted_scores[1]

    difference = (
        best
        -
        second_best
    )

    decision_confidence = min(
        100,
        50 + difference * 5
    )


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
        f"Decision Confidence: "
        f"{decision_confidence:.1f}%",
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
        (
            15,
            roi_start + 50
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        result,
        f"CENTER: {center_score:.1f}%",
        (
            zone_width + 15,
            roi_start + 50
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        result,
        f"RIGHT: {right_score:.1f}%",
        (
            zone_width * 2 + 15,
            roi_start + 50
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 22. DISPLAY
    # ========================================================

    # Put processed camera back into
    # the left side of the original frame.

    frame[
        :,
        :cam_width
    ] = result

    cv2.imshow(
        "UGV DS-1 Perception",
        frame
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

print(
    "DS-1 perception analysis completed!"
)