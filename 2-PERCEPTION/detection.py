import cv2
from ultralytics import YOLO

# ---------------------------------------
# 1. LOAD MODEL
# ---------------------------------------

model = YOLO("yolo11n.pt")

# ---------------------------------------
# 2. INPUT VIDEO
# ---------------------------------------

video_path = "1-DATA/SKD-1_Outdoor.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Video opened successfully!")


# ---------------------------------------
# 3. PROCESS VIDEO
# ---------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize
    frame = cv2.resize(frame, (960, 540))

    # ---------------------------------------
    # 4. YOLO DETECTION
    # ---------------------------------------

    results = model(frame, verbose=False)

    # Get annotated frame
    annotated = results[0].plot()

    # ---------------------------------------
    # 5. DISPLAY
    # ---------------------------------------

    cv2.imshow(
        "UGV Object Detection",
        annotated
    )

    # Press Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------------------
# 6. CLEANUP
# ---------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Detection completed!")