import cv2
import time
from ultralytics import YOLO

# -----------------------------
# Load trained YOLO model
# -----------------------------
model = YOLO("runs/detect/runs/tennis_ball-4/weights/best.pt")

# -----------------------------
# Connect to Phone Camera
# Replace with your phone's IP
# -----------------------------
cap = cv2.VideoCapture("http://192.168.1.2:8080/video")

if not cap.isOpened():
    print("Could not connect to phone camera.")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame")
        break

    # -----------------------------------
    # Flip Horizontally (Mirror Fix)
    # -----------------------------------
    frame = cv2.flip(frame, 1)

    # -----------------------------------
    # Run YOLO Detection
    # -----------------------------------
    results = model(frame, verbose=False)

    # Draw Bounding Boxes
    annotated = results[0].plot()

    # -----------------------------------
    # FPS Calculation
    # -----------------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # -----------------------------------
    # Detection Information
    # -----------------------------------
    boxes = results[0].boxes
    count = len(boxes)

    highest = 0
    if count > 0:
        highest = max(float(box.conf) for box in boxes)

    # -----------------------------------
    # Display Information
    # -----------------------------------
    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        annotated,
        f"Balls: {count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        annotated,
        f"Confidence: {highest * 100:.1f}%",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )

    # -----------------------------------
    # Show Output
    # -----------------------------------
    cv2.imshow("🎾 BallVision AI", annotated)

    # Press Q to Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()