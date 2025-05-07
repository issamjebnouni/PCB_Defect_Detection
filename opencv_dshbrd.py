import cv2
import torch
import time
from ultralytics import YOLO
import os

# ───── Constants ───────────────────────────────────────────────────────────────
COMPONENT_MODEL_PATH = "defect_detector.pt"             # Your YOLOv8 Nano model for defects
VIDEO_SOURCE         = "20250507_221551.mp4" # 0 for webcam or path to file
NO_CARD_FRAMES       = 20                    # frames of absence to confirm exit
MIN_AREA, MAX_AREA   = 10000, 20000          # contour area bounds for your card
MIN_POINTS           = 20                    # filter out noisy contours
# ────────────────────────────────────────────────────────────────────────────────

SAVE_DIR = "saved_frames"  # NEW: Directory to save frames
os.makedirs(SAVE_DIR, exist_ok=True)

# ───── Load Models ─────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = YOLO(COMPONENT_MODEL_PATH).to(device)
class_names = model.names
NUM_CLASSES = 6
print(f"Loaded defect model on {device.upper()}. Expecting {NUM_CLASSES} classes.")
# ────────────────────────────────────────────────────────────────────────────────

# ───── Video Setup ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video source {VIDEO_SOURCE}")
# ────────────────────────────────────────────────────────────────────────────────

# ───── State Variables ─────────────────────────────────────────────────────────
card_seen             = False
absence_counter       = 0
card_count            = 0
damaged_count         = 0
# For each card, we'll collect only the “full detections” (all classes)
# Each entry is a tuple (frame_index, damaged_bool)
full_detections       = []
frame_index           = 0
# ────────────────────────────────────────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    frame_index += 1
    # Preprocess for contour detection
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    resized = cv2.resize(frame, (640, 640))
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur    = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blur, 50, 150)

    # ── Contour-based card presence ────────────────────────────────────────────
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found_card  = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA and len(cnt) > MIN_POINTS:
            found_card = True
            # draw the contour
            cv2.drawContours(resized, [cnt], -1, (0, 255, 0), 2)
            break
    # ────────────────────────────────────────────────────────────────────────────

    # ── State transitions & defect detection ──────────────────────────────────
    if found_card:
        # Card has (re)appeared
        if not card_seen:
            # Starting new card
            full_detections = []
            card_seen = True
            absence_counter = 0
            print(f"\n[FRAME {frame_index}] Card appeared.")
        else:
            # still present, reset absence counter
            absence_counter = 0

        # Run defect model on this frame
        start = time.time()
        results = model(resized)
        dt = time.time() - start

        # parse results
        detected = set()
        damaged  = False
        for box in results[0].boxes.data.cpu().numpy():
            cls = int(box[5])
            label = class_names[cls]
            detected.add(cls)
            if label.lower().endswith("damaged"):
                damaged = True

        # if all classes present, save this detection
        if len(detected) == NUM_CLASSES:
            full_detections.append((frame_index, damaged))
            # annotate
            annotated = results[0].plot()
            cv2.imshow("Defect Detections", annotated)

            card_id_str = f"{card_count + 1:03d}"  # card_count is incremented after card ends
            fname = f"card_{card_id_str}_frame_{frame_index:04d}_{'damaged' if damaged else 'ok'}.jpg"
            save_path = os.path.join(SAVE_DIR, fname)
            cv2.imwrite(save_path, annotated)
        else:
            cv2.imshow("Defect Detections", resized)

        # optionally print per-frame status
        print(f" Frame {frame_index}: inference={dt:.3f}s, "
              f"classes={len(detected)}/{NUM_CLASSES}, "
              f"{'DAMAGED' if damaged else 'OK'}")
    else:
        # Card not found in this frame
        if card_seen:
            absence_counter += 1
            # Confirm exit after N frames
            if absence_counter >= NO_CARD_FRAMES:
                card_seen = False
                card_count += 1

                # Decide damage: if any full detection was damaged
                was_damaged = any(d for _, d in full_detections)
                if was_damaged:
                    damaged_count += 1

                print(f"\n[COUNT] Card #{card_count} passed. "
                      f"{'DAMAGED' if was_damaged else 'OK'} "
                      f"(from {len(full_detections)} full detections)\n")

        # Show normal video when no card
        cv2.imshow("Defect Detections", resized)
    # ────────────────────────────────────────────────────────────────────────────

    # Overlay counts
    cv2.putText(resized, f"Total: {card_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(resized, f"Damaged: {damaged_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
