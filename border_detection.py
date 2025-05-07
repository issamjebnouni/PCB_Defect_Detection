import cv2

VIDEO_SOURCE = "20250507_221551.mp4"
cap = cv2.VideoCapture(VIDEO_SOURCE)

# State
card_seen = False
absence_counter = 0
card_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    resized = cv2.resize(frame, (640, 640))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_card = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if 10000 < area < 20000 and len(contour) > 20:
            found_card = True
            cv2.drawContours(resized, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(resized, "Card (custom shape)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Logic for robust counting
    if found_card:
        card_seen = True
        absence_counter = 0
    elif card_seen:
        absence_counter += 1
        if absence_counter >= 20:
            card_count += 1
            print(f"[COUNT] Card #{card_count} passed.")
            card_seen = False
            absence_counter = 0

    # Overlay count
    cv2.putText(resized, f"Total Cards: {card_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Custom Card Detection", resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
