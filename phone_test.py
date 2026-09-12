import cv2

# Replace with the IP address shown in the IP Webcam app
cap = cv2.VideoCapture("http://192.168.1.2:8080/video")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive video.")
        break

    cv2.imshow("Phone Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()