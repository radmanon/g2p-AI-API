from djitellopy import tello
import cv2
import requests

PI_URL = "http://165.227.45.231:8000/detect_faces/"

drone = tello.Tello()
drone.connect()
print("Battery:", drone.get_battery())

# Start Video Stream
drone.streamon()

while True:
    # Get the frame from the drone
    frame = drone.get_frame_read().frame

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Encode frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)

    # Send frame to API for face detection
    response = requests.post(API_URL, files={"image": img_encoded.tobytes()})

    if response.status_code == 200:
        faces = response.json().get("faces", [])

        # Draw bounding boxes around detected faces
        for face in faces:
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Tello Face Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
drone.streamoff()
