import cv2

# Load Haar Cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Load video
cap = cv2.VideoCapture("face_video.mp4")

# Get video properties for saving
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_question2.mp4", fourcc, fps, (width, height))

# Blink control
no_eye_frames = 0
BLINK_THRESHOLD = 3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) > 0:
        # Take largest face
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])

        face_gray = gray[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        # Eye state logic
        if len(eyes) >= 2:
            no_eye_frames = 0
            color = (0, 255, 0)   # GREEN
            eye_status = "Eyes Open"
        else:
            no_eye_frames += 1
            if no_eye_frames >= BLINK_THRESHOLD:
                color = (0, 0, 255)   # RED
                eye_status = "Eye Blink"
            else:
                color = (0, 255, 0)
                eye_status = "Eyes Open"

        # Driver focus (simple logic)
        if no_eye_frames >= BLINK_THRESHOLD:
            driver_status = "Distracted"
        else:
            driver_status = "Focused"

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Display info
        cv2.putText(frame, eye_status, (x, y-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, "Status: " + driver_status, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    else:
        # No face detected
        cv2.putText(frame, "No Face Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show video
    cv2.imshow("Driver Monitoring System", frame)

    # Save video
    out.write(frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()