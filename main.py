import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# ================= SETTINGS =================
path = 'photos'
WIDTH, HEIGHT = 640, 480   # Lower resolution = higher FPS
FRAME_RESIZE_SCALE = 0.25 # Smaller processing frame

# ================= CAMERA (LAPTOP ONLY) =================
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW reduces latency on Windows

# Reduce internal buffer (VERY important for lag)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not video_capture.isOpened():
    print("❌ Laptop webcam not found")
    exit()

cv2.namedWindow("Attendance System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Attendance System", WIDTH, HEIGHT)

# ================= LOAD FACE DATABASE =================
known_face_encodings = []
known_faces_names = []

if not os.path.exists(path):
    os.makedirs(path)

print("[INFO] Loading face database...")
for file in os.listdir(path):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        image = face_recognition.load_image_file(os.path.join(path, file))
        enc = face_recognition.face_encodings(image)
        if enc:
            known_face_encodings.append(enc[0])
            known_faces_names.append(os.path.splitext(file)[0])

students_not_present = known_faces_names.copy()

# ================= CSV =================
current_date = datetime.now().strftime("%Y-%m-%d")
csv_file = open(current_date + ".csv", "a+", newline="")
csv_writer = csv.writer(csv_file)

# ================= VARIABLES =================
face_locations = []
face_names = []
process_this_frame = True

print("✅ System Running (Optimized for FPS & Low Latency)")

# ================= MAIN LOOP =================
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Resize for faster face recognition
    small_frame = cv2.resize(
        frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE
    )
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        # Use HOG model for speed
        face_locations = face_recognition.face_locations(
            rgb_small_frame, model="hog"
        )
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding, tolerance=0.5
            )
            name = "Unknown"

            distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )

            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]

                    if name in students_not_present:
                        students_not_present.remove(name)
                        csv_writer.writerow(
                            [name, datetime.now().strftime("%H:%M:%S")]
                        )
                        csv_file.flush()

            face_names.append(name)

    process_this_frame = not process_this_frame

    # ================= DRAW =================
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= int(1 / FRAME_RESIZE_SCALE)
        right *= int(1 / FRAME_RESIZE_SCALE)
        bottom *= int(1 / FRAME_RESIZE_SCALE)
        left *= int(1 / FRAME_RESIZE_SCALE)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(
            frame,
            name.upper(),
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    present = len(known_faces_names) - len(students_not_present)
    cv2.putText(
        frame,
        f"PRESENT: {present}",
        (10, HEIGHT - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= CLEANUP =================
video_capture.release()
cv2.destroyAllWindows()
csv_file.close()
