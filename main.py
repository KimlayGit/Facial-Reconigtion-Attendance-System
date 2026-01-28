# import face_recognition
# import cv2
# import numpy as np
# import csv
# import os
# from datetime import datetime

# # --- SETTINGS ---
# path = 'photos'
# WIDTH, HEIGHT = 1280, 720  # Set the desired camera window size

# # Initialize Camera
# video_capture = cv2.VideoCapture(0)
# # Request HD resolution from the webcam
# video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
# video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# # Create a resizable window
# cv2.namedWindow("Smart Attendance", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Smart Attendance", WIDTH, HEIGHT)

# # --- LOAD DATABASE ---
# known_face_encodings = []
# known_faces_names = []

# if not os.path.exists(path):
#     os.makedirs(path)
#     print(f"Created {path} folder. Please add photos (e.g., lay.jpg) and restart.")

# print("Loading student database...")
# for filename in os.listdir(path):
#     if filename.lower().endswith((".jpg", ".png", ".jpeg")):A
#         image = face_recognition.load_image_file(f"{path}/{filename}")
#         encs = face_recognition.face_encodings(image)
#         if len(encs) > 0:
#             known_face_encodings.append(encs[0])
#             known_faces_names.append(os.path.splitext(filename)[0])
#             print(f"Registered: {filename}")

# # List to track attendance for the current session
# students_not_present = known_faces_names.copy()

# # Prepare CSV for the day
# current_date = datetime.now().strftime("%Y-%m-%d")
# f = open(current_date + ".csv", "a+", newline="")
# lnwriter = csv.writer(f)

# face_locations = []
# face_names = []
# process_this_frame = True

# print("SYSTEM ACTIVE. Press 'q' on your keyboard to EXIT.")

# while True:
#     ret, frame = video_capture.read()
#     if not ret:
#         break

#     # Optional: Mirror the frame so it feels like a mirror
#     frame = cv2.flip(frame, 1)

#     # 1. AI FACE PROCESSING
#     # Resize to 1/4 size ONLY for processing to maintain high FPS
#     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#     rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#     if process_this_frame:
#         # Detect face locations and encodings
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         face_names = []
#         for face_encoding in face_encodings:
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
#             name = "Unknown"

#             # Use face distance to find the best match
#             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#             if len(face_distances) > 0:
#                 best_match_index = np.argmin(face_distances)
#                 if matches[best_match_index]:
#                     name = known_faces_names[best_match_index]
                    
#                     # Mark attendance if the student is recognized for the first time
#                     if name in students_not_present:
#                         students_not_present.remove(name)
#                         current_time = datetime.now().strftime("%H:%M:%S")
#                         lnwriter.writerow([name, current_time])
#                         f.flush() # Immediately save to CSV
#                         print(f"Attendance recorded: {name} at {current_time}")

#             face_names.append(name)

#     # Toggle processing to save CPU power
#     process_this_frame = not process_this_frame

#     # 2. DRAWING VISUALS
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         # Scale coordinates back up by 4x to match the 1280x720 frame
#         top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
        
#         color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
#         # Draw the rectangle around the face
#         cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
#         # Draw the name label below the face
#         cv2.rectangle(frame, (left, bottom - 40), (right, bottom), color, cv2.FILLED)
#         cv2.putText(frame, name.upper(), (left + 10, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

#     # 3. SCREEN OVERLAYS
#     cv2.putText(frame, f"PRESENT: {len(known_faces_names) - len(students_not_present)}", 
#                 (20, HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.putText(frame, "PRESS 'Q' TO EXIT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#     # Display the final image
#     cv2.imshow("Smart Attendance", frame)

#     # Standard exit key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # --- CLEANUP ---
# video_capture.release()
# cv2.destroyAllWindows()
# f.close()



import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# --- SETTINGS ---
path = 'photos'
WIDTH, HEIGHT = 1280, 720 

# --- USB CAMERA SELECTION ---
# Try 1 if 0 is your laptop's built-in camera. 
# If you have no other camera, 0 might work for the virtual driver.
CAMERA_INDEX = 1 

video_capture = cv2.VideoCapture(CAMERA_INDEX)

# Set high resolution - USB can handle this without the Wi-Fi lag!
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not video_capture.isOpened():
    print(f"Error: Could not find camera at index {CAMERA_INDEX}. Trying index 0...")
    video_capture = cv2.VideoCapture(0)

cv2.namedWindow("Attendance System - USB Phone", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Attendance System - USB Phone", WIDTH, HEIGHT)

# --- LOAD DATABASE ---
known_face_encodings = []
known_faces_names = []

if not os.path.exists(path):
    os.makedirs(path)

print("Loading database...")
for filename in os.listdir(path):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        image = face_recognition.load_image_file(f"{path}/{filename}")
        encs = face_recognition.face_encodings(image)
        if len(encs) > 0:
            known_face_encodings.append(encs[0])
            known_faces_names.append(os.path.splitext(filename)[0])

students_not_present = known_faces_names.copy()

# Prepare CSV
current_date = datetime.now().strftime("%Y-%m-%d")
f = open(current_date + ".csv", "a+", newline="")
lnwriter = csv.writer(f)

face_locations = []
face_names = []
process_this_frame = True

print("SYSTEM ACTIVE via USB. No more Wi-Fi lag!")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Optional: Mirror the frame
    frame = cv2.flip(frame, 1)

    # AI PROCESSING (Fast detection on 1/4 size)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]
                    
                    if name in students_not_present:
                        students_not_present.remove(name)
                        lnwriter.writerow([name, datetime.now().strftime("%H:%M:%S")])
                        f.flush()
            face_names.append(name)

    process_this_frame = not process_this_frame

    # DRAWING
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        cv2.rectangle(frame, (left, bottom - 40), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name.upper(), (left + 10, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(frame, f"PRESENT: {len(known_faces_names) - len(students_not_present)}", (20, HEIGHT - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Attendance System - USB Phone", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()