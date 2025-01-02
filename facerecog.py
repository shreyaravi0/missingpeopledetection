import cv2
import face_recognition
import os


missing_persons_folder = "/Users/shreya/metronew/missing/missingpeople"
output_directory = "/Users/shreya/metronew/missing/output"
video_path = "/Users/shreya/metronew/missing/coach5.mp4"  



os.makedirs(output_directory, exist_ok=True)


missing_person_encodings = {}
print("Loading missing persons' encodings...")
for file_name in os.listdir(missing_persons_folder):
    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        file_path = os.path.join(missing_persons_folder, file_name)
        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            missing_person_encodings[file_name] = encodings[0]
            print(f"Loaded encoding for {file_name}")
        else:
            print(f"Warning: No face detected in {file_name}")


if not missing_person_encodings:
    print("No valid encodings found. Exiting.")
    exit()


print("Processing video...")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    exit()

frame_number = 0
detected_persons = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    frame_number += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        best_match = None
        best_distance = float("inf")
        for person_name, person_encoding in missing_person_encodings.items():
            matches = face_recognition.compare_faces([person_encoding], face_encoding, tolerance=0.5)
            face_distance = face_recognition.face_distance([person_encoding], face_encoding)[0]

            # Update best match if distance is smaller
            if matches[0] and face_distance < best_distance:
                best_match = person_name
                best_distance = face_distance

        if best_match:
            print(f"Detected {best_match} in frame {frame_number} (Distance: {best_distance:.2f})")
            detected_persons.add(best_match)

       
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, best_match, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            
            output_path = os.path.join(output_directory, f"{best_match}_frame_{frame_number}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved frame {frame_number} for {best_match} to {output_path}")

   
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Detected missing persons: {detected_persons}")
print("Processing complete.")
