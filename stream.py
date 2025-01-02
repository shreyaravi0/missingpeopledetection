import streamlit as st
import cv2
import face_recognition
import os
import numpy as np

# Define paths
missing_persons_folder = "/Users/shreya/Desktop/missingpeopledetection/missingpeople"
output_directory = "/Users/shreya/Desktop/missingpeopledetection/output"
video_path = "/Users/shreya/Desktop/missingpeopledetection/coach5.mp4"

# Streamlit app
st.title("Missing Person Detection in Metro Coaches")
st.write("Detect missing persons from a predefined video file using uploaded images.")

# Upload missing persons images
uploaded_images = st.file_uploader(
    "Upload Missing Persons Images (PNG, JPG, JPEG)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process when images are uploaded
if uploaded_images:
    # Load encodings for missing persons
    st.write("Loading missing persons' encodings...")
    missing_person_encodings = {}
    for image_file in uploaded_images:
        image = face_recognition.load_image_file(image_file)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            missing_person_encodings[image_file.name] = encodings[0]
            st.write(f"Loaded encoding for {image_file.name}")
        else:
            st.warning(f"No face detected in {image_file.name}")

    # Check if any encodings were loaded
    if not missing_person_encodings:
        st.error("No valid encodings found. Please upload images with detectable faces.")
    else:
        # Process the predefined video
        st.write("Processing the video...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Unable to open video file at {video_path}.")
        else:
            frame_number = 0
            detected_persons = set()

            # Create a placeholder for displaying frames
            frame_placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.write("End of video.")
                    break

                frame_number += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces in the frame
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
                        detected_persons.add(best_match)

                        # Draw bounding box and label on the frame
                        top, right, bottom, left = face_location
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, best_match, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # Save the frame
                        output_path = os.path.join(output_directory, f"{best_match}_frame_{frame_number}.jpg")
                        cv2.imwrite(output_path, frame)

                # Display the frame
                frame_placeholder.image(frame, channels="BGR")

            cap.release()
            st.success(f"Detected missing persons: {detected_persons}")
            st.write("Processing complete.")
