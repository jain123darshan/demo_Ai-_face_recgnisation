from fastapi import FastAPI, File, UploadFile
import face_recognition
import numpy as np
import cv2
import os

app = FastAPI()

# 🔹 Load known faces from the dataset
dataset_path = r"C:\\Users\\Lenovo\\PycharmProjects\\Face_Recognition_API\\dataset"

known_face_encodings = []
known_face_names = []

# 🔹 Read images from the dataset
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if os.path.isdir(person_folder):  # Check if it's a folder
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            # Process only image files
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:  # If face found, add it to the dataset
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(person_name)

@app.post("/recognize/")
async def recognize_face(file: UploadFile = File(...)):
    # 🔹 Read and decode the uploaded image
    image_data = await file.read()
    image = np.asarray(bytearray(image_data), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # 🔹 Convert BGR to RGB (required for face_recognition)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 🔹 Detect faces in the image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    recognized_faces = []

    for face_encoding in face_encodings:
        # Compare detected face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = np.argmax(matches)
            name = known_face_names[match_index]

        recognized_faces.append(name)

    return {"recognized_faces": recognized_faces}
