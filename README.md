# Face Recognition Mini Project

## 📌 Project Overview

This is a mini project built using Python that demonstrates basic face recognition capabilities. The system is designed to recognize two prominent historical figures: **Bhagat Singh** and **Chhatrapati Shivaji Maharaj**. It uses image data of these two individuals to train the face recognition model and identify them in real-time or static images.

## 🎯 Objectives

- Understand and implement basic face recognition using Python libraries.
- Train the model using image datasets of specific individuals.
- Perform face detection and recognition in real-time or from input images.

## 🛠️ Technologies Used

- **Python 3.x**
- **OpenCV** – for image processing and camera input.
- **face_recognition** – for encoding and recognizing faces.
- **dlib** – for face detection and facial feature extraction.
- **NumPy** – for handling arrays.

## 📁 Project Structure

face_recognition_project/ ├── bhagat_singh/ # Folder containing images of Bhagat Singh ├── shivaji_maharaj/ # Folder containing images of Chhatrapati Shivaji Maharaj ├── main.py # Main script to run the face recognition ├── encodings.pickle # Precomputed face encodings (if any) └── README.md # Project description and guide

markdown
Copy
Edit

## 🚀 How to Run the Project

1. **Clone the repository** or download the project files.
2. **Install required libraries** (if not already installed):

   ```bash
   pip install opencv-python face_recognition dlib numpy
Add images of Bhagat Singh and Shivaji Maharaj to their respective folders.

Run the script:

bash
Copy
Edit
python main.py
The system will activate your webcam (or process input images) and try to recognize the faces based on the trained data.

🧠 How It Works
Loads sample images of Bhagat Singh and Chhatrapati Shivaji Maharaj.

Encodes the facial features using face_recognition.

Compares the face encodings from live webcam input (or static images) with known encodings.

Displays the recognized name on the output frame.

📸 Sample Output
The output will display live video frames (or processed images) with bounding boxes around detected faces and labels such as:

Bhagat Singh

Shivaji Maharaj

Unknown (if the face is not recognized)

✅ Future Enhancements
Add support for more individuals and larger datasets.

Improve accuracy with more diverse image samples.

Integrate with a GUI for user-friendly interaction.

👨‍💻 Author
Darshan Kumar Jain
Final year B.Tech student | AI and Software Enthusiast







