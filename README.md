## Dataset

The dataset for this project can be downloaded [from this link](https://www.kaggle.com/datasets/atharvadumbre/indian-sign-language-islrtc-referred/data?select=original_images).
# Sign Language to Text Project

📖 This project translates Indian Sign Language (ISL) gestures from a live webcam feed into text. It uses OpenCV, Mediapipe, and RandomForestClassifier to draw and recognize hand landmarks and predict the corresponding alphabet sign in real-time.

---

## Features

* **Real-Time Recognition:** Identifies sign language gestures directly from a webcam.
* **Landmark-Based:** Uses MediaPipe to detect 21 key hand landmarks, making it robust to different lighting conditions.
* **Machine Learning Core:** Employs a Random Forest Classifier to accurately distinguish between different signs.
* **Simple UI:** Overlays the predicted letter directly onto the video feed for easy reading.

---

## 🛠️ Tech Stack & Dependencies

This project is built with Python and relies on the following libraries:

* **OpenCV:** For capturing and processing the webcam feed.
* **MediaPipe:** For high-fidelity hand and landmark detection.
* **Scikit-learn:** For the Random Forest machine learning model.
* **NumPy:** For numerical operations and data handling.
---

## 🚀 How to Run the Project (Quick Start)

This project requires the pre-trained model and dataset files, which are too large for GitHub. You must download them manually.

### 1. Prerequisites
* Python 3.8+
* A webcam

### 2. Installation
1.  Clone this repository to your local machine:
    ```bash
    git clone [https://github.com/POO-NAM/Sign-language-to-text-conversion_project.git](https://github.com/POO-NAM/Sign-language-to-text-conversion_project.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd Sign-Language-To-Text-Project
    ```
3.  **Download the Model & Dataset:**
    * Download the `model.p` file from: (https://drive.google.com/file/d/1SrBLfsElQlMesMgybasX6dSB0OxYO8X4/view?usp=sharing)
    * Download the `dataset.pickle` file from: (https://drive.google.com/file/d/1Au0tMbXnHqEwXLST7Za6BVXgCnPmEqDe/view?usp=sharing)
    * Place both files inside the `Sign-Language-To-Text-Project` folder.

4.  **Install Dependencies:**
    Install all the required Python libraries
    
### 3. Run the Application
Once the model is in your folder and the libraries are installed, you can run the app:
```bash
python real_time_testing_of_model_using_local_webcam.py
