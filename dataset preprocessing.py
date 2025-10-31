# importing all the dependencies
import mediapipe as mp
import os
import cv2
import pickle

# Declaring variables to draw landmarks on the hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Loading the dataset
data = 'original_images'

dataset = [] # To store landmarks of all the images
labels = [] # To store corresponding alphabet of each image
data_aux = [] # To store landmarks of each image

for dir_ in os.listdir(data): #Iterating through each folder
    for path in os.listdir(os.path.join(data, dir_)): # Iterating through each image
        img = cv2.imread(os.path.join(data, dir_, path))   # finding image path and converting its color to bgr
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data_aux = []

        results = hands.process(img_rgb)  # Processing the image for landmarks
        if results.multi_hand_landmarks:  # If landmarks are detected i.e., hands are detected in the image
            for hand_landmarks in results.multi_hand_landmarks:     # We will iterate for each hand, therefore this loop will run twice atmost.
                for i in range(len(hand_landmarks.landmark)):    # Iterating through each landmark and storing its X- and Y-coordinates.
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            num1 = 21 # number of landmarks in one hand
            num2 = num1 * 2 # Total number of landmarks in both hands
            max_len = num2 * 2 # We will store two co-ordinates for each landmark (i.e., X and Y co-ordinates)

            # If number of landmarks detected in the image are less than then the total number of landmarks in both hands.
            if len(data_aux) < max_len:
                data_aux.extend([0.0] * (max_len - len(data_aux))) #Then fill value 0.0 for remaining landmarks
            elif max_len < len(data_aux): #else if the number exceeds
                data_aux = data_aux[:max_len]   # Only consider the values upto max_len

            dataset.append(data_aux)
            labels.append(dir_)

f = open('dataset.pickle', 'wb')
pickle.dump({'dataset' : dataset, 'labels' : labels}, f)  # Storing the dataset locally, for later use.
f.close()

