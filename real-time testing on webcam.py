import cv2
import mediapipe as mp
import numpy as np
import pickle

# Retrieving the saved model.
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Using local machine webcam to capture the hand-signs made.
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    # lists to store the hand landmarks
    data_aux = []
    X_ = []
    Y_ = []
    ret, frame = cap.read()

    H, W, _ = frame.shape

    # Converting the image color from bgr to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks: # if landmarks are present
        # Drawing landmarks on the captured image using mediapipe.
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Extracting those landmarks and storing those into lists
        for hand_landmarks in result.multi_hand_landmarks:  # We will iterate for each hand, therefore this loop will run twice atmost.
            for i in range(len(hand_landmarks.landmark)):  # Iterating through each landmark and storing its X- and Y-coordinates.
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                X_.append(x)
                Y_.append(y)

        num1 = 21  # number of landmarks in one hand
        num2 = num1 * 2  # Total number of landmarks in both hands
        max_len = num2 * 2  # We will store two co-ordinates for each landmark (i.e., X and Y co-ordinates)

        # If number of landmarks detected in the image are less than then the total number of landmarks in both hands.
        if len(data_aux) < max_len:
            data_aux.extend([0.0] * (max_len - len(data_aux)))  # Then fill value 0.0 for remaining landmarks
        elif max_len < len(data_aux):  # else if the number exceeds
            data_aux = data_aux[:max_len]  # Only consider the values upto max_len

        x1 = int(min(X_) * W) - 10
        y1 = int(min(Y_) * H) - 10

        x2 = int(max(X_) * W) - 10
        y2 = int(max(Y_) * H) - 10

        res = model.predict([np.asarray(data_aux)])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, res[0], (x1, y1 - 15), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
