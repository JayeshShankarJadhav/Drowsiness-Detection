import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from pygame import mixer
import time

# Initialize alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascade classifiers for face & eyes
face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
left_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Load trained model
model = load_model('models/drowsiness_model.keras')

# Labels for classification
labels = ['Closed', 'Open']

# Open webcam
cap = cv2.VideoCapture(0)

# Variables for drowsiness detection logic
score = 0
thickness = 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    left_eyes = left_eye_cascade.detectMultiScale(gray)
    right_eyes = right_eye_cascade.detectMultiScale(gray)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # Process right eye
    rpred = None
    for (x, y, w, h) in right_eyes:
        r_eye = gray[y:y + h, x:x + w]
        r_eye = cv2.resize(r_eye, (48, 48)) / 255.0
        r_eye = np.expand_dims(r_eye, axis=(0, -1))  # Reshape for CNN
        prediction = model.predict(r_eye)
        rpred = np.argmax(prediction)  # Get class index
        break  # Only process the first detected eye

    # Process left eye
    lpred = None
    for (x, y, w, h) in left_eyes:
        l_eye = gray[y:y + h, x:x + w]
        l_eye = cv2.resize(l_eye, (48, 48)) / 255.0
        l_eye = np.expand_dims(l_eye, axis=(0, -1))  # Reshape for CNN
        prediction = model.predict(l_eye)
        lpred = np.argmax(prediction)  # Get class index
        break  # Only process the first detected eye

    # Check if both eyes are closed
    if rpred == 0 and lpred == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Score should not go negative
    score = max(score, 0)

    # Display score
    cv2.putText(frame, f'Score: {score}', (100, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Trigger alarm if drowsiness is detected
    if score > 15:
        cv2.imwrite("drowsy_image.jpg", frame)
        try:
            sound.play()
        except:
            pass

        # Increase and decrease thickness for alerting effect
        thickness = min(thickness + 2, 16) if thickness < 16 else max(thickness - 2, 2)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thickness)

    # Show frame
    cv2.imshow('Drowsiness Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
