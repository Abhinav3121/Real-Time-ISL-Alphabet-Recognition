import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response
import os

app = Flask(__name__)

MODEL_PATH = 'isl_model.h5'
model = load_model(MODEL_PATH)
LABELS = sorted([folder for folder in os.listdir('ISL_Data') if os.path.isdir(os.path.join('ISL_Data', folder))])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            prediction_text = "No Hands Detected"

            if results.multi_hand_landmarks:
                all_landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for landmark in hand_landmarks.landmark:
                        all_landmarks.extend([landmark.x, landmark.y])
                
                # --- NEW PADDING LOGIC FOR PREDICTION ---
                landmarks_for_model = []
                # If one hand is detected, pad with zeros
                if len(all_landmarks) == 42:
                    landmarks_for_model = np.array(all_landmarks + [0.0] * 42).reshape(1, -1)
                # If two hands are detected, use directly
                elif len(all_landmarks) == 84:
                    landmarks_for_model = np.array(all_landmarks).reshape(1, -1)
                
                # Only predict if we have valid data
                if len(landmarks_for_model) > 0:
                    prediction = model.predict(landmarks_for_model)
                    predicted_class = np.argmax(prediction)
                    predicted_label = LABELS[predicted_class]
                    prediction_text = f"Prediction: {predicted_label}"

            cv2.putText(frame, prediction_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)