import cv2
import mediapipe as mp
import os
import csv
import numpy as np
from tqdm import tqdm

DATA_PATH = "ISL_Data"
OUTPUT_CSV_FILE = "landmarks.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

header = ['label']
# We create a header for 84 features (42 landmarks * 2 coordinates)
for i in range(42 * 2):
    header += [f'p{i}']

with open(OUTPUT_CSV_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    sign_folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]

    for sign in tqdm(sign_folders, desc="Processing signs"):
        sign_path = os.path.join(DATA_PATH, sign)
        image_files = [f for f in os.listdir(sign_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in image_files:
            image_path = os.path.join(sign_path, img_name)
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                all_landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        all_landmarks.extend([landmark.x, landmark.y])
                
                # --- NEW PADDING LOGIC ---
                # If only one hand is detected (42 features), pad with 42 zeros
                if len(all_landmarks) == 42:
                    padded_landmarks = all_landmarks + [0.0] * 42
                    row = [sign] + padded_landmarks
                    writer.writerow(row)
                # If two hands are detected (84 features), use them directly
                elif len(all_landmarks) == 84:
                    row = [sign] + all_landmarks
                    writer.writerow(row)

hands.close()
print(f"\nData processing complete. Landmarks saved to '{OUTPUT_CSV_FILE}'")