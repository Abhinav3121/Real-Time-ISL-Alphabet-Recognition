import cv2
import os
import time

# --- Constants ---
DATA_PATH = "ISL_Data"
num_images_per_sign = 400
# Time interval between captures, in seconds
capture_interval = 0.1 

# --- Get Sign Name ---
sign_name = input("Enter the name of the sign you are collecting data for: ")

# --- Create Directory for the Sign ---
sign_path = os.path.join(DATA_PATH, sign_name)
os.makedirs(sign_path, exist_ok=True) # A simpler way to create the directory

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Data Collection Loop ---
print("\nGet ready... The camera will start capturing automatically in 5 seconds.")
# Countdown before starting
for i in range(5, 0, -1):
    print(i, end='...', flush=True)
    time.sleep(1)
print("\nStarting capture!")

img_counter = 0
last_capture_time = time.time() # Initialize the timer

while img_counter < num_images_per_sign:
    success, frame = cap.read()
    if not success:
        print("Error: Failed to capture frame.")
        break
        
    frame = cv2.flip(frame, 1)

    # --- Automatic Capture Logic ---
    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        img_name = os.path.join(sign_path, f"{sign_name}_{img_counter}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Image saved: {img_name}")
        img_counter += 1
        last_capture_time = current_time # Reset the timer

    # --- Display Information on Screen ---
    cv2.putText(frame, f"Sign: {sign_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Images collected: {img_counter}/{num_images_per_sign}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Automated Data Collection", frame)

    # Allow quitting with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting data collection.")
        break

# --- Cleanup ---
print("Data collection complete.")
cap.release()
cv2.destroyAllWindows()