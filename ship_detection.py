import numpy as np
import cv2
from mss import mss
import multiprocessing
import time

# Initialize a multiprocessing Value to store the ship count
ship_count = multiprocessing.Value('i', 0)  # 'i' stands for integer

# Load the ship cascade classifier
shipCascade = cv2.CascadeClassifier("Resources\haarrcascade_ships4.xml")

# Define the screen capture region
bounding_box = {'top': 100, 'left': 100, 'width': 500, 'height': 1000}

# Create an MSS object
sct = mss()

# Parameters for flicker reduction
min_consecutive_frames = 15
consecutive_frames_with_detection = 0

# Function to update the ship count
def update_ship_count(ships_detected):
    global ship_count
    ship_count.value = ships_detected  # Update the shared ship_count

if __name__ == '__main__':
    try:
        while True:
            # Capture the screen region
            sct_img = sct.grab(bounding_box)

            # Convert the captured image to a format that OpenCV can work with
            img = np.array(sct_img)
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect ships in the grayscale image
            ships = shipCascade.detectMultiScale(imgGray, scaleFactor=1.05, minNeighbors=2)

            # Always update the ship count, even if no ships are detected
            update_ship_count(len(ships))  # Update the shared ship_count

            # Store the ship_count value in a file
            with open("ship_count.txt", "w") as file:
                file.write(str(ship_count.value))

            # Continuously print the ship count to the terminal
            print(f'Ship count: {ship_count.value}')

            # Check if ships are detected
            if len(ships) > 0:
                consecutive_frames_with_detection += 1

                # Draw rectangles around detected ships if detected in consecutive frames
                if consecutive_frames_with_detection >= min_consecutive_frames:
                    for (x, y, w, h) in ships:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Display the number of detected ships
                    cv2.putText(img, f'Detected Ships: {len(ships)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                consecutive_frames_with_detection = 0  # Reset consecutive frames count when no ships detected

            # Display the captured image with rectangles drawn around ships
            cv2.imshow("Boxed", img)

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {str(e)}")

    # Close all OpenCV windows
    cv2.destroyAllWindows()
