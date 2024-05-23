import numpy as np
import cv2
from mss import mss
import serial

# Load the ship cascade classifier
shipCascade = cv2.CascadeClassifier("Resources/haarcascade_encroachment2.xml")

# Define the screen capture region
bounding_box = {'top': 10, 'left': 0, 'width': 1270, 'height': 723}

# Create an MSS object
sct = mss()

# Parameters for flicker reduction
min_consecutive_frames = 15  # Adjust this value to control flicker reduction
consecutive_frames_with_detection = 0
ship_coordinates = []

# Load the map image
map = cv2.imread('Resources/cochi_map.jpg')

# Size for the red dots (rectangles) on the map
dot_size = 20  # Adjust this value to make the dots bigger or smaller

arduino = serial.Serial('COM6',9600)


while True:
    # Capture the screen region
    sct_img = sct.grab(bounding_box)

    # Convert the captured image to a format that OpenCV can work with
    img = np.array(sct_img)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect ships in the grayscale image
    ships = shipCascade.detectMultiScale(imgGray, scaleFactor=1.5, minNeighbors=20 ,minSize=(50, 50))

    # Check if sh
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # n >= min_consecutive_frames:


        for (x, y, w, h) in ships:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            print(f"Top-left coordinates: ({x}, {y})")
            print(f"Bottom-right coordinates: ({x + w}, {y + h})")

            # Append the ship coordinates to the list
            cv2.putText(img, f'Suspected Encroachment Regions: {len(ships)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ship_coordinates.append(((x, y), (x + w, y + h)))
            # Increment the encroachment count

    if len(ships) > 10:
        arduino.write(b'1')  # Send '1' command to turn on the LED
    else:
        arduino.write(b'0')  # Send '0' command to turn off the LED
    # Create an empty mask with the same dimensions as the map image
    mask = np.zeros_like(map, dtype=np.uint8)
    shade_color = (0, 0, 225)  # Shade color (BGR format)

    # Iterate through ship coordinates and add them to the mask
    for pts in ship_coordinates:
        left, right = pts

        # Define the vertices of the region to shade as a list of points
        vertices = np.array([[(left[0], left[1]), (right[0], left[1]), (right[0], right[1]), (left[0], right[1])]],
                            dtype=np.int32)

        # Fill the defined region with the shade color
        cv2.fillPoly(mask, [vertices], shade_color)

    # Apply the mask to the map image to shade the specified regions
    shaded_image = cv2.bitwise_and(map, mask)

    alpha = 0.9  # Adjust the alpha value for blending
    beta = 1 - alpha
    blended_image = cv2.addWeighted(map, alpha, shaded_image, beta, 0)

    # Resize the map image to the desired dimensions for display
    resize_map = cv2.resize(blended_image, (1280, 800))

    # Display the blended image
    cv2.imshow('Blended Map', resize_map)

    # Draw red dots (rectangles) on the map image for each detected ship
    for (x, y, w, h) in ships:
        center_x = (x + w // 2)
        center_y = (y + h // 2)//2

        # Calculate the coordinates for the top-left and bottom-right corners of the rectangle
        rect_x1 = max(center_x - dot_size // 2, 0)
        rect_y1 = max(center_y - dot_size // 2, 0)
        rect_x2 = min(center_x + dot_size // 2, map.shape[1])
        rect_y2 = min(center_y + dot_size // 2, map.shape[0])


        # Draw a red rectangle on the map image
        cv2.rectangle(resize_map, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)  # -1 for filled rectangle

    # Display the map image with red dots (rectangles)
    cv2.imshow("New Ships Spotted", resize_map)
    cv2.imshow("Detection Window",img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
