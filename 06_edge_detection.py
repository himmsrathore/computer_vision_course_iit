import cv2
import numpy as np

# Load an image
image_path = 'example.jpg'
image = cv2.imread(image_path)

# Check if image loading was successful
if image is None:
    print(f'Failed to load image at path: {image_path}')
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform Canny edge detection
edges = cv2.Canny(gray, threshold1=100, threshold2=200)  # Adjust thresholds as needed

# Display the original image and the edges
cv2.imshow('Original Image', image)
cv2.imshow('Edge Detection', edges)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
