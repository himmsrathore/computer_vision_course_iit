import cv2
import numpy as np

# Load an image
image_path = 'himmat.png'
image = cv2.imread(image_path)

# Check if image loading was successful
if image is None:
    print(f'Failed to load image at path: {image_path}')
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Display the image with keypoints
cv2.imshow('Image with SIFT Keypoints', image_with_keypoints)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
