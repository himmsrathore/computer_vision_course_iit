import cv2
import numpy as np

# Load an image
image_path = 'himmat_square.png'
image = cv2.imread(image_path)

# Check if image loading was successful
if image is None:
    print(f'Failed to load image at path: {image_path}')
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform Harris corner detection
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Dilate the corner points to enhance their visibility
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image
image[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark detected corners in red

# Display the image with marked corners
cv2.imshow('Harris Corner Detection', image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
