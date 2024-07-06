import cv2

# Load an image
image_path = 'noice.png'
image = cv2.imread(image_path)

# Check if image loading was successful
if image is None:
    print(f'Failed to load image at path: {image_path}')
    exit()

# Apply Gaussian blur
blurred = cv2.GaussianBlur(image, (15, 5), 0)  # Adjust kernel size (5, 5) as needed

# Display the original image and the blurred image
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
