import cv2

# Read an image
image = cv2.imread('example.jpg')

# Display the image
cv2.imshow('Example Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
