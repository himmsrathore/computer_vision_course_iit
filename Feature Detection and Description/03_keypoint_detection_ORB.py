import cv2

# Read an image
image = cv2.imread('example.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = orb.detectAndCompute(gray_image, None)

# Draw keypoints on the image
keypoint_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

# Display the image with keypoints
cv2.imshow('ORB Keypoints', keypoint_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
