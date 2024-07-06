import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine

# Load the pre-trained FaceNet model
model = load_model('facenet_keras.h5')

# Function to preprocess an image for FaceNet
def preprocess_image(image):
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    return np.expand_dims(image, axis=0)

# Function to calculate embedding
def get_embedding(model, face):
    face = preprocess_image(face)
    return model.predict(face)[0]

# Load and preprocess an example image
image = cv2.imread('face_image.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces using Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Assume we have another face to compare against (e.g., a stored embedding of a known person)
stored_embedding = np.load('stored_face_embedding.npy')

# Compare the detected face with the stored embedding
for (x, y, w, h) in faces:
    face = image[y:y+h, x:x+w]
    embedding = get_embedding(model, face)
    distance = cosine(stored_embedding, embedding)
    print(f'Distance: {distance}')
