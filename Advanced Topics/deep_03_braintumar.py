import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# Load the pre-trained U-Net model
model = tf.keras.models.load_model('unet_model.h5')

# Load and preprocess an example medical image
image = load_img('brain_mri.jpg', target_size=(256, 256))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)

# Perform segmentation
prediction = model.predict(image)

# Reshape and display the result
segmented_image = np.squeeze(prediction, axis=0)
plt.imshow(segmented_image, cmap='gray')
plt.show()
