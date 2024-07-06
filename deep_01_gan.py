import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load a pre-trained GAN model (e.g., DCGAN on MNIST dataset)
generator = tf.keras.models.load_model('generator_model.h5')

# Generate random noise
noise = np.random.normal(0, 1, (1, 100))

# Generate an image
generated_image = generator.predict(noise)

# Rescale the image from [-1, 1] to [0, 1]
generated_image = 0.5 * generated_image + 0.5

# Plot the generated image
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()
