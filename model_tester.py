import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("model.h5")

img = cv2.imread("tomorrow_logo_test.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("img not found")

img = cv2.resize(img, (28, 28))

img = img.astype("float32") / 255.0

img = np.expand_dims(img, axis=(0, -1))

plt.imshow(img.squeeze(), cmap='gray')
plt.title("this is the image you inputted. x this tab to see prediction")
plt.show()

prediction = model.predict(img)
print("Raw prediction:", prediction)
print("Predicted class:", np.argmax(prediction))