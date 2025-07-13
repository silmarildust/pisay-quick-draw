import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("model_final.h5")

img = cv2.imread("test/kalachuchi_test.png", cv2.IMREAD_GRAYSCALE) #change for testing
if img is None:
    raise FileNotFoundError("img not found")

img = cv2.resize(img, (28, 28))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=(0, -1))

plt.imshow(img.squeeze(), cmap='gray')
plt.title("this is the image you inputted. x this tab to see prediction")
plt.show()

prediction = model.predict(img)

class_names = ['gazebo', 'kalachuchi', 'siklab', 'tomorrow_logo']
predicted_index = np.argmax(prediction)
predicted_class = class_names[predicted_index]

print("Raw prediction:", prediction)
print("Predicted class:", predicted_class)
