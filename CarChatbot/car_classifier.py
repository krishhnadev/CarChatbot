import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog
import os


model = tf.keras.models.load_model("model/car_classification_model.h5")
print("✅ Model loaded!")


class_names = ['audi', 'bmw', 'mercedes', 'toyota', 'volkswagen']


Tk().withdraw()
img_path = filedialog.askopenfilename(
    title="Select a car image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if not img_path:
    print("❌ No image selected.")
    exit()

print("🖼 Selected image:", img_path)

# Prepare image for prediction
img = image.load_img(img_path, target_size=(224, 224))
plt.imshow(img)
plt.axis('off')
plt.title("Selected Image")
plt.show()

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict car brand
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions)
predicted_label = class_names[predicted_class_index]

print("🚗 Predicted Car Brand:", predicted_label)
