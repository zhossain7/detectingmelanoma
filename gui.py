import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

# Load the trained model
model_path = 'locationhere/melanoma_classifier.keras'
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = preprocess_image(file_path)
        prediction = model.predict(img)[0][0]
        result = "Melanoma" if prediction > 0.8 else "Benign Mole"

        img_pil = Image.open(file_path)
        img_pil.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk

        messagebox.showinfo("Result", f"The image is classified as: {result}")

# GUI setup
root = tk.Tk()
root.title("Skin Disease Classifier")
root.geometry("600x600")

panel = tk.Label(root)
panel.pack(pady=20)

upload_btn = tk.Button(root, text="Upload Image", command=classify_image)
upload_btn.pack(pady=10)

root.mainloop()
