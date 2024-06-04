# Detecting Melanoma Using Machine Vision

## Table of Contents
1. [Project Introduction](#project-introduction)
2. [Project Overview](#project-overview)
3. [Establish a Dataset](#establish-a-dataset)
4. [Machine Learning Models](#machine-learning-models)
   - [EfficientNetB0](#efficientnetb0)
5. [Data Analysis](#data-analysis)
6. [Data Preprocessing](#data-preprocessing)
   - [Image Data Augmentation](#image-data-augmentation)
7. [Methodology](#methodology)
   - [Model Building](#model-building)
   - [Training and Fine-Tuning](#training-and-fine-tuning)
8. [Model Evaluation](#model-evaluation)
9. [GUI Application](#gui-application)
10. [Final Thoughts](#final-thoughts)

## Project Introduction
This project was initiated to explore the application of machine vision in detecting melanoma, a serious form of skin cancer. Leveraging advancements in deep learning and computer vision, the project aims to develop an accurate and efficient model to assist in early detection of melanoma, potentially saving lives through timely diagnosis and treatment. The project took around 18 hours to train on my own computer with a dataset comprised of over 70,000 images.

## Project Overview
The core of this project involves training a convolutional neural network (CNN) using the EfficientNetB0 architecture, fine-tuning it for the specific task of classifying skin lesions as either benign or malignant. The trained model is then integrated into a user-friendly graphical user interface (GUI) to make it accessible for practical use.

## Establish a Dataset
The dataset used for this project comprises images of skin lesions categorized into benign and malignant classes. This dataset was sourced from a reliable medical repository and split into training and validation sets to evaluate model performance. The dataset was retrieved from the The International Skin Imaging Collaboration​ (ISIC) and can be accessed via [this link.](https://gallery.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=%5B%22benign_malignant%7Cbenign%22%5D)

## Machine Learning Models

### EfficientNetB0
EfficientNetB0 is a state-of-the-art CNN architecture known for its efficiency and accuracy. It serves as the base model in this project, with additional custom layers added for the specific classification task.

## Data Analysis
The dataset contains images of varying sizes and qualities. Exploratory data analysis revealed significant class imbalance, with fewer images of malignant lesions. This necessitated the use of data augmentation techniques to artificially increase the dataset size and diversity.

## Data Preprocessing

### Image Data Augmentation
Data augmentation techniques such as rotation, width and height shifts, shear transformations, zoom, and horizontal flips were applied to enhance the dataset. This approach helps in improving the generalization ability of the model by exposing it to a wide variety of image transformations.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
```

## Methodology
### Model Building
The EfficientNetB0 architecture was chosen for its balance of accuracy and computational efficiency. The model was extended with custom dense layers and dropout for regularization.
```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model
```
## Training and Fine-Tuning
The model was initially trained with the base layers frozen, followed by fine-tuning where more layers were unfrozen and trained with a lower learning rate for better performance.
```python
from tensorflow.keras.optimizers import Adam

model = build_model()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=10
)

# Fine-tuning
for layer in model.layers[:100]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=10
)
```
## GUI Application
A user-friendly GUI was developed to allow users to upload images and receive instant classification results. This application was built using Python's Tkinter library and integrates the trained model to provide real-time predictions.

``` import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

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

root = tk.Tk()
root.title("Skin Disease Classifier")
root.geometry("600x600")

panel = tk.Label(root)
panel.pack(pady=20)

upload_btn = tk.Button(root, text="Upload Image", command=classify_image)
upload_btn.pack(pady=10)

root.mainloop()
```

![image](https://github.com/zhossain7/detectingmelanoma/assets/100549035/5daf4582-0d3a-4292-aa61-58c2d49b59b6)

## Conclusion
This project highlights the potential of machine vision and deep learning in the medical field, specifically for detecting melanoma. The combination of a robust CNN model and a user-friendly GUI demonstrates a practical approach to assist healthcare providers in early diagnosis. Future work could involve expanding the dataset, improving model accuracy, and integrating additional features for a more comprehensive diagnostic tool.

