import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

# Define paths to your dataset and model
dataset_dir = 'C:/Users/zoobe/Documents/projects/skin detection/dataset'
model_path = 'melanoma_classifier.keras'

# Image data generators for loading and augmenting images
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of the data for validation
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'  # Set as training data
)

# Validation data generator
val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Set as validation data
)

# Print the detected class indices
print("Training class indices:", train_generator.class_indices)
print("Validation class indices:", val_generator.class_indices)

# Calculate the number of images
num_train_images = train_generator.samples
num_val_images = val_generator.samples
print(f"Number of training images: {num_train_images}")
print(f"Number of validation images: {num_val_images}")

# Ensure steps_per_epoch and validation_steps are calculated correctly
steps_per_epoch = num_train_images // train_generator.batch_size
validation_steps = num_val_images // val_generator.batch_size
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Function to build the model
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

# Load existing model or build a new one
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Loaded existing model.")
else:
    model = build_model()
    print("Built a new model.")

# Compile the model (again after loading)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with additional debugging to check for issues
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=10
    )
except Exception as e:
    print(f"An error occurred during training: {e}")

# Fine-tuning the model
for layer in model.layers[:100]:
    layer.trainable = True

# Recompile the model with a lower learning rate (again)
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Continue training the model with fine-tuning
try:
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=10
    )
except Exception as e:
    print(f"An error occurred during fine-tuning: {e}")

# Save the trained model in the new Keras format
model.save(model_path, save_format='keras')
print("Model saved.")
