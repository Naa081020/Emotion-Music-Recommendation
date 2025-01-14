import os
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.applications import VGG16

# Define directories for training and validation data
train_dir = 'data/train'
val_dir = 'data/test'

# Enhanced Data augmentation for training and rescaling for both train and validation
train_datagen = ImageDataGenerator(

    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation for validation data

# Train and validation generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),  # Resize for model
    batch_size=64,
    color_mode="grayscale",  # Grayscale images
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),  # Resize for model
    batch_size=64,
    color_mode="grayscale",  # Grayscale images
    class_mode='categorical'
)

# Load the VGG16 model with pre-trained weights (transfer learning)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze the layers in the base model to avoid retraining
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the VGG16 base model
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(7, activation='softmax')(x)  # 7 classes for emotion classification

# Define the final model
emotion_model = Model(inputs=base_model.input, outputs=x)

# Compile the model
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

# Callbacks to adjust learning rate and stop early if validation loss doesn't improve
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=1e-7)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Train the model with the training data, validation data, and added callbacks
emotion_model_info = emotion_model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=75,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[lr_reduction, early_stopping]  # Use learning rate reduction and early stopping
)

# Save the trained model's weights
emotion_model.save_weights('emotion_model_vgg16.h5')

# Evaluate the model on validation data to check accuracy
val_loss, val_accuracy = emotion_model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")