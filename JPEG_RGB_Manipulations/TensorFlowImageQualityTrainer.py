import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TensorFlowImageQualityTrainer:
    def __init__(self, root_dir='.'):
        self.root_dir = root_dir
        self.train_datagen = None
        self.train_generator = None
        self.model = None

    def define_data_generators(self):
        self.train_datagen = ImageDataGenerator(rescale=1./255)
        self.train_generator = self.train_datagen.flow_from_directory(
            self.root_dir,
            target_size=(256, 256),
            batch_size=32,
            class_mode='categorical',
            classes=['R', 'G', 'B']
        )

    def define_and_compile_model(self):
        self.model = models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, epochs=10):
        if not self.train_generator or not self.model:
            raise ValueError("Data generator or model not initialized. Call define_data_generators() and define_and_compile_model() first.")

        self.model.fit(
            self.train_generator,
            epochs=epochs,
            steps_per_epoch=len(self.train_generator)
        )
        print("Training complete!")
