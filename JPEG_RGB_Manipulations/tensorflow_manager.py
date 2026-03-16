import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
#import numpy as np
from PIL import Image
from llm_runner import LLMRunner

class TensorFlowManager(LLMRunner):
    def __init__(self, config):
        self.root_dir = config['training_input']
        self.raw_input_dir = config['raw_input']
        self.models_dir = config['models']
        self.output_json = config['output_json']
        self.model = None
        self.classes = ['R', 'G', 'B']

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

    def define_data_generators(self):
        self.train_datagen = ImageDataGenerator(rescale=1./255)
        self.train_generator = self.train_datagen.flow_from_directory(
            self.root_dir,
            target_size=(256, 256),
            batch_size=32,
            class_mode='categorical',
            classes=self.classes
        )

    def train(self, epochs=10, model_path='model.h5'):
        if not self.model or not hasattr(self, 'train_generator'):
            raise ValueError("Model or data generator not initialized. Call define_and_compile_model() and define_data_generators() first.")

        self.model.fit(
            self.train_generator,
            epochs=epochs,
            steps_per_epoch=len(self.train_generator)
        )
        print("Training complete!")
        self.save(model_path)

    def save(self, model_filename):
        model_path = os.path.join(self.models_dir, model_filename)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_filename):
        model_path = os.path.join(self.models_dir, model_filename)
        self.model = models.load_model(model_path)
        print(f"Model loaded from {model_path}")

    def predict(self):
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")

        results = {}
        for img_name in os.listdir(self.raw_input_dir):
            if img_name.lower().endswith('.jpg'):
                img_path = os.path.join(self.raw_input_dir, img_name)
                img = Image.open(img_path)
                img = img.resize((256, 256))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Create batch axis
                img_array /= 255.0  # Normalize

                predictions = self.model.predict(img_array)
                predicted_class_idx = tf.argmax(predictions[0]).numpy()
                main_color = self.classes[predicted_class_idx]
                results[img_name] = main_color

        self.save_results(results)
        return results

    def save_results(self, results):
        with open(self.output_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {self.output_json}")
