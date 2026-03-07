import os
from tensorflow.keras import models
from TensorFlowImageQualityTrainer import TensorFlowImageQualityTrainer
from llm_runner import LLMRunner

class TensorFlowManager(LLMRunner):
    def __init__(self, root_dir='.'):
        self.root_dir = root_dir
        self.trainer = TensorFlowImageQualityTrainer(root_dir=self.root_dir)

    def train(self, epochs, model_path):
        self.trainer.define_data_generators()
        self.trainer.define_and_compile_model()
        self.trainer.train(epochs=epochs)
        self.trainer.model.save(model_path)
        print(f"TensorFlow model saved to {model_path}")

    def predict(self, model_path):
        self.trainer.model = models.load_model(model_path)
        print(f"TensorFlow model loaded from {model_path}")

        # Add prediction logic here (e.g., call a prediction function)

        # Add prediction logic here
