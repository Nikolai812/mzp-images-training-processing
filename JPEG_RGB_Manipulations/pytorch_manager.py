#import torch
import os
from PyTorchImageQualityTrainer import PyTorchImageQualityTrainer
from llm_runner import LLMRunner

class PyTorchManager(LLMRunner):
    def __init__(self, root_dir='.'):
        self.root_dir = root_dir
        self.trainer = PyTorchImageQualityTrainer(root_dir=self.root_dir)

    def train(self, epochs, model_path):
        models_dir = "./MODELS/"
        pytorch_subdir = 'PYTORCH'
        pytorch_models_path = os.path.join(models_dir, pytorch_subdir)
        os.makedirs(pytorch_models_path, exist_ok=True)

        self.trainer.load_dataset()
        self.trainer.initialize_model()
        self.trainer.train(epochs=epochs)

        model_filename = os.path.join(pytorch_models_path, model_path)
        self.trainer.save(model_filename)
        #torch.save(self.trainer.model.state_dict(), model_filename)
        print(f"PyTorch model saved to {model_filename}")

    def predict(self, model_path):
        models_dir = "./MODELS/"
        pytorch_subdir = 'PYTORCH'
        pytorch_models_path = os.path.join(models_dir, pytorch_subdir)

        model_filename = os.path.join(pytorch_models_path, model_path)
        self.trainer.model = PyTorchImageQualityTrainer.ColorCNN()
        self.trainer.model.load_state_dict(torch.load(model_filename))
        self.trainer.model.eval()
        print(f"PyTorch model loaded from {model_filename}")

        # Add prediction logic here
        # Example: predict_and_save(self.trainer.model, input_dir="./RAWINPUT", output_file="output_dict.json")
