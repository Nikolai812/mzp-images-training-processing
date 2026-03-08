#import torch
import os
#from PyTorchImageQualityTrainer import PyTorchImageQualityTrainer
from llm_runner import LLMRunner

class PyTorchManager(LLMRunner):
    def __init__(self, root_dir='.'):
        #from PyTorchImageQualityTrainer import PyTorchImageQualityTrainer
        self.root_dir = root_dir
        #self.trainer = PyTorchImageQualityTrainer(root_dir=self.root_dir)

    def train(self, epochs, model_path):
        from PyTorchImageQualityTrainer import PyTorchImageQualityTrainer
        self.trainer = PyTorchImageQualityTrainer(root_dir=self.root_dir)

        models_dir = "./MODELS/"
        pytorch_subdir = 'PYTORCH'
        pytorch_models_path = os.path.join(models_dir, pytorch_subdir)
        os.makedirs(pytorch_models_path, exist_ok=True)

        self.trainer.load_dataset()
        self.trainer.initialize_model()
        self.trainer.train(epochs=epochs)

        model_filename = os.path.join(pytorch_models_path, model_path)
        self.trainer.save(model_filename)
        print(f"PyTorch model saved to {model_filename}")

    def predict(self, model_path):
        # Path to your trained model
        models_dir = "./MODELS/"
        pytorch_subdir = 'PYTORCH'
        model_path = os.path.join(models_dir, pytorch_subdir, model_path)
        # model_path = "./MODELS/PYTORCH/model.pth"

        # Directory containing new images to classify
        input_dir = "./RAW_INPUT"
        image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]

        from PyTorchQualityPredictor import PyTorchQualityPredictor
        # Initialize predictor
        predictor = PyTorchQualityPredictor(model_path)

        # Predict main colors
        results = predictor.predict(image_paths)

        # Print results
        print("Classification results:")
        for filename, main_color in results.items():
            print(f"{filename}: {main_color}")

        # Save results to file
        predictor.save_results(results, "output_dict.json")


