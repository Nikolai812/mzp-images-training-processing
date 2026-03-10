import torch
import os
import json
from torchvision import transforms
from color_dataset import ColorDataset

class PyTorchQualityPredictor:
    def __init__(self, model_path):
        from PyTorchImageQualityTrainer import PyTorchImageQualityTrainer
        self.model = PyTorchImageQualityTrainer.ColorCNN()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.classes = ['R', 'G', 'B']

    def predict(self, input_dir='./RAW_INPUT'):
        dataset = ColorDataset(root_dir=input_dir, transform=self.transform, is_training=False)
        results = {}
        with torch.no_grad():
            for image, img_path in dataset:
                image = image.unsqueeze(0)
                outputs = self.model(image)
                _, predicted = torch.max(outputs, 1)
                main_color = self.classes[predicted[0]]
                results[os.path.basename(img_path)] = main_color
        return results

    def save_results(self, results, output_file):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")
