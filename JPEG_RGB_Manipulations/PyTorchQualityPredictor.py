import os
import torch
from torchvision import transforms
from PIL import Image
import json

class PyTorchQualityPredictor:
    class ColorDataset:
        def __init__(self, image_paths, transform=None):
            self.image_paths = image_paths
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path

    def __init__(self, model_path):
        self.model = PyTorchImageQualityTrainer.ColorCNN()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.classes = ['R', 'G', 'B']

    def predict(self, image_paths):
        dataset = self.ColorDataset(image_paths, transform=self.transform)
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

# Example usage
if __name__ == "__main__":
    from PyTorchImageQualityTrainer import PyTorchImageQualityTrainer

    # Path to your trained model
    model_path = "./MODELS/PYTORCH/model.pth"

    # Directory containing new images to classify
    input_dir = "./RAW_INPUT"
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]

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
