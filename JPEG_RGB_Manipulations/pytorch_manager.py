import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from color_dataset import ColorDataset
import json
from PIL import Image

class PyTorchManager:
    class ColorCNN(nn.Module):
        def __init__(self):
            super(PyTorchManager.ColorCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(32 * 64 * 64, 128)
            self.fc2 = nn.Linear(128, 3)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 32 * 64 * 64)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    def __init__(self, root_dir='./TRAINING_INPUT'):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.classes = ['R', 'G', 'B']

    def load_dataset(self):
        self.dataset = ColorDataset(root_dir=self.root_dir, transform=self.transform, is_training=True)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)

    def initialize_model(self):
        self.model = self.ColorCNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epochs=10):
        if not self.dataloader or not self.model or not self.optimizer:
            raise ValueError("Dataset, model, or optimizer not initialized. Call load_dataset() and initialize_model() first.")

        for epoch in range(epochs):
            for i, (images, labels) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if i % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(self.dataloader)}], Loss: {loss.item():.4f}')
        print("Training complete!")

    def save(self, model_filename):
        torch.save(self.model.state_dict(), model_filename)
        print(f"Model saved to {model_filename}")

    def load_model(self, model_filename):
        self.model = self.ColorCNN()
        self.model.load_state_dict(torch.load(model_filename))
        self.model.eval()
        print(f"Model loaded from {model_filename}")

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
