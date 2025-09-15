import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class PyTorchImageQualityTrainer:
    class ColorDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.classes = ['R', 'G', 'B']
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.images = []
            for cls_name in self.classes:
                cls_dir = os.path.join(self.root_dir, cls_name)
                for img_name in os.listdir(cls_dir):
                    self.images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path, label = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label

    class ColorCNN(nn.Module):
        def __init__(self):
            super(PyTorchImageQualityTrainer.ColorCNN, self).__init__()
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

    def __init__(self, root_dir='.'):
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

    def load_dataset(self):
        self.dataset = self.ColorDataset(root_dir=self.root_dir, transform=self.transform)
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
