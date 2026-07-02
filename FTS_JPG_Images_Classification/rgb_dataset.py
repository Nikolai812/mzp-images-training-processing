import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class RgbDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True, training_classes=None):
        """
        Args:
            root_dir (str): Root directory containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_training (bool): If True, expects subdirectories R, G, B for training.
                                If False, treats all images in root_dir as raw input for prediction.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        if is_training:
            # For training: images are in R, G, B subdirectories
            self.classes = training_classes
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            for cls_name in self.classes:
                cls_dir = os.path.join(self.root_dir, cls_name)
                for img_name in os.listdir(cls_dir):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
        else:
            # For prediction: all images are in root_dir
            for img_name in os.listdir(self.root_dir):
                if img_name.lower().endswith('.jpg'):
                    self.image_paths.append(os.path.join(self.root_dir, img_name))
                    self.labels.append(-1)  # Dummy label for prediction

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.labels[idx] != -1:  # Return label only for training
            return image, self.labels[idx]
        else:
            return image, img_path  # Return image path for prediction
