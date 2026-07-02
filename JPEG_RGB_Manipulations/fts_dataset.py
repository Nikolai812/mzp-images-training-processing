import os

import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits


class FtsDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
        is_training=True,
        training_classes=None
    ):
        """
        Dataset for FITS/FTS astronomical images.

        Parameters
        ----------
        root_dir : str
            Root directory.

        transform : callable, optional
            Optional transform. Normally not used for FITS.

        is_training : bool
            If True, expects subdirectories corresponding to the
            training classes.

        training_classes : list[str]
            Names of the training categories.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        if is_training:

            self.classes = training_classes
            self.class_to_idx = {
                cls_name: i
                for i, cls_name in enumerate(self.classes)
            }

            for cls_name in self.classes:

                cls_dir = os.path.join(root_dir, cls_name)

                for filename in os.listdir(cls_dir):

                    if filename.lower().endswith((".fts", ".fits", ".fit")):
                        self.image_paths.append(
                            os.path.join(cls_dir, filename)
                        )
                        self.labels.append(
                            self.class_to_idx[cls_name]
                        )

        else:

            for filename in os.listdir(root_dir):

                if filename.lower().endswith((".fts", ".fits", ".fit")):
                    self.image_paths.append(
                        os.path.join(root_dir, filename)
                    )
                    self.labels.append(-1)

    def __len__(self):
        return len(self.image_paths)

    def _load_fits(self, filename):

        with fits.open(filename, memmap=False) as hdul:

            image = hdul[0].data

        if image is None:
            raise ValueError(f"No image data in {filename}")

        image = np.asarray(image, dtype=np.float32)

        #
        # Remove singleton dimensions if present.
        #
        image = np.squeeze(image)

        #
        # Normalize to [0,1]
        #
        minimum = np.min(image)
        maximum = np.max(image)

        if maximum > minimum:
            image = (image - minimum) / (maximum - minimum)
        else:
            image = np.zeros_like(image, dtype=np.float32)

        #
        # Convert grayscale image to 3-channel image.
        #
        image = np.stack([image, image, image], axis=0)

        return torch.from_numpy(image)

    def __getitem__(self, idx):

        filename = self.image_paths[idx]

        image = self._load_fits(filename)

        if self.transform is not None:
            image = self.transform(image)

        if self.labels[idx] != -1:
            return image, self.labels[idx]

        return image, filename