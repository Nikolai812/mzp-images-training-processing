
import os
import json

import numpy as np
import tensorflow as tf

from PIL import Image
from astropy.io import fits
from tensorflow.keras import layers, models

from llm_runner import LLMRunner


class TensorFlowManager(LLMRunner):
    """
    TensorFlow CNN classifier supporting:
      - JPG/JPEG RGB images
      - FTS/FITS/FIT astronomical images

    Training directory:
        training_input/
            GOOD/
            FAINT/
            OUT_OF_FIELD/

    Prediction directory:
        raw_input/
            image1.jpg
            image2.fts
            image3.fits
    """

    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 32

    RGB_EXTENSIONS = (".jpg", ".jpeg")
    FITS_EXTENSIONS = (".fts", ".fits", ".fit")

    def __init__(self, config):
        super().__init__(config)

        self.data_type = self.runner_config.get(
            "data_type", "jpg"
        ).lower()

        if self.data_type in ("jpg", "jpeg", "rgb"):
            self.data_type = "jpg"
            self.supported_extensions = self.RGB_EXTENSIONS

        elif self.data_type in ("fts", "fits", "fit"):
            self.data_type = "fts"
            self.supported_extensions = self.FITS_EXTENSIONS

        else:
            raise ValueError(
                f"Unsupported data type: {self.data_type}. "
                "Use 'jpg' or 'fts'."
            )

        self.model = None
        self.train_generator = None

        print(
            f"Initialized TensorFlowManager for "
            f"{self.data_type.upper()} images"
        )

    # ---------------------------------------------------------
    # Image loading and preprocessing
    # ---------------------------------------------------------

    def load_image(self, image_path):
        """
        Load an image and return a float32 NumPy array
        with shape (256, 256, 3) and values in [0, 1].
        """

        extension = os.path.splitext(image_path)[1].lower()

        if extension in self.RGB_EXTENSIONS:
            return self._load_rgb(image_path)

        if extension in self.FITS_EXTENSIONS:
            return self._load_fits(image_path)

        raise ValueError(
            f"Unsupported image extension: {extension}"
        )

    def _load_rgb(self, image_path):
        """Load a JPG/JPEG image as RGB."""

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize(
                self.IMAGE_SIZE,
                Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32)

        array /= 255.0

        return array

    def _load_fits(self, image_path):
        """
        Load a FITS/FTS image, normalize it independently
        to [0, 1], convert grayscale to three channels,
        and resize to 256 x 256.
        """

        with fits.open(image_path, memmap=False) as hdul:
            image = hdul[0].data

        if image is None:
            raise ValueError(
                f"No image data found in {image_path}"
            )

        image = np.asarray(image, dtype=np.float32)
        image = np.squeeze(image)

        # This implementation follows the grayscale FITS
        # convention used by the existing PyTorch dataset.
        if image.ndim != 2:
            raise ValueError(
                f"Expected a 2D FITS image after squeezing, "
                f"but got shape {image.shape} in {image_path}"
            )

        # Handle invalid pixels safely.
        finite_mask = np.isfinite(image)

        if not finite_mask.any():
            raise ValueError(
                f"No finite pixel values found in {image_path}"
            )

        minimum = np.min(image[finite_mask])
        maximum = np.max(image[finite_mask])

        # Match the per-image min-max normalization used
        # by FtsDataset in the PyTorch implementation.
        normalized = np.zeros_like(image, dtype=np.float32)

        if maximum > minimum:
            normalized[finite_mask] = (
                image[finite_mask] - minimum
            ) / (maximum - minimum)

        # Replace invalid pixels with zero.
        # FITS grayscale: (H, W) -> (H, W, 3).
        normalized = np.stack(
            [normalized, normalized, normalized],
            axis=-1
        )

        # Resize using TensorFlow.
        normalized = tf.image.resize(
            normalized,
            self.IMAGE_SIZE,
            method="bilinear"
        )

        return normalized.numpy().astype(np.float32)

    # ---------------------------------------------------------
    # Dataset creation
    # ---------------------------------------------------------

    def _list_images(self, directory):
        """Return supported image paths in deterministic order."""

        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Directory does not exist: {directory}"
            )

        return sorted(
            os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, filename))
            and filename.lower().endswith(
                self.supported_extensions
            )
        )

    def _training_samples(self):
        """
        Collect (image_path, class_index) pairs from
        training_input/<class_name>/ directories.
        """

        samples = []

        for class_index, class_name in enumerate(self.classes):
            class_dir = os.path.join(
                self.root_dir, class_name
            )

            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Training directory for class "
                    f"'{class_name}' does not exist: {class_dir}"
                )

            image_paths = self._list_images(class_dir)

            if not image_paths:
                print(
                    f"Warning: no {self.data_type} images "
                    f"found in {class_dir}"
                )

            samples.extend(
                (path, class_index) for path in image_paths
            )

        if not samples:
            raise ValueError(
                f"No training images found in {self.root_dir}"
            )

        return samples

    def define_data_generators(self):
        """
        Create a tf.data training pipeline.

        Labels are integer class indices, matching the
        PyTorch CrossEntropyLoss convention.
        """

        samples = self._training_samples()

        def sample_generator():
            for image_path, label in samples:
                image = self.load_image(image_path)
                yield image, np.int32(label)

        dataset = tf.data.Dataset.from_generator(
            sample_generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(256, 256, 3),
                    dtype=tf.float32
                ),
                tf.TensorSpec(
                    shape=(),
                    dtype=tf.int32
                )
            )
        )

        dataset = dataset.shuffle(
            buffer_size=len(samples),
            reshuffle_each_iteration=True
        )

        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self.train_generator = dataset

        print(
            f"Training images: {len(samples)}; "
            f"classes: {self.classes}"
        )

    # ---------------------------------------------------------
    # Model definition
    # ---------------------------------------------------------

    def define_and_compile_model(self):
        """Define the CNN using the PyTorch model's structure."""

        number_of_classes = len(self.classes)

        if number_of_classes < 2:
            raise ValueError(
                "At least two training categories are required."
            )

        self.model = models.Sequential([
            layers.Input(shape=(256, 256, 3)),

            layers.Conv2D(
                16, (3, 3),
                padding="same",
                activation="relu"
            ),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(
                32, (3, 3),
                padding="same",
                activation="relu"
            ),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(number_of_classes, activation="softmax")
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def pre_train(self):
        """Prepare the dataset and initialize the model."""

        self.define_data_generators()
        self.define_and_compile_model()

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def train(self):
        epochs = int(self.runner_config.get("epochs", 10))

        if self.model is None or self.train_generator is None:
            raise ValueError(
                "Model or training dataset not initialized. "
                "Call pre_train() first."
            )

        self.model.fit(
            self.train_generator,
            epochs=epochs
        )

        print("Training complete!")
        self.save()

    # ---------------------------------------------------------
    # Saving and loading
    # ---------------------------------------------------------

    def save(self):
        model_filename = self.runner_config.get(
            "model_file", "model.keras"
        )

        model_path = os.path.join(
            self.models_dir, model_filename
        )

        os.makedirs(self.models_dir, exist_ok=True)

        self.model.save(model_path)

        print(f"Model saved to {model_path}")

    def load_model(self, model_filename):
        model_path = os.path.join(
            self.models_dir, model_filename
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model file does not exist: {model_path}"
            )

        self.model = models.load_model(model_path)

        print(f"Model loaded from {model_path}")

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict(self):
        """
        Classify every supported image in raw_input_dir.

        Returns:
            Dictionary mapping filename to predicted class.
        """

        if self.model is None:
            raise ValueError(
                "Model not loaded. Call load_model() first "
                "or initialize and train the model."
            )

        image_paths = self._list_images(self.raw_input_dir)

        results = {}

        for image_path in image_paths:
            try:
                image = self.load_image(image_path)

                # Add the batch dimension.
                batch = np.expand_dims(image, axis=0)

                probabilities = self.model.predict(
                    batch, verbose=0
                )

                predicted_index = int(
                    np.argmax(probabilities[0])
                )

                predicted_class = self.classes[predicted_index]

                results[os.path.basename(image_path)] = (
                    predicted_class
                )

            except Exception as exc:
                # Do not silently classify unreadable images.
                print(
                    f"Failed to process {image_path}: {exc}"
                )
                raise

        self.save_results(results)

        return results

    def save_results(self, results):
        output_dir = os.path.dirname(
            os.path.abspath(self.output_json)
        )
        os.makedirs(output_dir, exist_ok=True)

        with open(self.output_json, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

        print(f"Results saved to {self.output_json}")
