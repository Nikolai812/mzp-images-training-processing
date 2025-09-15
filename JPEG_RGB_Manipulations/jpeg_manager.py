from PIL import Image
from typing import Tuple
import numpy as np
import os
import random


class JPEGManager:
    """
    A utility class for creating and analyzing JPEG images with fixed RGB values.
    """

    @classmethod
    def standard_file_name(cls, red: int, green: int, blue: int,
                           width: int = 256, height: int = 256) -> str:
        """
        Generate a standardized filename for an RGB image.

        Args:
            red: Red channel value.
            green: Green channel value.
            blue: Blue channel value.
            width: Image width (default: 256).
            height: Image height (default: 256).

        Returns:
            str: Standardized filename.
        """
        return f"rgb_{red}_{green}_{blue}_{width}x{height}.jpg"

    @classmethod
    def create_jpeg_rgb(cls, red: int, green: int, blue: int,
                        width: int = 256, height: int = 256,
                        output_dir: str = ".") -> str:
        """
        Creates a JPEG image with fixed RGB values.

        Args:
            red: Red channel value.
            green: Green channel value.
            blue: Blue channel value.
            width: Image width (default: 256).
            height: Image height (default: 256).
            output_dir: Directory to save the image.

        Returns:
            str: Path to the created JPEG file.
        """
        image_array = np.zeros((height, width, 3), dtype=np.uint8)
        image_array[:, :, 0] = red
        image_array[:, :, 1] = green
        image_array[:, :, 2] = blue
        image = Image.fromarray(image_array, "RGB")

        file_name = cls.standard_file_name(red, green, blue, width, height)
        file_path = os.path.join(output_dir, file_name)

        image.save(file_path, "JPEG")
        print(f"JPEG image created: {file_path}")

        return file_path

    @classmethod
    def calculate_average_rgb(cls, image_path: str) -> Tuple[float, float, float]:
        """
        Reads an RGB image file and calculates the average R, G, and B values.

        Args:
            image_path: Path to the RGB image file (e.g., JPEG, PNG).

        Returns:
            tuple: (average_red, average_green, average_blue) as floats.
        """
        image = Image.open(image_path)
        image_array = np.array(image)

        # Calculate the average for each channel
        average_r = float(np.mean(image_array[:, :, 0]))
        average_g = float(np.mean(image_array[:, :, 1]))
        average_b = float(np.mean(image_array[:, :, 2]))

        return average_r, average_g, average_b

    @classmethod
    def generate_monocolour_files(cls, number: int,
                                  width: int = 256, height: int = 256) -> None:
        """
                Generate 'number' monocolour JPEG files with random RGB values.
                Files are stored in folders R, G, B depending on the max channel(s).

                Args:
                    number: How many files to generate.
                    width: Image width (default: 256).
                    height: Image height (default: 256).
                """
        # Ensure directories exist
        for folder in ["R", "G", "B"]:
            os.makedirs(folder, exist_ok=True)

        for i in range(number):
            # Random RGB values
            r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            max_val = max(r, g, b)

            # Create JPEG file
            file_name = cls.standard_file_name(r, g, b)
            #temp_file_path = cls.create_jpeg_rgb(r, g, b, width, height)

            # Copy/move file to channel folders
            if r == max_val:
                cls.create_jpeg_rgb(r, g, b, width, height, output_dir="R")
            if g == max_val:
                cls.create_jpeg_rgb(r, g, b, width, height, output_dir="G")
            if b == max_val:
                cls.create_jpeg_rgb(r, g, b, width, height, output_dir="B")

            print(f"Generated RGB ({r}, {g}, {b}) → saved to corresponding folder(s)")

