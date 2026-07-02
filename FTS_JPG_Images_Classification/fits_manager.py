from astropy.io import fits
import numpy as np
import os


class FITSManager:
    """
    A utility class for creating FITS files with fixed RGB values.
    """

    @classmethod
    def standard_file_name(cls, rc: int, gc: int, bc: int,
                           width: int = 256, height: int = 256,
                           suffix: str = "") -> str:
        """
        Generate a standardized filename for a FITS image.

        Args:
            rc: Red channel value.
            gc: Green channel value.
            bc: Blue channel value.
            width: Image width (default: 256).
            height: Image height (default: 256).
            suffix: Optional suffix for the filename.

        Returns:
            str: Standardized filename.
        """
        if suffix:
            return f"{rc}_{gc}_{bc}_{width}x{height}_{suffix}.fits"
        return f"{rc}_{gc}_{bc}_{width}x{height}.fits"

    @classmethod
    def create_fits_file(cls, rc: int, gc: int, bc: int,
                         width: int = 256, height: int = 256,
                         output_dir: str = ".") -> str:
        """
        Creates a FITS file with fixed RGB values (stored as separate planes).

        Args:
            rc: Red channel value.
            gc: Green channel value.
            bc: Blue channel value.
            width: Image width (default: 256).
            height: Image height (default: 256).
            output_dir: Directory to save the FITS file.

        Returns:
            str: Path to the created FITS file.
        """
        r = np.full((height, width), rc, dtype=np.uint8)
        g = np.full((height, width), gc, dtype=np.uint8)
        b = np.full((height, width), bc, dtype=np.uint8)

        rgb_image = np.stack([r, g, b], axis=0)
        hdu = fits.PrimaryHDU(rgb_image)

        file_name = cls.standard_file_name(rc, gc, bc, width, height)
        file_path = os.path.join(output_dir, file_name)

        hdu.writeto(file_path, overwrite=True)
        print(f"FITS file created: {file_path}")

        return file_path

    @classmethod
    def create_fits_file_as_3d_cube(cls, rc: int, gc: int, bc: int,
                                    width: int = 256, height: int = 256,
                                    output_dir: str = ".") -> str:
        """
        Creates a FITS file as a 3D RGB cube.

        Args:
            rc: Red channel value.
            gc: Green channel value.
            bc: Blue channel value.
            width: Image width (default: 256).
            height: Image height (default: 256).
            output_dir: Directory to save the FITS file.

        Returns:
            str: Path to the created FITS file.
        """
        r = np.full((height, width), rc, dtype=np.float32)
        g = np.full((height, width), gc, dtype=np.float32)
        b = np.full((height, width), bc, dtype=np.float32)

        rgb_cube = np.stack([r, g, b], axis=0)
        hdu = fits.PrimaryHDU(rgb_cube)

        # Add cube-specific metadata
        hdu.header["NAXIS"] = 3
        hdu.header["NAXIS3"] = 3
        hdu.header["CTYPE3"] = "RGB"

        file_name = cls.standard_file_name(rc, gc, bc, width, height, suffix="3d")
        file_path = os.path.join(output_dir, file_name)

        hdu.writeto(file_path, overwrite=True)
        print(f"3D FITS cube created: {file_path}")

        return file_path
