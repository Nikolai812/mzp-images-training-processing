from fits_manager import FITSManager
from jpeg_manager import *
from jpeg_manager import JPEGManager


def print_hi(name: str) -> None:
    print(f"Hi, {name}")

if __name__ == "__main__":
    print_hi("PyCharm")
    # Uncomment to use:
    # create_fits_file_as_3d_cube(10, 50, 200)
    r, g, b = 50, 200, 30
    #JPEGManager.create_jpeg_rgb(r,g,b)
    #tpl = JPEGManager.calculate_average_rgb(JPEGManager.standard_file_name(r,g,b))

    #print(f"average rgbs is r={tpl[0]} g={tpl[1]} b={tpl[2]}")

    JPEGManager.generate_monocolour_files(100)
    print(f"Done!")



