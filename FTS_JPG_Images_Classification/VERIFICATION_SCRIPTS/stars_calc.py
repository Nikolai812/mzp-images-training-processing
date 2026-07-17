from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from dataclasses import dataclass
from configparser import ConfigParser
from pathlib import Path
import shutil

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class FTSConfig:
    fts_input: str
    fts_classified: list[str]
    fwhm: float
    low_threshold: float
    high_threshold: float
    dry_run: bool
    predicted_json: str
    calculated_json: str
    calculation_output: str


def read_config(filename: str = "fts_config.ini") -> FTSConfig:
    parser = ConfigParser()
    parser.read(filename)

    cfg = parser["DEFAULT"]

    return FTSConfig(
        fts_input=cfg.get("fts_input", "FTS").strip(),
        fts_classified=[
            s.strip()
            for s in cfg.get("fts_classified", "").split(",")
            if s.strip()
        ],
        fwhm=cfg.getfloat("fwhm", 3.0),
        low_threshold=cfg.getfloat("low_threshold", 5.0),
        high_threshold=cfg.getfloat("high_threshold", 10.0),
        dry_run=cfg.getboolean("dry_run", True),
        predicted_json=cfg.get("predicted_json", "predicted.json"),
        calculated_json=cfg.get("calculated_json", "calculatedjson"),
        calculation_output=cfg.get("calculation_output", "OUTPUTS/CALCULATION_FTS"),
    )

def read_json_file(json_file: Path) -> Any | None:
    """
    Read a JSON file and return its content as a Python object.

    Parameters:
        json_file (Path): Path to the JSON file.

    Returns:
        Any | None:
            Parsed JSON content, or None if the file cannot be read.
    """
    try:
        with json_file.open("r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError:
        logger.warning("JSON file not found: %s", json_file)

    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON format in file %s: %s", json_file, e)

    except OSError as e:
        logger.warning("Could not read JSON file %s: %s", json_file, e)



def get_input_directories(root_dir: str) -> list[Path]:
    """
    Return all first-level subdirectories of the configured FTS root.
    """
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(root)

    return sorted(
        directory
        for directory in root.iterdir()
        if directory.is_dir()
    )


def get_star_sources(
    fits_image_path: str,
    fwhm: float = 3.0,
    threshold_factor: float = 5.0,
):
    data = fits.getdata(fits_image_path)

    mean, median, std = sigma_clipped_stats(data)

    daofind = DAOStarFinder(
        fwhm=fwhm,
        threshold=threshold_factor * std,
    )

    return daofind(data - median)

def copy_classified_files(config: FTSConfig, non_zero_stars: list[str], zero_stars_only_th10: list[str],
                          zero_stars_th5: list[str], calc_dir: Path) -> None:
    """
    Copy classified FITS files into the configured output directories.

    Args:
        calc_dir: 
    """

    if len(config.fts_classified) < 3:
        raise ValueError(
            "fts_classified must contain exactly three directories."
        )

    destinations = [
        calc_dir / config.fts_classified[0],
        calc_dir / config.fts_classified[1],
        calc_dir / config.fts_classified[2],
    ]

    #
    # Create destination directories if they do not exist.
    #
    for dst in destinations:
        dst.mkdir(parents=True, exist_ok=True)

    classifications = [
        (non_zero_stars, destinations[0], "BRIGHT_STARS"),
        (zero_stars_only_th10, destinations[1], "DIM_STARS"),
        (zero_stars_th5, destinations[2], "NO_STARS"),
    ]

    for files, destination, label in classifications:
        for filename in files:
            src = Path(filename)
            dst = destination / src.name

            shutil.copy2(src, dst)

            print(
                f"[COPY] {label:<12} : "
                f"{src} -> {dst}"
            )


def process_fts_file(
    fts_file: Path,
    config: FTSConfig,
    zero_stars_th5: list[str],
    zero_stars_only_th10: list[str],
    non_zero_stars: list[str],
    calc_dict: dict[str, str],
) -> None:

    print()
    print("####################################")
    print(fts_file)

    sources_low = get_star_sources(
        fits_image_path=str(fts_file),
        fwhm=config.fwhm,
        threshold_factor=config.low_threshold,
    )

    sources_high = get_star_sources(
        fits_image_path=str(fts_file),
        fwhm=config.fwhm,
        threshold_factor=config.high_threshold,
    )

    num_stars_low = 0 if sources_low is None else len(sources_low)
    num_stars_high = 0 if sources_high is None else len(sources_high)

    print(
        f"Detected {num_stars_low} stars "
        f"for threshold={config.low_threshold}, "
        f"fwhm={config.fwhm}"
    )

    print(
        f"Detected {num_stars_high} stars "
        f"for threshold={config.high_threshold}, "
        f"fwhm={config.fwhm}"
    )

    star_categories = config.fts_classified

    if num_stars_high > 0:
        non_zero_stars.append(str(fts_file))
        calc_dict[str(fts_file)] = star_categories[0]

        df = sources_high.to_pandas()
        brightest = df.sort_values("flux", ascending=False)
        print("Top 10 brightest (high threshold):")
        print(brightest.head(10))

    elif num_stars_low > 0:
        zero_stars_only_th10.append(str(fts_file))
        calc_dict[str(fts_file)] = star_categories[1]

        df = sources_low.to_pandas()
        brightest = df.sort_values("flux", ascending=False)
        print("Top 10 brightest (low threshold):")
        print(brightest.head(10))

    else:
        zero_stars_th5.append(str(fts_file))
        calc_dict[str(fts_file)] = star_categories[2]

    print("########################################")


def main() -> None:
    import os
    from pathlib import Path

    ###### SETTING ROOT
    pwd = os.path.dirname(__file__)
    root_dir = Path(pwd).parent
    #
    print('pwd:', pwd)
    print('root_dir:', root_dir)
    ###### END OF SETTING ROOT

    config = read_config()
    predicted_json=  read_json_file(Path(root_dir) / config.predicted_json)

    zero_stars_th5 = []
    zero_stars_only_th10 = []
    non_zero_stars = []
    calc_dict = {}

    for fts_dir in get_input_directories(str(root_dir / config.fts_input)):
        print(f"\nProcessing {fts_dir}")

        for fts_file in sorted(fts_dir.glob("*.fts")):
            process_fts_file(
                fts_file=fts_file,
                config=config,
                zero_stars_th5=zero_stars_th5,
                zero_stars_only_th10=zero_stars_only_th10,
                non_zero_stars=non_zero_stars,
                calc_dict=calc_dict,
            )


    print()
    print("========== SUMMARY ==========")
    print("non zero stars:", non_zero_stars)
    print("zero stars only high threshold:", zero_stars_only_th10)
    print("zero stars low threshold:", zero_stars_th5)

    #### SAVING Calculated json
    calculated_json = (Path(root_dir) / config.calculated_json)
    print(f"#######  Saving calculated star categories to: {calculated_json}#################")
    with open(calculated_json, 'w') as f:
        json.dump(calc_dict, f, indent=4)

    if config.dry_run:
        print("\nDry run enabled: no files copied.")
    else:
        print("\nCopying classified FITS files...")
        copy_classified_files(config, non_zero_stars, zero_stars_only_th10, zero_stars_th5, root_dir)


if __name__ == "__main__":
    main()