from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

from dataclasses import dataclass
from configparser import ConfigParser
from pathlib import Path
import shutil

import csv
import json
import logging
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
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
    unified_csv: str
    summary_json: str


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
        calculated_json=cfg.get("calculated_json", "calculated.json"),
        calculation_output=cfg.get("calculation_output", "OUTPUTS/CALCULATION_FTS"),
        unified_csv=cfg.get("unified_csv", "OUTPUTS/unified.csv"),
        summary_json=cfg.get("summary_json", "OUTPUTS/summary.json"),
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

# Copies the file from fts input to classified folders
# (expects that there should be  >= 3 classes)
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

# Precesses fts file with astropy to get stars and
# to classify it according to category classes
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
        non_zero_stars.append(fts_file.name)
        calc_dict[fts_file.name] = star_categories[0]

        df = sources_high.to_pandas()
        brightest = df.sort_values("flux", ascending=False)
        print("Top 10 brightest (high threshold):")
        print(brightest.head(10))

    elif num_stars_low > 0:
        zero_stars_only_th10.append(fts_file.name)
        calc_dict[fts_file.name] = star_categories[1]

        df = sources_low.to_pandas()
        brightest = df.sort_values("flux", ascending=False)
        print("Top 10 brightest (low threshold):")
        print(brightest.head(10))

    else:
        zero_stars_th5.append(fts_file.name)
        calc_dict[fts_file.name] = star_categories[2]

    print("########################################")



def look_up_keys(
    predicted_json: dict[str, Any],
    calculated_json: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """
    Compare the keys of two dictionaries.

    Parameters:
        predicted_json: The left dictionary.
        calculated_json: The right dictionary.

    Returns:
        A tuple containing:
            - both_keys: Keys present in both dictionaries, in the order
              they appear in predicted_json.
            - left_only_keys: Keys only present in predicted_json, in the
              order they appear in predicted_json.
            - right_only_keys: Keys only present in calculated_json, in the
              order they appear in calculated_json.
    """
    calculated_keys = set(calculated_json)

    both_keys = []
    left_only_keys = []

    for key in predicted_json:
        if key in calculated_keys:
            both_keys.append(key)
        else:
            left_only_keys.append(key)

    predicted_keys = set(predicted_json)

    right_only_keys = [
        key for key in calculated_json
        if key not in predicted_keys
    ]

    return both_keys, left_only_keys, right_only_keys



## Create unified json for comparison of predicted
## and calculates
def unify_predicted_and_calculated_json(
    predicted_json: Path,
    calculated_json: Path,
    unified_csv: Path,
):
    """
    Create a TSV file containing the comparison between predicted and
    calculated values.

    Columns:
        filename    predicted    calculated    mismatch

    Only keys present in both dictionaries are written.
    """

    print("going to read predicted and calculated jsons....")
    with predicted_json.open("r", encoding="utf-8") as file:
        predicted_json = json.load(file)

    with calculated_json.open("r", encoding="utf-8") as file:
        calculated_json = json.load(file)

    both_keys, predicted_only_keys, calculated_only_keys = \
        look_up_keys(predicted_json, calculated_json)

    #
    # Ensure the output directory exists.
    #
    print(f"going to write unified.csv to: \n  {unified_csv.name}")
    unified_csv.parent.mkdir(parents=True, exist_ok=True)

    with unified_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=",")

        #
        # Header
        #
        writer.writerow([
            "filename",
            "predicted",
            "calculated",
            "mismatch",
        ])

        #
        # Rows
        #
        for key in both_keys:
            predicted = predicted_json[key]
            calculated = calculated_json[key]
            mismatch = "True" if predicted != calculated else ""

            writer.writerow([
                key,
                predicted,
                calculated,
                mismatch
            ])

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

    # This cycle runs over subfolders. It is expected to have multiple subfolders with .fts files
    # within the input directory (each day the telescope output is inside a separate folder).
    for fts_dir in get_input_directories(str(root_dir / config.fts_input)):
        print(f"\nProcessing {fts_dir}")

        # This cycle runs over .fts files in folder. It classifies .fts file and adds it to the corresponding list
        # the calc_dict is updated as well on each fts file, adding the filename as a key and the category as the value
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

    ### writing unified .tsv for intersected keys

    unify_predicted_and_calculated_json(Path(root_dir) / config.predicted_json,
                                        Path(root_dir) / config.calculated_json,
                                        Path(root_dir) / config.unified_csv)


    if config.dry_run:
        print("\nDry run enabled: no files copied.")
    else:
        print("\nCopying classified FITS files...")
        copy_classified_files(config, non_zero_stars, zero_stars_only_th10, zero_stars_th5, root_dir)


if __name__ == "__main__":
    main()