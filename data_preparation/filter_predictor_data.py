import argparse
import pandas as pd
import os

from cybench.config import KEY_LOC, PATH_DATA_DIR


def filter_by_yield_label(
    yield_file: str,
    data_files: list[str],
    output_dir: str,
    input_dir: str,
) -> None:
    """
    Filters the data files based on the adm_ids present in the yield file and saves the filtered data.

    Args:
        yield_file (str): Path to the yield data file.
        data_files (list[str]): List of paths to the data files to be filtered.
        output_dir (str): The directory where the filtered data will be saved.
        input_dir (str): The root directory containing the crop data.

    Returns:
        None: The function saves the filtered data to the output directory.
    """
    # Create symlink for the yield file in the output directory
    yield_output_file = os.path.join(output_dir, os.path.relpath(yield_file, input_dir))
    os.makedirs(os.path.dirname(yield_output_file), exist_ok=True)

    if not os.path.exists(yield_output_file):
        os.symlink(yield_file, yield_output_file)

    # Read the yield data (contains adm_id and yield)
    yield_df = pd.read_csv(yield_file)

    # Extract the list of valid adm_ids from the yield file
    valid_adm_ids = yield_df["adm_id"].dropna().unique()

    # Filter the predictor data files
    for data_file in data_files:
        # Read each predictor data CSV
        data_df = pd.read_csv(data_file)

        # Report how many rows are being removed
        original_size = len(data_df)
        filtered_data = data_df[data_df[KEY_LOC].isin(valid_adm_ids)]
        removed_entries = original_size - len(filtered_data)

        # Construct the output file path
        output_file = os.path.join(output_dir, os.path.relpath(data_file, input_dir))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if removed_entries > 0:
            # Save the filtered data if changes were made
            filtered_data.to_csv(output_file, index=False)
            print(
                f"Removed {removed_entries} rows and saved filtered data to: {output_file}"
            )
        else:
            # Create a symlink for unmodified files
            if not os.path.exists(output_file):
                os.symlink(data_file, output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter predictor data by matching adm_ids with yield labels."
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=PATH_DATA_DIR,
        help=f"Root directory to search for crop data (default: {PATH_DATA_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the processed and filtered files.",
    )

    args = parser.parse_args()

    # List of crops to process
    crops = ["maize", "wheat"]

    # Traverse the directory and process each crop and its subdirectories (e.g., AT, SE)
    for crop in crops:
        crop_dir = os.path.join(args.directory, crop)

        if os.path.isdir(crop_dir):
            # Process each country (e.g., AT, SE, DE)
            for subdir in os.listdir(crop_dir):
                subdir_path = os.path.join(crop_dir, subdir)

                if os.path.isdir(subdir_path):
                    country_code = subdir

                    # Find the yield file and other predictor files in the subdirectory
                    yield_file = None
                    predictor_files = []

                    for file in os.listdir(subdir_path):
                        file_path = os.path.join(subdir_path, file)

                        if file.endswith(f"yield_{crop}_{country_code}.csv"):
                            yield_file = file_path
                        elif file.endswith(".csv") and any(
                            indicator in file
                            for indicator in [
                                "crop_calendar",
                                "fpar",
                                "meteo",
                                "ndvi",
                                "soil",
                                "soil_moisture",
                            ]
                        ):
                            predictor_files.append(file_path)

                    if yield_file:
                        print(f"Processing {crop} data in: {subdir_path}")
                        # Pass both the output directory and input directory to the filtering function
                        filter_by_yield_label(
                            yield_file, predictor_files, args.output_dir, args.directory
                        )
                    else:
                        print(
                            f"No yield file found for {crop} in {country_code} at: {subdir_path}"
                        )


if __name__ == "__main__":
    main()
