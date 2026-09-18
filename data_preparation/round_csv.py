import argparse
import pandas as pd
import pathlib


def round_csv(file_path: pathlib.Path, output_dir: pathlib.Path, decimals: int = 3):
    """Rounds numerical columns in a CSV file while keeping categorical ones intact."""
    df = pd.read_csv(file_path)

    # Detect categorical columns (assume non-numeric columns remain unchanged)
    numeric_cols = df.select_dtypes(include=["float64", "float32"]).columns

    # Round numeric columns
    df[numeric_cols] = df[numeric_cols].round(decimals)

    # Replicate the subdirectory structure in the output directory
    relative_path = file_path.relative_to(
        file_path.parents[2]
    )  # Adjusting to start from the first parent
    new_file_path = output_dir / relative_path

    # Ensure the parent directory exists in the output directory
    new_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the rounded file in the output directory
    df.to_csv(new_file_path, index=False)
    print(f"Processed: {file_path} -> {new_file_path}")


def find_and_process_csvs(
    directory: str, indicators: list, output_dir: str, decimals: int
):
    """Finds CSV files matching the given indicator patterns (e.g., soil, ndvi) and processes them."""
    base_path = pathlib.Path(directory)
    output_path = pathlib.Path(output_dir)

    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    for indicator in indicators:
        # Define the pattern for each indicator
        patterns = [f"{indicator}_maize_*.csv", f"{indicator}_wheat_*.csv"]

        for pattern in patterns:
            csv_files = base_path.rglob(
                pattern
            )  # Recursive search with the dynamic pattern

            for csv_file in csv_files:
                round_csv(csv_file, output_path, decimals)


def main():
    parser = argparse.ArgumentParser(description="Round numerical values in CSV files.")
    parser.add_argument(
        "--directory", type=str, help="Directory to search for CSV files."
    )
    parser.add_argument(
        "--indicators",
        type=str,
        nargs="+",
        default=["crop_calendar", "fpar", "meteo", "ndvi", "soil", "soil_moisture"],
        help="List of indicators to match files (default: 'crop_calendar', 'fpar', 'meteo', 'ndvi', 'soil', 'soil_moisture').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the processed files.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Number of decimals to round to (default: 3).",
    )

    args = parser.parse_args()
    find_and_process_csvs(
        args.directory, args.indicators, args.output_dir, args.decimals
    )


if __name__ == "__main__":
    main()
