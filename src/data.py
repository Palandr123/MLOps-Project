import os
from subprocess import run
from pathlib import Path
from zipfile import ZipFile

from hydra import compose, initialize
import pandas as pd


def download_data(user_name: str, dataset_name: str, save_path: str | Path):
    """
    Downloads data from the given URL to the specified directory using Kaggle API token.

    Args:
        user_name (str): name of Kaggle user that loaded the target dataset
        dataset_name (str): name of the dataset to be downloaded
        data_dir (str | Path): Directory to save the downloaded data.
    """
    data_path = Path(save_path)
    data_path.parent.mkdir(exist_ok=True, parents=True)
    if not data_path.exists():
        run(
            [
                "poetry",
                "run",
                "kaggle",
                "datasets",
                "download",
                f"{user_name}/{dataset_name}",
            ],
            check=True,
        )
        with ZipFile(f"{dataset_name}.zip", 'r') as zip_file:
            # Assuming there's only one CSV file in the archive
            csv_file = zip_file.namelist()[0]
            zip_file.extract(csv_file, data_path.parent)
            (data_path.parent / csv_file).rename(data_path)
        os.remove(Path(f"{dataset_name}.zip"))

    else:
        print(f"Data already exists: {data_path}")


def sample_data() -> pd.DataFrame:
    """
    Reads the data file, sorts by posting_date, and samples a portion.

    Returns:
        pd.DataFrame: The sampled data as a pandas DataFrame.
    """
    # Initialize Hydra with config path (replace with your config file)
    initialize(config_path="../configs", version_base="1.1")
    cfg = compose(config_name="sample_data")

    # Download data if not present
    download_data(cfg.user_name, cfg.dataset_name, cfg.save_path)

    # Read and sort data
    data = pd.read_csv(cfg.save_path)
    data = data.sort_values(by="posting_date")

    # Sample data
    if not 0 <= (cfg.sample_num - 1) * int(cfg.sample_size * len(data)) <= len(data):
        raise ValueError(
            "Make sure the the sample number and size lie in the range of dataset rows"
        )
    sample = data.iloc[
        (cfg.sample_num - 1)
        * int(cfg.sample_size * len(data)) : cfg.sample_num
        * int(cfg.sample_size * len(data))
    ]

    # Save sample
    sample_path = Path("data") / "samples" / "sample.csv"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False)
    print(f"Sample data saved to: {Path('data') / 'samples' / 'sample.csv'}")

    # Return sampled data
    return sample


if __name__ == "__main__":
    sample_data()
