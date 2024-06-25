import os
import sys
from subprocess import run
from pathlib import Path
from zipfile import ZipFile

from hydra import compose, initialize
import pandas as pd
import great_expectations as gx
from great_expectations.data_context import FileDataContext


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

def validate_initial_data() -> bool:
    """
    Validates a data sample against specified expectations
    
    Returns:
        bool: validation result
    """
    
    # Set up data sample access
    context = FileDataContext(project_root_dir='services')
    sample_source = context.sources.add_or_update_pandas('data_sample')
    sample_asset = sample_source.add_csv_asset(name='data_sample_asset', filepath_or_buffer='data/samples/sample.csv')
    batch_request = sample_asset.build_batch_request()
    batches = sample_asset.get_batch_list_from_batch_request(batch_request)
    
    # Create expectations suite
    context.add_or_update_expectation_suite('expectation_suite')
    
    # Create validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name='expectation_suite',
    )
    
    # Define expectations
    # Ex 1
    validator.expect_column_values_to_not_be_null(
        column='fuel',
        mostly=0.75,
    )
    # Ex 2
    validator.expect_column_values_to_be_in_set(
        column='fuel',
        value_set=['gas', 'diesel', 'hybrid', 'electric', 'other']
    )
    # Ex 3
    validator.expect_column_most_common_value_to_be_in_set(
        column='fuel',
        value_set=['gas'],
        ties_okay=True,
    )
    # Ex 4
    validator.expect_column_values_to_not_be_null(
        column='manufacturer',
        mostly=0.75,
    )
    # Ex 5
    validator.expect_column_unique_value_count_to_be_between(
        column='manufacturer',
        min_value=1,
        max_value=100,
    )
    # Ex 6
    validator.expect_column_value_lengths_to_be_between(
        column='manufacturer',
        min_value=1,
        max_value=60,
    )
    # Ex 7
    validator.expect_column_values_to_not_be_null(
        column='VIN',
        mostly=0.5,
    )
    # Ex 8
    validator.expect_column_values_to_match_regex(
        column='VIN',
        regex='^(?=.*[0-9])(?=.*[A-z])[0-9A-z-]{17}$',
        mostly=0.5,
    )
    # Ex 9
    validator.expect_column_values_to_be_null(
        column='county'
    )
    # Ex 10
    validator.expect_column_values_to_not_be_null(
        column='transmission',
        mostly=0.7,
    )
    # Ex 11
    validator.expect_column_values_to_be_in_set(
        column='transmission',
        value_set=['automatic', 'manual'],
        mostly=0.7,
    )
    # Ex 12
    validator.expect_column_most_common_value_to_be_in_set(
        column='transmission',
        value_set=['automatic'],
        ties_okay=True,
    )
    # Ex 13
    validator.expect_column_values_to_not_be_null(
        column='type',
        mostly=0.75,
    )
    # Ex 14
    validator.expect_column_values_to_be_in_set(
        column='type',
        value_set=['sedan', 'SUV', 'pickup',
                   'truck', 'coupe', 'hatchback',
                   'min-van', 'offroad', 'bus', 'van',
                   'counvertible', 'wagon', 'other'],
        mostly=0.75,
    )
    # Ex 15
    validator.expect_column_unique_value_count_to_be_between(
        column='type',
        min_value=7,
    )
    # Ex 16
    validator.expect_column_values_to_not_be_null(
        column='lat',
        mostly=0.9,
    )
    # Ex 17
    validator.expect_column_values_to_be_between(
        column='lat',
        min_value=-90,
        max_value=90,
    )
    # Ex 18
    validator.expect_column_values_to_not_be_null(
        column='long',
        mostly=0.9,
    )
    # Ex 19
    validator.expect_column_values_to_be_between(
        column='long',
        min_value=-180,
        max_value=180,
    )
    # Ex 20
    validator.expect_column_values_to_not_be_null(
        column='drive',
        mostly=0.6,
    )
    # Ex 21
    validator.expect_column_values_to_be_in_set(
        column='drive',
        value_set=['fwd', 'rwd', '4wd'],
        mostly=0.6,
    )
    
    # Store expectation suite
    validator.save_expectation_suite(
        discard_failed_expectations = False
    )
    
    # Create checkpoint
    checkpoint = context.add_or_update_checkpoint(
        name="checkpoint",
        validator=validator,
    )
    
    # Run validation
    checkpoint_result = checkpoint.run()
    return checkpoint_result.success


if __name__ == "__main__":
    sample_data()
    if not validate_initial_data():
        print('Data sample fails to pass data validation')
        sys.exit(1) # return non-zero code to simplify interactions with shell scripts
    else:
        print('Data sample validated successfully')