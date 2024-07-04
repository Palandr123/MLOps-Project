import os
import sys
from subprocess import run
from pathlib import Path
from zipfile import ZipFile

from hydra import compose, initialize
import pandas as pd
import numpy as np
import great_expectations as gx
from great_expectations.data_context import FileDataContext
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler, LabelEncoder
from category_encoders.one_hot import OneHotEncoder
import zenml

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

def read_datastore() -> tuple[pd.DataFrame, str]:
    """
    Read sample and return in dataframe format to ZenML pipeline

    Returns:
        pd.DataFrame: data sample
        str: version number of sample
    """
    # Initialize Hydra with config path (replace with your config file)
    initialize(config_path="../configs", version_base="1.1")
    cfg = compose(config_name="sample_data")
    version_num = cfg.sample_num

    sample_path = Path("data") / "samples" / "sample.csv"
    df = pd.read_csv(sample_path)
    return df, version_num

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data step in ZenML pipeline

    Returns:
        pd.DataFrame: transformed features
        pd.DataFrame: target feature
    """
    # Columns with lot of nans
    lot_nans = ['condition', 'cylinders', 'VIN', 'drive', 'size', 'type', 'paint_color', 'county']
    df = df.drop(lot_nans, axis=1)
    # Convert to datetime
    df['posting_date_datetime'] = pd.to_datetime(df['posting_date'])
    # drop unused features
    features_not_used = ['image_url', 'description', 'posting_date', 'id', 'url', 'region_url']
    df = df.drop(features_not_used, axis=1)
    # Impute with most frequent
    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    cols_most_frequent = ['manufacturer', 'model', 'fuel', 'title_status', 'transmission']

    df[cols_most_frequent] = imp_most_frequent.fit_transform(df[cols_most_frequent])
    # Impute median
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    cols_median = ['year']

    df[cols_median] = imp_median.fit_transform(df[cols_median])

    
    df['posting_date_datetime'] = df['posting_date_datetime'].apply(lambda x: np.nan if x is pd.NaT else x.timestamp())
    # Impute mean
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    cols_mean = ['odometer', 'lat', 'long', 'posting_date_datetime']

    df[cols_mean] = imp_mean.fit_transform(df[cols_mean])
    # Transform date to month, day to further transform
    df['posting_date_datetime'] = df['posting_date_datetime'].apply(lambda x: pd.Timestamp.fromtimestamp(x))
    df['posting_date_month'] = df['posting_date_datetime'].apply(lambda x: x.month)
    df['posting_date_day'] = df['posting_date_datetime'].apply(lambda x: x.day)
    df = df.drop(['posting_date_datetime'], axis=1)

    # Ordinal encoding
    ordinal_enc_title = OrdinalEncoder(categories=[['parts_only', 'missing', 'salvage', 'lien', 'rebuilt', 'clean']], handle_unknown='use_encoded_value', unknown_value=-1)
    df['title_status'] = ordinal_enc_title.fit_transform(df[['title_status']])

    # One-hot encoding
    ohe_cols = ['transmission', 'fuel']
    ohe = OneHotEncoder(cols=ohe_cols, use_cat_names=True)
    df = ohe.fit_transform(df)

    # Label encoding
    label_cols = ['state', 'manufacturer', 'region', 'model']
    label_enc = LabelEncoder()

    for feature in label_cols:
        df[feature] = label_enc.fit_transform(df[feature])

    # Periodic encode
    def generate_periodic_encoder(encode_type='sin', offset=0, period=12):
        if encode_type == 'sin':
            return lambda x: np.sin(2 * np.pi * (x + offset) / period)
        elif encode_type == 'cos':
            return lambda x: np.cos(2 * np.pi * (x + offset) / period)

    df['lat_sin'] = df['lat'].apply(generate_periodic_encoder('sin', 90, 180))
    df['lat_cos'] = df['lat'].apply(generate_periodic_encoder('cos', 90, 180))
    df['long_sin'] = df['long'].apply(generate_periodic_encoder('sin', 0, 180))
    df['long_cos'] = df['long'].apply(generate_periodic_encoder('cos', 0, 180))
    df['posting_date_month_sin'] = df['posting_date_month'].apply(generate_periodic_encoder('sin', 0, 12))
    df['posting_date_month_cos'] = df['posting_date_month'].apply(generate_periodic_encoder('cos', 0, 12))
    df['posting_date_day_sin'] = df['posting_date_day'].apply(generate_periodic_encoder('sin', 0, 31))
    df['posting_date_day_cos'] = df['posting_date_day'].apply(generate_periodic_encoder('cos', 0, 31))
    df = df.drop(['lat', 'long', 'posting_date_month', 'posting_date_day'], axis=1)

    df = df[df['price'] >= 1000]
    df = df[df['price'] <= 40000]

    min_max_scale_cols = ['region', 'year', 'model', 'title_status', 'state', 'manufacturer']
    scaler = MinMaxScaler()

    df[min_max_scale_cols] = scaler.fit_transform(df[min_max_scale_cols])

    std_scale_cols = ['odometer']
    scaler = StandardScaler()
    df[std_scale_cols] = scaler.fit_transform(df[std_scale_cols])

    y = df[['price']]
    X = df.drop(['price'], axis=1)
    
    return X, y

def validate_features(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate features for ZenML pipeline

    Returns:
        pd.DataFrame: validated features
        pd.DataFrame: validated target feature
    """
    context = gx.get_context()
    ds_x = context.sources.add_or_update_pandas(name = "transformed_data")
    da_x = ds_x.add_dataframe_asset(name = "pandas_dataframe")
    batch_request_x = da_x.build_batch_request(dataframe = X)

    # Create expectations suite
    context.add_or_update_expectation_suite('transformed_data_expectation')
    
    # Create validator for X
    validator_x = context.get_validator(
        batch_request=batch_request_x,
        expectation_suite_name='transformed_data_expectation',
    )

    min_max_scale_cols = ['region', 'year', 'model', 'title_status', 'state', 'manufacturer']
    # Assume all columns scaled with min max in 0-1 range
    for col in min_max_scale_cols:
        validator_x.expect_column_values_to_be_between(
            column=col,
            min_value=0,
            max_value=1,
        )

    ohe_cols = ['fuel_gas', 'fuel_diesel', 'fuel_other', 'fuel_electric',
                'fuel_hybrid', 'transmission_automatic', 'transmission_manual','transmission_other']
    # Assume all ohe-transformed cols are 0 or 1
    for col in ohe_cols:
        validator_x.expect_column_values_to_be_in_set(
            column=col,
            value_set=[0, 1]
        )
    
    periodic_cols = ['lat_sin', 'lat_cos', 'long_sin',
                    'long_cos', 'posting_date_month_sin', 'posting_date_month_cos',
                    'posting_date_day_sin', 'posting_date_day_cos']
    # Assume all periodic-transformed in range (-1; 1)
    for col in periodic_cols:
        validator_x.expect_column_values_to_be_between(
            column=col,
            min_value=-1,
            max_value=1,
        )
    
    # Assume odometer is not null
    validator_x.expect_column_values_to_not_be_null(
        column='odometer'
    )

    # Store expectation suite
    validator_x.save_expectation_suite(
        discard_failed_expectations = False
    )
    
    # Create checkpoint
    checkpoint_x = context.add_or_update_checkpoint(
        name="checkpoint_x",
        validator=validator_x,
    )
    
    # Run validation
    checkpoint_result_x = checkpoint_x.run()


    ds_y = context.sources.add_or_update_pandas(name = "transformed_target")
    da_x = ds_y.add_dataframe_asset(name = "pandas_dataframe")
    batch_request_y = da_x.build_batch_request(dataframe = y)
    # Create expectations suite
    context.add_or_update_expectation_suite('transformed_target_expectation')

    validator_y = context.get_validator(
        batch_request=batch_request_y,
        expectation_suite_name='transformed_target_expectation',
    )
    # Assume price between 1000 and 40000
    validator_y.expect_column_values_to_be_between(
        column='price',
        min_value=1000,
        max_value=40000,
    )
    
    # Store expectation suite
    validator_y.save_expectation_suite(
        discard_failed_expectations = False
    )
    
    # Create checkpoint
    checkpoint_y = context.add_or_update_checkpoint(
        name="checkpoint_y",
        validator=validator_y,
    )
    
    # Run validation
    checkpoint_result_y = checkpoint_y.run()

    if checkpoint_result_x.success and checkpoint_result_y.success:
        return X, y

def load_features(X: pd.DataFrame, y: pd.DataFrame, ver: str):
    zenml.save_artifact(data = X, name = "features", tags = [ver])
    zenml.save_artifact(data = y, name = "target", tags = [ver])



if __name__ == "__main__":
    sample_data()
    if not validate_initial_data():
        print('Data sample fails to pass data validation')
        sys.exit(1) # return non-zero code to simplify interactions with shell scripts
    else:
        print('Data sample validated successfully')