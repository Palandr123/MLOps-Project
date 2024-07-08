import pandas as pd
from unittest.mock import patch

import pytest

from src.data import sample_data


EXPECTED_SAMPLE_SIZE = len(pd.read_csv("data/samples/sample.csv"))


@patch("src.data.download_data")
def test_sample_data(mock_download_data):
    """
    Tests if `sample_data` downloads data (using the mock) and returns a sample DataFrame.

    Mocks the `download_data` function to prevent actual data download during testing.
    """
    # Mock download_data to avoid actual download
    mock_download_data.return_value = None

    # Call the function
    sample = sample_data()

    # Assert sample is a DataFrame and has the expected size
    assert isinstance(sample, pd.DataFrame)
    assert len(sample) == EXPECTED_SAMPLE_SIZE


@patch("src.data.download_data")
def test_sample_data_erro(mock_download_data):
    """
    Tests if `sample_data` raises a ValueError for invalid sample number or size.

    Mocks the `download_data` function and injects a sample config with invalid values.
    """
    # Mock download_data to avoid actual download
    mock_download_data.return_value = None

    # Patch Hydra initialization (assuming it's from Hydra)
    with patch("hydra.initialize") as mock_initialize:
        # Mock config values to trigger the error (replace with actual values)
        mock_config = {"sample_num": -1, "sample_size": 2}
        mock_initialize.return_value.config = mock_config

        with pytest.raises(ValueError):
            sample_data()
