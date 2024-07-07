import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from src.data import read_datastore


class TestReadDatastore(unittest.TestCase):
    @patch("src.data.initialize")
    @patch("src.data.compose")
    @patch("src.data.pd.read_csv")
    def test_read_datastore(self, mock_read_csv, mock_compose, mock_initialize):
        # Set up the mock config and return values
        mock_compose.return_value.sample_num = "1.0"
        mock_df = pd.DataFrame({"column1": [1, 2, 3], "column2": [4, 5, 6]})
        mock_read_csv.return_value = mock_df

        # Call the method
        df, version_num = read_datastore()

        # Assertions
        self.assertEqual(version_num, "1.0")
        pd.testing.assert_frame_equal(df, mock_df)
        mock_initialize.assert_called_with(config_path="../configs", version_base="1.1")
        mock_compose.assert_called_with(config_name="sample_data")
        mock_read_csv.assert_called_with(Path("data") / "samples" / "sample.csv")
