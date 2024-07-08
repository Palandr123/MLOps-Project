import unittest
from unittest.mock import patch

import pandas as pd

from src.data import load_features


X_data = {
    "region": [0.1, 0.2],
    "year": [0.5, 0.6],
    "model": [0.7, 0.8],
    "title_status": [0.9, 1.0],
    "state": [0.3, 0.4],
    "manufacturer": [0.5, 0.6],
    "odometer": [30000, 40000],
    "fuel_gas": [1, 0],
    "fuel_diesel": [0, 1],
    "transmission_automatic": [1, 0],
    "transmission_manual": [0, 1],
    "lat_sin": [0.5, 0.6],
    "lat_cos": [0.7, 0.8],
    "long_sin": [0.9, 1.0],
    "long_cos": [0.1, 0.2],
    "posting_date_month_sin": [0.3, 0.4],
    "posting_date_month_cos": [0.5, 0.6],
    "posting_date_day_sin": [0.7, 0.8],
    "posting_date_day_cos": [0.9, 1.0],
}
y_data = {"price": [15000, 25000]}
VERSION = "v1"


class TestLoadFeatures(unittest.TestCase):
    @patch("src.data.zenml.save_artifact")
    def test_load_features(self, mock_save_artifact):
        X = pd.DataFrame(X_data)
        y = pd.DataFrame(y_data)

        # Call the method
        load_features(X, y, VERSION)

        # Assertions
        self.assertTrue(mock_save_artifact.called)
        # Check that the save_artifact method was called with the correct parameters
        args_list = mock_save_artifact.call_args_list

        # Extract data arguments from mock calls
        call_X_data = args_list[0][1]["data"]
        call_y_data = args_list[1][1]["data"]

        # Compare the DataFrames
        pd.testing.assert_frame_equal(call_X_data, X)
        pd.testing.assert_frame_equal(call_y_data, y)

        # Check other arguments
        self.assertEqual(args_list[0][1]["name"], "features")
        self.assertEqual(args_list[0][1]["tags"], [VERSION])
        self.assertEqual(args_list[1][1]["name"], "target")
        self.assertEqual(args_list[1][1]["tags"], [VERSION])
