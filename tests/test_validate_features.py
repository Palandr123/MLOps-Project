import unittest
from unittest.mock import patch

import pandas as pd

from src.data import validate_features


X = pd.DataFrame(
    {
        "region": [0.1, 0.2],
        "year": [0.3, 0.4],
        "model": [0.5, 0.6],
        "title_status": [0.7, 0.8],
        "state": [0.9, 1.0],
        "manufacturer": [0.1, 0.2],
        "fuel_gas": [0, 1],
        "fuel_diesel": [1, 0],
        "fuel_other": [0, 0],
        "fuel_electric": [0, 0],
        "fuel_hybrid": [0, 0],
        "transmission_automatic": [1, 0],
        "transmission_manual": [0, 1],
        "transmission_other": [0, 0],
        "lat_sin": [0.5, -0.5],
        "lat_cos": [0.5, 0.5],
        "long_sin": [0, 1],
        "long_cos": [1, 0],
        "posting_date_month_sin": [0, 1],
        "posting_date_month_cos": [1, 0],
        "posting_date_day_sin": [0, 1],
        "posting_date_day_cos": [1, 0],
        "odometer": [10000, 20000],
    }
)
y = pd.DataFrame({"price": [15000, 25000]})


class TestValidateFeatures(unittest.TestCase):
    @patch("src.data.gx.get_context")
    def test_validate_features(self, MockGetContext):
        # Mock the context and other components
        mock_context = MockGetContext.return_value
        mock_checkpoint_x = mock_context.add_or_update_checkpoint.return_value
        mock_checkpoint_result_x = mock_checkpoint_x.run.return_value
        mock_checkpoint_result_x.success = True

        mock_checkpoint_y = mock_context.add_or_update_checkpoint.return_value
        mock_checkpoint_result_y = mock_checkpoint_y.run.return_value
        mock_checkpoint_result_y.success = True

        # Call the method
        validated_X, validated_y = validate_features(X, y)

        # Assertions
        self.assertTrue((validated_X == X).all().all())
        self.assertTrue((validated_y == y).all().all())
        mock_context.sources.add_or_update_pandas.assert_called()
        mock_context.add_or_update_checkpoint.assert_called()
        mock_checkpoint_x.run.assert_called()
        mock_checkpoint_y.run.assert_called()


if __name__ == "__main__":
    unittest.main()
