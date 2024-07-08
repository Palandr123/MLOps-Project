import unittest
from unittest.mock import patch

from src.data import validate_initial_data


class TestValidateInitialData(unittest.TestCase):
    @patch("src.data.FileDataContext")
    def test_validate_initial_data(self, MockFileDataContext):
        # Mock the context and other components
        mock_context = MockFileDataContext.return_value
        mock_source = mock_context.sources.add_or_update_pandas.return_value
        mock_asset = mock_source.add_csv_asset.return_value
        mock_batch_request = mock_asset.build_batch_request.return_value
        mock_checkpoint = mock_context.add_or_update_checkpoint.return_value
        mock_checkpoint_result = mock_checkpoint.run.return_value
        mock_checkpoint_result.success = True

        # Call the method
        result = validate_initial_data()

        # Assertions
        self.assertTrue(result)
        mock_context.sources.add_or_update_pandas.assert_called_with("data_sample")
        mock_source.add_csv_asset.assert_called_with(
            name="data_sample_asset", filepath_or_buffer="data/samples/sample.csv"
        )
        mock_context.get_validator.assert_called_with(
            batch_request=mock_batch_request, expectation_suite_name="expectation_suite"
        )
        mock_context.add_or_update_checkpoint.assert_called()
        mock_checkpoint.run.assert_called()
