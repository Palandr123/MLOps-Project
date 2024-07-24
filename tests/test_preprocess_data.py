import unittest
from unittest.mock import patch, MagicMock, mock_open

import pandas as pd
import numpy as np

from src.data import preprocess_data


DATA = {
    'target': [10, 20, 30],
    'feature1': [1, 2, 2],
    'feature2': [5.0, 6.5, np.nan],
    'feature3': [1.0, np.nan, 2.0],
    'feature4': [0.1, 0.2, 0.3],
    'feature5': [1, 2, 3],
    'posting_date': ["2022-01-01", "2022-01-02", "2022-01-03"],
    'VIN': ['1HGCM82633A123456', '1HGCM82633A654321', '1HGCM82633A789012'],
    'cat_col1': ['A', 'B', 'A'],
    'cat_col2': ['X', 'Y', 'Z'],
    'unnecessary_col': [0, 1, 2]
}


class TestPreprocessData(unittest.TestCase):
    
    @patch('src.data.zenml.client.Client')  # Mocking zenml.client.Client
    @patch('src.data.compose')  # Mocking compose function
    @patch('builtins.open', new_callable=mock_open)
    def test_preprocess_data(self, mock_open, mock_compose, mock_client):
        # Mocking the config returned by compose
        mock_cfg = MagicMock()
        mock_cfg.data.target_cols = ['target']
        mock_cfg.data.target_low = 0
        mock_cfg.data.target_high = 100
        mock_cfg.data.drop_rows = ['feature1']
        mock_cfg.data.dt_feature = ['posting_date']
        mock_cfg.data.impute_most_frequent = ['feature1']
        mock_cfg.data.impute_median = ['feature2']
        mock_cfg.data.impute_mean = ['feature3']
        mock_cfg.data.min_max_scale = ['feature4']
        mock_cfg.data.std_scale = ['feature5']
        mock_cfg.data.ohe_cols = ['cat_col1']
        mock_cfg.data.label_cols = ['cat_col2']
        mock_cfg.data.periodic_transform = {
            'posting_date_month': {
                'offset': 0,
                'period': 12,
            },
            'posting_date_day':{
                'offset': 0,
                'period': 31
            }
        }
        mock_cfg.data.drop_cols = ['unnecessary_col']

        mock_compose.return_value = mock_cfg

        # Mocking the client
        mock_artifact = MagicMock()
        mock_artifact.load.return_value = MagicMock(
            transform=MagicMock(return_value=MagicMock(toarray=MagicMock(return_value=np.array([[1, 0], [0, 1]])))),
            get_feature_names_out=MagicMock(return_value=['ohe_feature1', 'ohe_feature2'])
        )
        mock_client.return_value.list_artifacts.return_value = [mock_artifact]

        # Creating a sample DataFrame
        df = pd.DataFrame(DATA)

        # Calling the function
        X, y = preprocess_data(df)
        # Asserting the results
        self.assertTrue('WMI' in X.columns)
        self.assertTrue('VDS' in X.columns)
        self.assertTrue('ohe_feature1' in X.columns)
        self.assertTrue('ohe_feature2' in X.columns)
        self.assertTrue('posting_date_day_cos' in X.columns)
        self.assertTrue('posting_date_day_sin' in X.columns)
        self.assertTrue('posting_date_month_cos' in X.columns)
        self.assertTrue('posting_date_month_sin' in X.columns)
        self.assertFalse('unnecessary_col' in X.columns)

        # Check y
        self.assertTrue((y == df[['target']]).all().all())

        self.assertTrue(mock_open.called)
        mock_open.assert_any_call('configs/ohe_out_names.yaml', 'w')
        mock_open.assert_any_call('configs/label_out_names.yaml', 'w')
