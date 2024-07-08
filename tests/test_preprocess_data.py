import unittest
import pandas as pd
import numpy as np

from src.data import preprocess_data


DATA = {
    "condition": [np.nan, "good", "excellent"],
    "cylinders": [np.nan, 4, 6],
    "VIN": ["1HGCM82633A123456", np.nan, "1HGCM82633A789012"],
    "drive": ["fwd", np.nan, "rwd"],
    "size": [np.nan, "compact", "midsize"],
    "type": [np.nan, "sedan", "SUV"],
    "paint_color": [np.nan, "red", "blue"],
    "county": [np.nan, np.nan, np.nan],
    "posting_date": ["2022-01-01", "2022-01-02", "2022-01-03"],
    "image_url": ["url1", "url2", "url3"],
    "description": ["desc1", "desc2", "desc3"],
    "id": [1, 2, 3],
    "url": ["url1", "url2", "url3"],
    "region_url": ["url1", "url2", "url3"],
    "manufacturer": [np.nan, "ford", "toyota"],
    "model": [np.nan, "focus", "corolla"],
    "fuel": ["gas", np.nan, "diesel"],
    "title_status": ["clean", "salvage", np.nan],
    "transmission": ["automatic", "manual", np.nan],
    "year": [2000, np.nan, 2010],
    "odometer": [50000, np.nan, 75000],
    "lat": [34.05, np.nan, 36.16],
    "long": [-118.25, np.nan, -115.15],
    "price": [15000, 25000, 35000],
    "state": ["CA", "NV", "TX"],
    "region": ["west", "southwest", "south"],
}
MIN_MAX_SCALE_COLS = [
    "region",
    "year",
    "model",
    "title_status",
    "state",
    "manufacturer",
]


class TestPreprocessData(unittest.TestCase):
    def test_preprocess_data(self):
        # Create a mock dataframe
        df = pd.DataFrame(DATA)

        # Call the method
        X, y = preprocess_data(df)

        # Assertions
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.DataFrame)
        self.assertNotIn("price", X.columns)
        self.assertIn("price", y.columns)
        self.assertTrue(np.all(X[MIN_MAX_SCALE_COLS] >= 0))
        self.assertTrue(np.all(X[MIN_MAX_SCALE_COLS] <= 1))
        self.assertEqual(y.shape, (3, 1))
