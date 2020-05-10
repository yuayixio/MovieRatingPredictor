import unittest
import src.predict
import numpy as np
import os

class TestSubmission(unittest.TestCase):
    def setUp(self):
        dirname = os.path.dirname(__file__)
        os.chdir(os.path.join(dirname, "src"))
        self.test_data, self.test_data_with_rating = src.predict.predict_ratings()

    def test_data_prediction_has_rating_column(self):
        """
        test if prediction dataframe has a "rating" column
        """
        self.assertIn("rating", self.test_data_with_rating.columns)

    def test_data_prediction_length_equals_test_data(self):
        """
        test if prediction dataframe is of same length
        """
        self.assertEqual(len(self.test_data), len(self.test_data_with_rating))

    def test_data_with_ratings_has_no_missing_values(self):
        """
        test if prediction dataframe has no missing values
        """
        self.assertFalse(self.test_data_with_rating["rating"].isnull().values.any())

    def test_ratings_in_test_data_with_ratings_is_float(self):
        """
        test if prediction dataframe's column "rating" is of type float
        """
        self.assertTrue(self.test_data_with_rating["rating"].dtype == np.float64)
