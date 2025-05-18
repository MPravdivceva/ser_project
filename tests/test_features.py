import unittest
import os
import numpy as np
from app import extract_features

class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        # A sample test audio file path
        self.valid_file = "static/samples/Happiness.wav"
        self.empty_file = "static/samples/empty.wav"
        self.invalid_file = "static/samples/nonexistent.wav"

    def test_valid_file_output(self):
        """Test feature extraction with a valid audio file."""
        features = extract_features(self.valid_file)
        self.assertIsNotNone(features, "Expected non-None output for a valid file")
        self.assertEqual(features.shape, (1, 130, 79, 1), "Unexpected output shape")

    def test_invalid_path_returns_none(self):
        """Test feature extraction handles non-existent file."""
        features = extract_features(self.invalid_file)
        self.assertIsNone(features, "Expected None for invalid file path")

    def test_empty_file_returns_none(self):
        """Test that an empty audio file is rejected."""
        # Make sure the file is actually empty
        if os.path.exists(self.empty_file):
            self.assertEqual(os.path.getsize(self.empty_file), 0)
        features = extract_features(self.empty_file)
        self.assertIsNone(features, "Expected None for empty file")

    def test_feature_normalization_range(self):
        """Test that normalized values fall within range."""
        features = extract_features(self.valid_file)
        self.assertIsNotNone(features)
        self.assertTrue(np.max(features) < 5)
        self.assertTrue(np.min(features) > -5)

if __name__ == '__main__':
    unittest.main()
