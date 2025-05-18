import unittest
import os
from app import app, extract_features, model
from io import BytesIO
import numpy as np

class FlaskAppTestCase(unittest.TestCase):

    def setUp(self):
        # Set up test client
        self.app = app.test_client()
        self.app.testing = True

        # Create a small valid audio file to simulate uploads
        self.test_audio = BytesIO()
        self.test_audio.write(b'\x00' * 44100)  # Dummy 1 second silent audio
        self.test_audio.seek(0)

    def test_homepage(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Analyze Your Emotions", response.data)

    def test_upload_get(self):
        response = self.app.get('/upload')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Upload Your Audio File", response.data)

    def test_invalid_upload_post(self):
        response = self.app.post('/upload', data={})
        self.assertIn(b"No file part", response.data)

    def test_upload_valid_file(self):
        response = self.app.post(
            '/upload',
            content_type='multipart/form-data',
            data={'file': (self.test_audio, 'test.wav')},
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Result", response.data)  # Because redirect goes to prediction

    def test_prediction_route_with_existing_file(self):
        features = np.random.randn(1, 130, 79, 1)
        prediction = model.predict(features)
        self.assertTrue(prediction.shape[1], 5)

    def test_record_page(self):
        response = self.app.get('/record')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Record Your Voice", response.data)

    def test_dataset_page(self):
        response = self.app.get('/dataset')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Dataset", response.data)

    def test_model_page(self):
        response = self.app.get('/ml_model')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"ML Model", response.data)

    def test_about_page(self):
        response = self.app.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"About This Project", response.data)

if __name__ == '__main__':
    unittest.main()
