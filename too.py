from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, flash, url_for
import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import uuid
from datetime import datetime
import tensorflow as tf
import io
import numpy as np
import librosa
import soundfile as sf
import tempfile
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import traceback
SCALER_LOCAL_PATH = "scaler.pkl"

if os.path.exists(SCALER_LOCAL_PATH):
    scaler = joblib.load(SCALER_LOCAL_PATH)
    print("Scaler loaded successfully!")
else:
    raise FileNotFoundError("scaler.pkl not found! Make sure it's included in the deployment.")





# Load environment variables
load_dotenv()

# Fetch the Azure Blob Storage connection string from environment variables
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

if not connection_string:
    raise ValueError("Azure Storage Connection String not found. Ensure it is set in the .env file.")

import traceback  # Add this to the top of your script

def extract_features(file_path):
    try:
        print(f"Extracting features from: {file_path}")
        
        # Load the audio file
        y, sr = librosa.load(file_path, sr=22050)  # Convert to 22050 Hz
        print(f"Audio Loaded: y.shape={y.shape}, sr={sr}")

        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Ensure valid shapes
        if mfccs.shape[1] == 0 or chroma.shape[1] == 0 or mel_spec.shape[1] == 0:
            raise ValueError("Feature extraction resulted in empty arrays!")

        # Compute means
        mfccs_mean = np.mean(mfccs, axis=1)
        chroma_mean = np.mean(chroma, axis=1)
        mel_spec_mean = np.mean(mel_spec_db, axis=1)

        # Concatenate
        features = np.hstack((mfccs_mean, chroma_mean, mel_spec_mean))
        print(f"Extracted Features Shape: {features.shape}")

        # Convert to DataFrame
        feature_df = pd.DataFrame([features], columns=scaler.feature_names_in_)
        features_scaled = scaler.transform(feature_df)

        return features_scaled

    except Exception as e:
        print(f"Error extracting features: {e}")
        traceback.print_exc()  # Prints the full error message
        return None


# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Access the container ("uploads")
container_name = "uploads"
container_client = blob_service_client.get_container_client(container_name)

# Define model container and filename
MODEL_CONTAINER_NAME = "models"
MODEL_BLOB_NAME = "cnn_model.keras"

# Define local model path to avoid re-downloading
MODEL_LOCAL_PATH = "model.keras"

def load_model():
    try:
        # Check if model is already downloaded
        if os.path.exists(MODEL_LOCAL_PATH):
            print("Loading model from local storage...")
            return tf.keras.models.load_model(MODEL_LOCAL_PATH)

        print("Downloading model from Azure Blob Storage...")
        model_container_client = blob_service_client.get_container_client(MODEL_CONTAINER_NAME)
        blob_client = model_container_client.get_blob_client(MODEL_BLOB_NAME)

        # Download and save model
        model_stream = io.BytesIO(blob_client.download_blob().readall())
        with open(MODEL_LOCAL_PATH, "wb") as f:
            f.write(model_stream.read())

        print(f"Loading model from {MODEL_LOCAL_PATH}...")
        model = tf.keras.models.load_model(MODEL_LOCAL_PATH)

        print("Model loaded successfully!")
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model at startup
model = load_model()


# Flask application setup
app = Flask(__name__)
app.secret_key = "your-secret-key"

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            flash('No file part', 'upload-error')
            return redirect(request.url)

        file = request.files['file']

        # Ensure a valid file is selected
        if file.filename == '':
            flash('No file selected', 'upload-error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            try:
                # Upload the file to Azure Blob Storage
                blob_client = container_client.get_blob_client(blob=filename)
                blob_client.upload_blob(file, overwrite=True)

                # Redirect to the prediction page after upload
                return redirect(url_for('predict_emotion', filename=filename))

            except Exception as e:
                print(f"Error uploading to Azure or processing file: {e}")
                flash("File upload failed. Please try again.", 'upload-error')

        else:
            flash('Invalid file type. Please upload a valid audio file.', 'upload-error')

        return redirect(request.url)

    return render_template('upload.html')


# About Project page
@app.route('/about')
def about():
    return render_template('about.html')

# Dataset page
@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

# ML Model page
@app.route('/ml_model')
def ml_model():
    return render_template('ml_model.html')

# Record voice page
@app.route('/record', methods=['GET', 'POST'])
def record():
    if request.method == 'POST':
        # Check if the form has a file
        if 'audio-file' not in request.files:
            flash('No file uploaded!', 'record-error')
            return redirect(request.url)

        file = request.files['audio-file']

        if file and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)

            # Generate a unique filename (server-side)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_filename = f"{timestamp}_{original_filename}"

            try:
                # Upload the file to Azure Blob Storage
                blob_client = container_client.get_blob_client(blob=unique_filename)
                blob_client.upload_blob(file, overwrite=True)

                # After uploading, redirect to the prediction page
                return redirect(url_for('predict_emotion', filename=unique_filename))

            except Exception as e:
                print(f"Error uploading to Azure Blob Storage: {e}")
                flash("File upload failed. Please try again.", 'record-error')

        else:
            flash('Invalid file type. Please upload a valid audio file.', 'record-error')

        return redirect(request.url)

    return render_template('record.html')




@app.route('/files')
def list_files():
    try:
        # List all blobs (files) in the container
        blob_list = container_client.list_blobs()
        files = [
            {
                "name": blob.name,
                "url": f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob.name}"
            }
            for blob in blob_list
        ]
        return render_template('files.html', files=files)
    except Exception as e:
        print(f"Error listing files: {e}")
        flash("Could not fetch the list of files. Please try again later.", "files-error")
        return redirect(url_for('index'))

# Delete file
@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    try:
        # Get a reference to the blob client
        blob_client = container_client.get_blob_client(blob=filename)
        
        # Delete the file from Azure Blob Storage
        blob_client.delete_blob()

        flash(f'File "{filename}" deleted successfully.', 'files-success')
    except Exception as e:
        flash(f'Error deleting file "{filename}": {e}', 'files-error')
    
    return redirect('/files')



@app.route('/predict/<filename>', methods=['GET'])
def predict_emotion(filename):
    temp_audio_path = f"temp_{filename}"
    
    try:
        # Download the file from Azure Blob Storage
        blob_client = container_client.get_blob_client(blob=filename)
        file_stream = io.BytesIO(blob_client.download_blob().readall())

        # Save the file temporarily
        with open(temp_audio_path, "wb") as f:
            f.write(file_stream.read())

        # Extract features from the audio
        features = extract_features(temp_audio_path)
        if features is None:
            raise ValueError("Feature extraction failed!")

         

        # Run prediction using the CNN model
        prediction = model.predict(features)  
        print(f"Raw Model Output: {prediction}")  # Print probabilities
        predicted_label = np.argmax(prediction)
        print(f"Predicted Label Index: {predicted_label}")  # Print index

        # Map the prediction to an emotion
        emotion_labels = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Frustration", "Excitement"]
        predicted_emotion = emotion_labels[predicted_label]
        print(f"Predicted Emotion: {predicted_emotion}")  # Print emotion

        # Render the prediction page with emotion and animation
        return render_template('prediction.html', filename=filename, emotion=predicted_emotion)
    
    except ValueError as e:
        flash(str(e), 'error')
        return redirect(url_for('index'))

    except Exception as e:
        flash("Something went wrong! Please try again.", 'error')
        print(f"Error processing prediction: {e}")
        return redirect(url_for('index'))
    
    finally:
        # Always delete the temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)