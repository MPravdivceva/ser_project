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

def extract_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=22050)  # Convert to 22050 Hz

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Extract Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Extract Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        
        # Convert Mel Spectrogram to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Flatten and concatenate features
        features = np.hstack((
            np.mean(mfccs, axis=1),  # Take mean across time axis
            np.mean(chroma, axis=1),
            np.mean(mel_spec_db, axis=1)
        ))

        return np.expand_dims(features, axis=0)  # Reshape for model input

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Load environment variables
load_dotenv()

# Fetch the Azure Blob Storage connection string from environment variables
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

if not connection_string:
    raise ValueError("Azure Storage Connection String not found. Ensure it is set in the .env file.")

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Access the container ("uploads")
container_name = "uploads"
container_client = blob_service_client.get_container_client(container_name)

# Define model container and filename
MODEL_CONTAINER_NAME = "models"
MODEL_BLOB_NAME = "cnn_model.keras"

# Load model from Azure Blob Storage
def load_model():
    try:
        print("Downloading model from Azure Blob Storage...")
        model_container_client = blob_service_client.get_container_client(MODEL_CONTAINER_NAME)
        blob_client = model_container_client.get_blob_client(MODEL_BLOB_NAME)
        
        # Download model file
        model_stream = io.BytesIO(blob_client.download_blob().readall())

        # Save the model to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_model_file:
            temp_model_file.write(model_stream.getbuffer())
            temp_model_path = temp_model_file.name  # Save the file path

        print(f"Loading model from {temp_model_path}...")
        
        # Load the model from the saved file
        model = tf.keras.models.load_model(temp_model_path)

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
    try:
        # Download the file from Azure Blob Storage
        blob_client = container_client.get_blob_client(blob=filename)
        file_stream = io.BytesIO(blob_client.download_blob().readall())

        # Save the file temporarily
        temp_audio_path = f"temp_{filename}"
        with open(temp_audio_path, "wb") as f:
            f.write(file_stream.read())

        # Extract features from the audio
        features = extract_features(temp_audio_path)

        if features is None:
            return render_template("error.html", message="Failed to extract features from audio file.")

        # Run prediction using the CNN model
        prediction = model.predict(features)
        predicted_label = np.argmax(prediction)

        # Map the prediction to an emotion
        emotion_labels = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Frustration", "Excitement"]
        predicted_emotion = emotion_labels[predicted_label]

        # Render the prediction page with emotion and animation
        return render_template('prediction.html', filename=filename, emotion=predicted_emotion)

    except Exception as e:
        print(f"Error processing prediction: {e}")
        return render_template("error.html", message="Failed to process prediction.")


# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)
