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
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load environment variables
load_dotenv()

# Fetch the Azure Blob Storage connection string from environment variables
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

if not connection_string:
    raise ValueError("Azure Storage Connection String not found. Ensure it is set in the .env file.")

def extract_features(file_path):
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist!")
            return None

        if os.path.getsize(file_path) == 0:
            print(f"Error: File {file_path} is empty!")
            return None

        print(f"Extracting features from: {file_path}")
        
        # Hardcoded mean and std values
        mean = np.array([-20.297016])
        std = np.array([42.73151]) 
        std[std==0] = 1 # Prevent devision by zero

        # Constants (Ensure consistency with training)
        FRAME_LENGTH = 2048
        HOP_LENGTH = 512
        MAX_LEN = 130

        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        print(f"Audio Loaded: y.shape={y.shape}, sr={sr}")

        # Extract features (matching training process)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram)
        mel_spectrogram_db = pad_sequences([mel_spectrogram_db.T], maxlen=MAX_LEN, padding='post', truncating='post')

        zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        zcr = pad_sequences([zcr.T], maxlen=MAX_LEN, padding='post', truncating='post')

        rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        rms = pad_sequences([rms.T], maxlen=MAX_LEN, padding='post', truncating='post')

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
        mfcc = pad_sequences([mfcc.T], maxlen=MAX_LEN, padding='post', truncating='post')

        # Check dimensions before concatenation
        print(f"Shapes before concatenation: Mel={mel_spectrogram_db.shape}, ZCR={zcr.shape}, RMS={rms.shape}, MFCC={mfcc.shape}")

        # Combine features in the same order as training
        features = np.concatenate([mel_spectrogram_db, zcr, rms, mfcc], axis=-1)

        # Apply normalization (Standardization)
        features = (features - mean) / std

        # Reshape to match model input (batch_size, 130, 79, 1)
        features = features.reshape(1, 130, 79, 1)

        print(f"Final feature shape: {features.shape}")

        return features  # Return normalized features

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Access the container ("uploads")
container_name = "uploads"
container_client = blob_service_client.get_container_client(container_name)

# Define model container and filename
MODEL_CONTAINER_NAME = "models"
MODEL_BLOB_NAME = "model.keras"

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
        # List all blobs in the container
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

@app.route('/features')
def features():
    return render_template('feature_extraction.html')

@app.route('/choose')
def choose():
    samples_folder = "static/samples"

    # Ensure directory exists
    if not os.path.exists(samples_folder):
        os.makedirs(samples_folder)

    # Get only audio files (.wav, .mp3, .m4a)
    audio_extensions = (".wav", ".mp3", ".ogg", ".m4a")
    local_files = [f for f in os.listdir(samples_folder) if f.endswith(audio_extensions)]


    if not local_files:
        flash("No audio files found in 'static/samples/'. Please add some files.", "error")

    return render_template('choose.html', files=local_files)



@app.route('/predict_existing', methods=['POST'])
def predict_existing_file():
    filename = request.form.get('filename')

    if not filename:
        flash("No file selected! Please choose an audio file.", "error")
        return redirect(url_for('choose'))

    return redirect(url_for('predict_local_emotion', filename=filename))



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

@app.route('/predict_local/<filename>', methods=['GET'])
def predict_local_emotion(filename):
    file_path = os.path.join("static/samples", filename)

    try:
        # Check if file exists locally
        if not os.path.exists(file_path):
            flash("Local file not found!", "error")
            return redirect(url_for('choose'))

        # Extract features from the local audio file
        features = extract_features(file_path)
        if features is None:
            raise ValueError("Feature extraction failed for local file!")

        # Run prediction using the CNN model
        prediction = model.predict(features)  
        predicted_label = np.argmax(prediction)

        # Map the prediction to an emotion
        emotion_labels = ["Anger", "Frustration", "Happiness", "Neutral", "Sad"]
        predicted_emotion = emotion_labels[predicted_label]

        return render_template('prediction.html', filename=filename, emotion=predicted_emotion)

    except Exception as e:
        flash("Something went wrong! Please try again.", 'error')
        print(f"Error processing local file prediction: {e}")
        return redirect(url_for('choose'))


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
        emotion_labels = ["Anger", "Frustration", "Happiness", "Neutral", "Sad"]
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
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)