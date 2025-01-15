from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, flash
import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
load_dotenv()

import os
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')


# Fetch the Azure Blob Storage connection string from environment variables
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Access the container (e.g., "uploads")
container_name = "uploads"
container_client = blob_service_client.get_container_client(container_name)


app = Flask(__name__)
app.secret_key = "your-secret-key"  # Replace with a secure key

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Upload page
# Define allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}


# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['file']

        # Ensure a valid file is selected
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            try:
                # Upload the file to Azure Blob Storage
                blob_client = container_client.get_blob_client(blob=filename)
                blob_client.upload_blob(file, overwrite=True)

                # Get the public URL of the uploaded file
                file_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{filename}"
                flash(f"File uploaded successfully! File URL: {file_url}", 'success')
            except Exception as e:
                print(f"Error uploading to Azure Blob Storage: {e}")
                flash("File upload failed. Please try again.", 'danger')
        else:
            flash('Invalid file type. Please upload a valid audio file.', 'danger')

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
@app.route('/record', methods=['GET'])
def record():
    return render_template('record.html')


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

