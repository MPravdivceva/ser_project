from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, flash
import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Fetch the Azure Blob Storage connection string from environment variables
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

if not connection_string:
    raise ValueError("Azure Storage Connection String not found. Ensure it is set in the .env file.")

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Access the container (e.g., "uploads")
container_name = "uploads"
container_client = blob_service_client.get_container_client(container_name)

# Flask application setup
app = Flask(__name__)
app.secret_key = "your-secret-key"  # Replace with a secure key

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

                # Get the public URL of the uploaded file
                file_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{filename}"
                flash(f"File uploaded successfully! File URL: {file_url}", 'upload-success')
            except Exception as e:
                print(f"Error uploading to Azure Blob Storage: {e}")
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

                # Get the public URL of the uploaded file
                file_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{unique_filename}"
                flash(f"File uploaded successfully! File URL: {file_url}", 'record-success')
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

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)
