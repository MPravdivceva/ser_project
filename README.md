# Speech Emotion Recognition (SER) Web Application

This project is a web-based **Speech Emotion Recognition (SER)** system that uses machine learning to classify emotions from speech recordings. Users can upload audio files or record audio directly through the app, and the system will analyze the emotions present in the speech.

## Features

1. **Upload Recorded Files**: Users can upload `.wav`, `.mp3`, or `.m4a` files for emotion recognition.
2. **Record Audio**: Users can record their voice directly in the browser.
3. **Responsive Design**: The app works on desktop, tablet, and mobile devices.
4. **Customizable**: Built with Python and Flask, easily extendable with new features.

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Frontend**: HTML, CSS, JavaScript (with a responsive design)
- **Storage**: Azure Blob Storage (optional, for production)
- **Deployment**: Designed for deployment on platforms like Azure or AWS

## Directory Structure

ser_project/ ├── app.py # Main Flask application ├── templates/ # HTML files │ ├── base.html # Base template for all pages │ ├── index.html # Home page │ ├── upload.html # Upload page │ ├── about.html # About project page │ ├── dataset.html # Dataset information page │ ├── ml_model.html # ML model details page ├── static/ # Static files │ ├── style.css # CSS styles │ ├── script.js # JavaScript (if used for interactivity) ├── requirements.txt # Python dependencies ├── README.md # Project documentation (this file)

