# Speech Emotion Recognition (SER) Web Application

This project is a web-based **Speech Emotion Recognition (SER)** system that uses machine learning to classify emotions from speech recordings. Users can upload audio files or record audio directly through the app, and the system will analyze the emotions present in the speech. Project is diployed on Azure and is accessible online-https://ser-project-fvh6g9d3ekbucgd4.ukwest-01.azurewebsites.net/. No installation is requered.

## Features

1. **Upload Recorded Files**: Users can upload `.wav`, `.mp3`, or `.m4a` files for emotion recognition.
2. **Record Audio**: Users can record their voice directly in the browser.
3. **Responsive Design**: The app works on desktop, tablet, and mobile devices.
4. **Customizable**: Built with Python and Flask, easily extendable with new features.

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Frontend**: HTML, CSS, JavaScript (with a responsive design)
- **Storage**: Azure Blob Storage (for production)
- **Deployment**: Designed for deployment on Azure

## Directory Structure

ser_project/ ├── app.py # Main Flask application ├── templates/ # HTML files │ ├── base.html # Base template for all pages │ ├── index.html # Home page │ ├── upload.html # Upload page │ ├── about.html # About project page │ ├── dataset.html # Dataset information page │ ├── ml_model.html # ML model details page ├── static/ # Static files │ ├── style.css # CSS styles │ ├── script.js # JavaScript ├── requirements.txt # Python dependencies ├── README.md # Project documentation (this file)

Throughout development, a number of public technical resources were reviewed to better understand best practices in web deployment, machine learning pipelines, and frontend design. These included GitHub repositories, official documentation, YouTube tutorials, and Kaggle notebooks. No source code was directly copied; however, selected components and techniques were adapted to suit the objectives of this project. Key areas of influence included:
•	Web application structure and file upload handling using Flask (e.g., Freecodecamp and Traversy Media tutorials on YouTube)
•	Feature extraction using Librosa, including MFCC and chroma vectors (based on GitHub repositories and Kaggle Learn content)
•	CNN-based model templates for SER (referenced and adapted from open-source projects and Kaggle notebooks)
•	Responsive frontend design informed by CSS templates and online tutorials
All external resources consulted are clearly referenced in the Reference List.
