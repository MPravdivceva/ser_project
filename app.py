from flask import Flask, render_template, request, redirect, flash

app = Flask(__name__)
app.secret_key = "your-secret-key"  # Replace with a secure key

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Simulate file upload success
        flash("File uploaded successfully! (Simulated for design)", 'success')
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
@app.route('/record')
def record():
    return render_template('record.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
