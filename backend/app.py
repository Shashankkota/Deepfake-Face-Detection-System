from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import threading
import time

# Import our model handling functions
from model_handler import ModelHandler

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('model', exist_ok=True)

# Initialize model handler
model_handler = ModelHandler()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """Handle dataset CSV upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please provide a valid CSV file'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the dataset
        result = model_handler.load_dataset(filepath)
        
        if result['success']:
            return jsonify({
                'message': 'Dataset uploaded successfully',
                'filename': filename,
                'total_images': result['total_images'],
                'classes': result['classes']
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Start model training"""
    try:
        if not model_handler.dataset_loaded:
            return jsonify({'error': 'Please upload dataset first'}), 400
        
        # Start training in a separate thread
        def train_async():
            model_handler.train_model()
        
        training_thread = threading.Thread(target=train_async)
        training_thread.start()
        
        return jsonify({'message': 'Training started successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training-status', methods=['GET'])
def training_status():
    """Get current training status"""
    return jsonify({
        'is_training': model_handler.is_training,
        'progress': model_handler.training_progress,
        'is_trained': model_handler.model_trained,
        'metrics': model_handler.get_metrics() if model_handler.model_trained else None
    })

@app.route('/api/detect', methods=['POST'])
def detect_deepfake():
    """Handle media upload for deepfake detection"""
    try:
        if not model_handler.model_trained:
            return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload image or video files.'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the file for detection
        start_time = time.time()
        result = model_handler.detect_deepfake(filepath)
        processing_time = round(time.time() - start_time, 2)
        
        if result['success']:
            return jsonify({
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'processing_time': processing_time,
                'filename': filename,
                'file_type': 'image' if filename.lower().endswith(('.jpg', '.jpeg', '.png')) else 'video'
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information and status"""
    return jsonify({
        'model_loaded': model_handler.model_trained,
        'dataset_loaded': model_handler.dataset_loaded,
        'is_training': model_handler.is_training,
        'training_progress': model_handler.training_progress,
        'metrics': model_handler.get_metrics() if model_handler.model_trained else None
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Deepfake Detection Server...")
    print("Loading face detection model...")
    
    # Check if Haar cascade file exists
    if not os.path.exists('model/haarcascade_frontalface_default.xml'):
        print("WARNING: Haar cascade file not found!")
        print("Please download haarcascade_frontalface_default.xml and place it in the 'model' folder.")
        print("You can download it from: https://github.com/opencv/opencv/tree/master/data/haarcascades")
    
    print("Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)