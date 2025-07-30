# Deepfake Detection Web Application

This web application converts the original Python Tkinter deepfake detection system into a fully functional Flask-based web application while maintaining all existing functionalities.

## Features

- **Dataset Upload**: Upload CSV datasets with deepfake face data
- **LSTM Model Training**: Train neural network models with progress tracking
- **Real-time Detection**: Analyze images and videos for deepfake detection
- **Performance Metrics**: View model accuracy, precision, recall, and F1-score
- **Responsive UI**: Beautiful, modern web interface with AI-themed design
- **Error Handling**: Comprehensive error handling and user feedback

## Project Structure

```
backend/
├── app.py              # Main Flask application
├── model_handler.py    # ML model handling functions
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Main web interface
├── static/
│   ├── style.css      # Custom CSS styling
│   └── script.js      # Frontend JavaScript
├── uploads/           # Uploaded files storage
└── model/             # ML model storage
```

## Setup Instructions

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Download Required Files

Download the Haar cascade file for face detection:
```bash
mkdir -p model
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml -O model/haarcascade_frontalface_default.xml
```

### 3. Prepare Dataset Structure

Create the following directory structure:
```
Dataset/
└── faces_224/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

The CSV file should contain columns:
- `videoname`: Image filename (without extension)
- `label`: Classification label (e.g., "Real", "Deepfake")

### 4. Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage Guide

### 1. Upload Dataset
- Click "Upload CSV" and select your dataset file
- The system will process and validate the dataset
- View the console for loading status and statistics

### 2. Train Model
- Click "Start Training" after dataset upload
- Monitor training progress with the progress bar
- View performance metrics once training completes

### 3. Detect Deepfakes
- Upload an image (JPG, PNG) or video (MP4, AVI, MOV)
- Click "Analyze" to start detection
- View results including prediction, confidence, and processing time

## API Endpoints

- `POST /api/upload-dataset` - Upload dataset CSV file
- `POST /api/train-model` - Start model training
- `GET /api/training-status` - Get training progress
- `POST /api/detect` - Analyze media for deepfake detection
- `GET /api/model-info` - Get model status information

## Configuration

### File Upload Limits
- Maximum file size: 100MB
- Supported formats: CSV, JPG, JPEG, PNG, MP4, AVI, MOV, MKV

### Model Configuration
- Input image size: 32x32 pixels
- Architecture: LSTM with CNN layers
- Training epochs: 50 (configurable)
- Batch size: 64

## Dependencies

### Required
- Flask: Web framework
- OpenCV: Computer vision and face detection
- NumPy: Numerical computations
- Pandas: Data manipulation
- Scikit-learn: Machine learning utilities

### Optional (for full ML functionality)
- TensorFlow/Keras: Deep learning framework
- Matplotlib: Plotting and visualization

## Production Deployment

### Using Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key-here
```

### Nginx Configuration (example)
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    client_max_body_size 100M;
}
```

## Error Handling

The application includes comprehensive error handling for:
- Invalid file formats
- Missing model files
- Training failures
- Detection errors
- Network issues

## Security Considerations

- File upload validation
- Secure filename handling
- CORS configuration
- Input sanitization
- Error message sanitization

## Performance Optimization

- Async training processes
- Progress tracking
- Memory management
- File cleanup
- Caching strategies

## Troubleshooting

### Common Issues

1. **Haar cascade file not found**
   - Download the file from OpenCV repository
   - Place in `model/haarcascade_frontalface_default.xml`

2. **No faces detected**
   - Ensure images contain clear, front-facing faces
   - Check image quality and lighting

3. **Training fails**
   - Verify dataset format and structure
   - Check available memory and disk space
   - Ensure TensorFlow/Keras is installed

4. **File upload errors**
   - Check file size limits
   - Verify file format support
   - Ensure proper permissions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project maintains the same license as the original codebase.