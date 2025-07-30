// Global variables
let modelStatus = {
    datasetLoaded: false,
    modelTrained: false,
    isTraining: false
};

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    updateModelStatus();
    setInterval(checkTrainingStatus, 1000); // Check training status every second
    logToConsole('System initialized. Ready for dataset upload.', 'info');
});

// Console logging
function logToConsole(message, type = 'info') {
    const consoleLog = document.getElementById('console-log');
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    logEntry.textContent = `[${timestamp}] ${message}`;
    consoleLog.appendChild(logEntry);
    consoleLog.scrollTop = consoleLog.scrollHeight;
}

// Update model status display
function updateModelStatus() {
    fetch('/api/model-info')
        .then(response => response.json())
        .then(data => {
            modelStatus = data;
            
            // Update status badges
            document.getElementById('dataset-status').textContent = data.dataset_loaded ? 'Loaded' : 'Not Loaded';
            document.getElementById('dataset-status').className = `badge ${data.dataset_loaded ? 'bg-success' : 'bg-secondary'}`;
            
            document.getElementById('training-status').textContent = data.is_training ? 'Training...' : 
                (data.model_loaded ? 'Complete' : 'Not Started');
            document.getElementById('training-status').className = `badge ${data.is_training ? 'bg-warning' : 
                (data.model_loaded ? 'bg-success' : 'bg-secondary')}`;
            
            document.getElementById('model-status').textContent = data.model_loaded ? 'Ready' : 'Not Ready';
            document.getElementById('model-status').className = `badge ${data.model_loaded ? 'bg-success' : 'bg-secondary'}`;
            
            // Update accuracy display
            if (data.metrics && data.metrics.accuracy) {
                document.getElementById('accuracy-display').textContent = `${data.metrics.accuracy}%`;
                document.getElementById('accuracy-display').className = 'badge bg-success';
            }
            
            // Update button states
            document.getElementById('train-btn').disabled = !data.dataset_loaded || data.is_training || data.model_loaded;
            document.getElementById('detect-btn').disabled = !data.model_loaded;
            
            // Update metrics if available
            if (data.metrics) {
                updateMetrics(data.metrics);
            }
        })
        .catch(error => {
            console.error('Error fetching model status:', error);
            logToConsole('Error fetching model status', 'error');
        });
}

// Upload dataset
function uploadDataset() {
    const fileInput = document.getElementById('dataset-file');
    const file = fileInput.files[0];
    
    if (!file) {
        logToConsole('Please select a CSV file to upload', 'warning');
        return;
    }
    
    if (!file.name.endsWith('.csv')) {
        logToConsole('Please select a valid CSV file', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    logToConsole(`Uploading dataset: ${file.name}`, 'info');
    
    fetch('/api/upload-dataset', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            logToConsole(`Dataset upload failed: ${data.error}`, 'error');
        } else {
            logToConsole(`Dataset uploaded successfully: ${data.total_images} images found`, 'success');
            logToConsole(`Classes found: ${data.classes.join(', ')}`, 'info');
            updateModelStatus();
        }
    })
    .catch(error => {
        console.error('Error uploading dataset:', error);
        logToConsole('Error uploading dataset', 'error');
    });
}

// Train model
function trainModel() {
    if (!modelStatus.datasetLoaded) {
        logToConsole('Please upload dataset first', 'warning');
        return;
    }
    
    logToConsole('Starting LSTM model training...', 'info');
    
    fetch('/api/train-model', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            logToConsole(`Training failed: ${data.error}`, 'error');
        } else {
            logToConsole('Model training started successfully', 'success');
            document.getElementById('training-progress').classList.remove('d-none');
            document.getElementById('train-btn').disabled = true;
        }
    })
    .catch(error => {
        console.error('Error starting training:', error);
        logToConsole('Error starting training', 'error');
    });
}

// Check training status
function checkTrainingStatus() {
    if (!modelStatus.isTraining && !modelStatus.modelTrained) return;
    
    fetch('/api/training-status')
        .then(response => response.json())
        .then(data => {
            if (data.is_training) {
                const progressBar = document.getElementById('progress-bar');
                progressBar.style.width = `${data.progress}%`;
                progressBar.textContent = `${Math.round(data.progress)}%`;
                
                if (data.progress > 0) {
                    logToConsole(`Training progress: ${Math.round(data.progress)}%`, 'info');
                }
            } else if (data.is_trained && modelStatus.isTraining) {
                // Training just completed
                document.getElementById('training-progress').classList.add('d-none');
                logToConsole('Model training completed successfully!', 'success');
                
                if (data.metrics) {
                    logToConsole(`Final accuracy: ${data.metrics.accuracy}%`, 'success');
                    updateMetrics(data.metrics);
                }
                
                updateModelStatus();
            }
            
            modelStatus.isTraining = data.is_training;
            modelStatus.modelTrained = data.is_trained;
        })
        .catch(error => {
            console.error('Error checking training status:', error);
        });
}

// Update metrics display
function updateMetrics(metrics) {
    document.getElementById('accuracy-metric').textContent = `${metrics.accuracy}%`;
    document.getElementById('precision-metric').textContent = `${metrics.precision}%`;
    document.getElementById('recall-metric').textContent = `${metrics.recall}%`;
    document.getElementById('f1-metric').textContent = `${metrics.f1_score}%`;
    
    document.getElementById('metrics-section').classList.remove('d-none');
}

// Detect deepfake
function detectDeepfake() {
    const fileInput = document.getElementById('media-file');
    const file = fileInput.files[0];
    
    if (!file) {
        logToConsole('Please select an image or video file', 'warning');
        return;
    }
    
    if (!modelStatus.modelTrained) {
        logToConsole('Model not trained yet. Please train the model first.', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Show analysis progress
    document.getElementById('analysis-progress').classList.remove('d-none');
    document.getElementById('results-section').classList.add('d-none');
    document.getElementById('detect-btn').disabled = true;
    
    logToConsole(`Starting analysis of: ${file.name}`, 'info');
    
    fetch('/api/detect', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('analysis-progress').classList.add('d-none');
        document.getElementById('detect-btn').disabled = false;
        
        if (data.error) {
            logToConsole(`Detection failed: ${data.error}`, 'error');
        } else {
            logToConsole(`Analysis complete: ${data.prediction} (${data.confidence}% confidence)`, 'success');
            displayResults(data);
        }
    })
    .catch(error => {
        console.error('Error during detection:', error);
        logToConsole('Error during detection', 'error');
        document.getElementById('analysis-progress').classList.add('d-none');
        document.getElementById('detect-btn').disabled = false;
    });
}

// Display detection results
function displayResults(data) {
    const predictionElement = document.getElementById('prediction-result');
    predictionElement.textContent = data.prediction;
    predictionElement.style.color = data.prediction === 'Real' ? '#10b981' : '#ef4444';
    
    document.getElementById('confidence-result').textContent = `${data.confidence}%`;
    document.getElementById('time-result').textContent = `${data.processing_time}s`;
    
    document.getElementById('results-section').classList.remove('d-none');
    
    // Add result card animation
    const resultCard = document.querySelector('.result-card');
    resultCard.style.animation = 'none';
    setTimeout(() => {
        resultCard.style.animation = 'fadeIn 0.5s ease forwards';
    }, 10);
}

// File input change handlers
document.getElementById('dataset-file').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        logToConsole(`Selected dataset file: ${file.name}`, 'info');
    }
});

document.getElementById('media-file').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        logToConsole(`Selected media file: ${file.name}`, 'info');
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'u') {
        e.preventDefault();
        document.getElementById('dataset-file').click();
    } else if (e.ctrlKey && e.key === 'd') {
        e.preventDefault();
        document.getElementById('media-file').click();
    }
});

// Add visual effects
function addGlowEffect(element) {
    element.classList.add('pulse-glow');
    setTimeout(() => {
        element.classList.remove('pulse-glow');
    }, 2000);
}

// Error handling
window.addEventListener('error', function(e) {
    logToConsole(`JavaScript error: ${e.message}`, 'error');
});

// Add floating animation to cards on hover
document.querySelectorAll('.card').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.classList.add('float');
    });
    
    card.addEventListener('mouseleave', function() {
        this.classList.remove('float');
    });
});