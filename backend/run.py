#!/usr/bin/env python3
"""
Deepfake Detection Web Application Launcher
This script provides an easy way to start the Flask application with proper configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'flask_cors', 'cv2', 'numpy', 'pandas', 'sklearn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'flask_cors':
                import flask_cors
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install them with: pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """Check if required model files exist"""
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    haar_cascade_path = model_dir / "haarcascade_frontalface_default.xml"
    
    if not haar_cascade_path.exists():
        print("⚠️  Haar cascade file not found!")
        print(f"   Expected location: {haar_cascade_path}")
        print("   You can download it from:")
        print("   https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
        print("\n💡 Download command:")
        print(f"   wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml -O {haar_cascade_path}")
        return False
    
    return True

def setup_directories():
    """Create required directories"""
    directories = ["uploads", "model", "Dataset/faces_224"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def main():
    """Main launcher function"""
    print("🚀 Deepfake Detection Web Application")
    print("=====================================")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]}")
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✓ All dependencies installed")
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Check model files
    print("\n🧠 Checking model files...")
    if not check_model_files():
        print("\n⚠️  Application will run with limited functionality")
        print("   Face detection may not work without the Haar cascade file")
        choice = input("\nContinue anyway? (y/n): ").lower().strip()
        if choice != 'y':
            sys.exit(1)
    else:
        print("✓ Model files found")
    
    # Start the application
    print("\n🌐 Starting Flask application...")
    print("   Access the application at: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()