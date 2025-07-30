
````markdown
# Deepfake Face Detection System (Web Version)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)
![Frontend](https://img.shields.io/badge/Frontend-HTML%20%7C%20CSS%20%7C%20Bootstrap-green.svg)
![Libraries](https://img.shields.io/badge/Libraries-OpenCV%20%7C%20TensorFlow%20%7C%20Keras-orange.svg)
![ML](https://img.shields.io/badge/Model-CNN%2BLSTM-red.svg)

---

## **🔹 Overview**  
The **Deepfake Face Detection System** is a web-based application designed to detect deepfake videos by analyzing facial regions and learning temporal-spatial features. The system uses a hybrid model combining **CNN (ResNeXt)** and **LSTM** to process extracted frames and predict authenticity.  

This project was originally a Python GUI application (Tkinter) and has now been converted into a fully functional **web application** using **Flask** for seamless deployment and accessibility.  

---

## **✨ Features**  
- Upload video files and detect if they are deepfakes.  
- Extracts frames using **Haar Cascade** face detection.  
- Uses **ResNeXt (CNN)** for spatial feature extraction.  
- **LSTM** captures temporal features across frames.  
- Displays prediction results with confidence scores.  
- Real-time metrics logging (Prometheus + Grafana integration).  

---

## **🛠️ Tech Stack**  
- **Frontend:** HTML, CSS, Bootstrap (Responsive UI)  
- **Backend:** Flask (Python)  
- **ML Model:** TensorFlow/Keras (ResNeXt + LSTM)  
- **Utilities:** OpenCV, NumPy, Pandas  
- **Monitoring:** Prometheus, Grafana (optional)  

---

## **⚡ Installation & Setup**  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/Shashankkota/Deepfake-Face-Detection-System.git
   cd Deepfake-Face-Detection-System
````

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**

   ```bash
   python app.py
   ```

5. **Access the app in your browser:**

   ```
   http://127.0.0.1:5000/
   ```

---

## **📊 Grafana Monitoring (Optional)**

This system supports **Prometheus metric logging**. You can set up Grafana dashboards to visualize model usage, latency, and accuracy trends. Refer to the `monitoring_setup.md` (if provided) for step-by-step instructions.

---


---


