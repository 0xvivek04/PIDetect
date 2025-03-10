# PIDetect - A PID-Based Process Control & Anomaly Detection

This project implements a **PID-based process control system** with anomaly detection, leveraging **Python, Streamlit, and Scikit-learn**. The system allows users to **adjust PID parameters**, visualize system behavior, and detect anomalies using **Isolation Forest**. It simulates a **first-order system** and provides real-time insights into control performance.

---

## Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Performance Metrics](#-performance-metrics)
- [Anomaly Detection](#-anomaly-detection)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Connect With Me](#-connect-with-me)

---

## Features
✅ **PID Controller Implementation**: Adjustable proportional, integral, and derivative gains.  
✅ **First-Order System Simulation**: Uses **ODE integration** for realistic process modeling.  
✅ **Real-Time Performance Metrics**: Computes **rise time, settling time, overshoot, and IAE**.  
✅ **Anomaly Detection**: Detects irregular behavior using **Isolation Forest**.  
✅ **Interactive Visualization**: **Streamlit dashboard** for intuitive tuning and monitoring.  
✅ **Customizable Setpoint**: Define your desired process control target.  

---

## Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/v1v3x/PIDetect.git
cd PIdetect
```
### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```
### **3. Run the Streamlit App**
```bash
streamlit run app.py
```

---

## 🎯 Usage
- Open the **Streamlit web interface**.
- Adjust **Kp, Ki, and Kd** values in the sidebar.
- Set a **desired process target (setpoint)**.
- Monitor **real-time graphs** of process behavior.
- View **performance metrics** to evaluate control efficiency.
- Detect **anomalies** and analyze control instability.

---

## 🔍 How It Works
1. **PID Controller**: Computes control outputs using **error-based feedback**.
2. **Process Model**: Simulates a **first-order system** using **ODE integration**.
3. **Performance Metrics**: Analyzes system response based on **control theory principles**.
4. **Anomaly Detection**: Identifies system deviations with **Isolation Forest**.
5. **Interactive Visualization**: Plots **process variables and control signals** in real time.

### **📌 Control System Simulation Details**
- The system follows a **first-order process model** with a **time constant (τ) of 10**.
- **PID tuning parameters (Kp, Ki, Kd)** influence system behavior.
- The simulation runs for **200 seconds** with a step size of **0.1 seconds**.

---

## 📊 Performance Metrics
| **Metric**         | **Description** |
|------------------|-------------|
| **Rise Time**   | Time taken to reach **90% of the setpoint** |
| **Settling Time** | Time required for the system to **stabilize within 5%** of the setpoint |
| **Overshoot**   | Maximum deviation **beyond the setpoint (%)** |
| **IAE (Integral Absolute Error)** | **Accumulated error** over time |

---

## ⚠️ Anomaly Detection
Anomalies in system behavior are detected using **Isolation Forest**:
- **Outlier Detection**: Identifies deviations in **process variable and control signal**.
- **Visualization**: Highlights anomalies as **red markers** in the graph.
- **Threshold-Based Classification**: 10% of data points are considered anomalies.

### **Anomaly Detection Implementation**
```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_anomalies(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(data_scaled)
    return anomalies == -1  # Returns True for anomalies
```

---

## 🚀 Future Enhancements
- ✅ **Adaptive PID tuning** using machine learning.
- ✅ **Real-time cloud deployment** with **IoT integration**.
- ✅ **Advanced anomaly detection** with deep learning techniques.
- ✅ **Dynamic process modeling** for different industries.

---

## 🤝 Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a new branch (`feature-name`).
3. Commit your changes.
4. Push to the branch.
5. Open a pull request.

---

## 📜 License
This project is licensed under the **Apache 2.0 License**.

---


