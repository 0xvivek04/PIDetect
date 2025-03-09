import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.integrate import odeint

# PID Controller
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def compute(self, process_variable, dt):
        error = self.setpoint - process_variable
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output

# Simulate process (first-order system)
def process_model(y, t, u):
    K = 2.0  # process gain
    tau = 10.0  # time constant
    dydt = (-y + K * u) / tau
    return dydt

# Simulate control system
def simulate_control_system(pid, t_span, dt):
    t = np.arange(t_span[0], t_span[1], dt)
    y = np.zeros_like(t)
    u = np.zeros_like(t)
    y[0] = 20.0  # initial temperature

    for i in range(1, len(t)):
        u[i] = pid.compute(y[i-1], dt)
        y[i] = odeint(process_model, y[i-1], [t[i-1], t[i]], args=(u[i],))[-1]

    return t, y, u

# Calculate control system performance metrics
def calculate_performance_metrics(t, y, u, setpoint):
    rise_time = np.interp(0.9 * setpoint, y, t) - np.interp(0.1 * setpoint, y, t)
    settling_time = np.where(np.abs(y - setpoint) <= 0.05 * setpoint)[0][0] * (t[1] - t[0])
    overshoot = max(0, (np.max(y) - setpoint) / setpoint * 100)
    iae = np.sum(np.abs(setpoint - y)) * (t[1] - t[0])
    return {
        "Rise Time": rise_time,
        "Settling Time": settling_time,
        "Overshoot": overshoot,
        "IAE": iae
    }

# Detect anomalies using Isolation Forest
def detect_anomalies(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(data_scaled)
    return anomalies == -1

# Main Streamlit app
def main():
    st.title("Process Control System Performance Monitoring")

    # Sidebar for parameter input
    st.sidebar.header("Control Parameters")
    Kp = st.sidebar.slider("Proportional Gain (Kp)", 0.1, 10.0, 1.0)
    Ki = st.sidebar.slider("Integral Gain (Ki)", 0.0, 1.0, 0.1)
    Kd = st.sidebar.slider("Derivative Gain (Kd)", 0.0, 10.0, 0.5)
    setpoint = st.sidebar.slider("Setpoint", 30.0, 80.0, 50.0)

    # Simulate control system
    pid = PIDController(Kp, Ki, Kd, setpoint)
    t_span = (0, 200)
    dt = 0.1
    t, y, u = simulate_control_system(pid, t_span, dt)

    # Calculate performance metrics
    metrics = calculate_performance_metrics(t, y, u, setpoint)

    # Detect anomalies
    anomalies = detect_anomalies(np.column_stack((y, u)))

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(t, y, label="Process Variable")
    ax1.plot(t, [setpoint] * len(t), "--", label="Setpoint")
    ax1.scatter(t[anomalies], y[anomalies], color='red', label="Anomalies")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temperature")
    ax1.legend()
    ax1.set_title("Process Variable vs Time")

    ax2.plot(t, u, label="Control Signal")
    ax2.scatter(t[anomalies], u[anomalies], color='red', label="Anomalies")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Control Signal")
    ax2.legend()
    ax2.set_title("Control Signal vs Time")

    st.pyplot(fig)

    # Display performance metrics
    st.header("Performance Metrics")
    for metric, value in metrics.items():
        st.metric(metric, f"{value:.2f}")

    # Display anomaly information
    st.header("Anomaly Detection")
    anomaly_count = np.sum(anomalies)
    st.write(f"Number of detected anomalies: {anomaly_count}")
    st.write("Anomalies are highlighted in red on the plots.")

if __name__ == "__main__":
    main()