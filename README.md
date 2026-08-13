Vision-based Obstacle Tracking and IMU for Robotic Fish Steering Navigation

A closed-loop perception–control system for an autonomous robotic fish that combines vision-based obstacle detection (YOLOv8 + adaptive image enhancement) with inertial heading stabilization (MPU6050 IMU) to steer around obstacles underwater. Steering commands are computed with an Artificial Potential Field (APF) controller and sent wirelessly to an MG995 servo actuating the tail fin.

Based on: A. A. Jamesh, D. Prabhu, Pranesh M, Sandhiya D, A. VM, "Vision-based Obstacle Tracking and IMU for the Robotic Fish Steering Navigation," Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham.

Overview

Underwater navigation is difficult because of turbid water, poor and variable lighting, and current-induced drift. This project addresses that with a perception → decision → control pipeline:

An ESP32 camera streams frames over Wi-Fi to a host machine.
A lightweight HPNN (Hyperparameter Neural Network) predicts per-frame enhancement parameters (color balance, contrast, sharpening) to counter underwater scattering, dimming, and desaturation — trained end-to-end with the detector, no clean-image ground truth needed.
YOLOv8n detects obstacles in the enhanced frame and returns bounding boxes.
Bounding box position/size are converted into a horizontal displacement and estimated distance, feeding an Artificial Potential Field (APF) controller that computes a repulsive steering force, with an adaptive gain based on a temporally-smoothed obstacle "risk" metric.
An MPU6050 IMU provides yaw feedback; a PD controller corrects heading drift caused by currents and servo actuation delay.
Vision-derived steering and IMU correction are fused and smoothed, then sent as a PWM command to the MG995 servo driving the tail fin.
Key results (from hardware-in-the-loop testing)
Metric	Value
Detection mAP@0.50	0.733
Detection mAP@0.50:0.95	0.456
Precision / Recall	~0.78 / 0.71
Processing rate	~5.0 FPS
Control update frequency	~2.1 Hz
End-to-end system latency	380–500 ms

IMU-based stabilization measurably reduced servo oscillation compared to vision-only control.

System Architecture
ESP32 Camera ──HTTP──▶ HPNN (Adaptive Image Enhancement)
                              │
                              ▼
                     YOLOv8 Detection (GPU)
                              │
                              ▼
                Obstacle Analysis (centroid, area)
                              │
                              ▼
              Artificial Potential Field (steering force)
                              │            ▲
                              ▼            │ IMU yaw feedback + PD correction
                     Steering Fusion & Smoothing
                              │
                              ▼
                  Servo Command (PWM) ──▶ MG995 Actuator
Hardware
ESP32 microcontroller (with OV2640 camera module, 640×480)
MPU6050 6-DOF IMU (3-axis accelerometer + 3-axis gyroscope)
MG995 servo motor (tail fin actuation)
Host PC/laptop with GPU (runs HPNN + YOLOv8 inference)
Wi-Fi link between ESP32 and host
Core Algorithms
Adaptive Image Enhancement (AIE): channel-wise color balancing → contrast enhancement → unsharp-mask sharpening, all parameterized by the HPNN and differentiable so detection loss backpropagates through it.
Distance estimation: normalized bounding box area, inversely proportional to squared distance (monocular, no depth sensor).
APF steering: repulsive force per obstacle beyond a proximity threshold, temporally smoothed, with adaptive gain from a rolling risk metric.
Heading stabilization: PD control on yaw error from the IMU (Kp = 0.6, Kd = 0.25).
Fusion: vision steering angle + IMU correction, exponentially smoothed (α = 0.35) before being sent to the servo every 0.12 s.
