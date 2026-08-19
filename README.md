# Vision-Based Obstacle Tracking and IMU for Robotic Fish Steering Navigation

A vision-inertial navigation framework for autonomous robotic fish that combines **HPNN-guided Adaptive Image Enhancement (AIE)**, **YOLOv8 underwater obstacle detection**, **Artificial Potential Field (APF) navigation**, and **MPU6050-based heading stabilization**.

The system follows a closed-loop **perception–decision–control** architecture in which underwater images are captured by an ESP32-CAM, enhanced using HPNN-guided AIE, processed by YOLOv8 for obstacle detection, converted into steering commands using APF, and stabilized using IMU-based PD feedback before being sent to an MG995 servo actuator.

---

## Abstract

Autonomous underwater navigation is challenging because underwater environments introduce **turbidity, varying lighting conditions, light scattering, color degradation, and water-induced disturbances**.

This project presents a robotic fish navigation framework that combines visual obstacle perception with inertial feedback. An **ESP32-CAM** captures underwater image frames and transmits them to a host computer. Before obstacle detection, the captured frames are processed by an **HPNN-guided Adaptive Image Enhancement (AIE)** module.

The enhanced frames are then processed using a lightweight **YOLOv8n** detector to identify obstacles and generate bounding boxes. The position and normalized area of the detected obstacles are used by an **Artificial Potential Field (APF)** controller to calculate a repulsive steering response.

At the same time, an **MPU6050 inertial measurement unit** provides yaw information. A **Proportional-Derivative (PD) controller** uses the heading deviation to generate a stabilization correction. The visual steering command and IMU correction are combined and smoothed before being transmitted wirelessly to an **MG995 servo motor**, which actuates the robotic fish tail fin.

The system was evaluated through indoor and hardware-in-the-loop experiments. The reported results demonstrate underwater obstacle detection, obstacle avoidance, and improved heading stability when visual perception and inertial feedback are combined.

---

# 1. System Overview

The proposed system consists of four major stages:

1. **Perception** – ESP32-CAM image acquisition, HPNN-guided AIE, and YOLOv8 obstacle detection.
2. **Obstacle Analysis** – extraction of bounding-box position and normalized area.
3. **Decision** – APF-based obstacle avoidance and steering computation.
4. **Control** – MPU6050-based PD heading stabilization, command smoothing, and MG995 servo actuation.

The paper describes this as a **perception–decision–control pipeline**.

---

# 2. System Architecture

```text
                         ┌─────────────────────┐
                         │      ESP32-CAM       │
                         │   Image Acquisition  │
                         └──────────┬──────────┘
                                    │
                              HTTP Video Stream
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      HPNN + AIE      │
                         │ Adaptive Enhancement │
                         │ Color / Contrast /   │
                         │ Sharpening           │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       YOLOv8        │
                         │ Obstacle Detection  │
                         └──────────┬──────────┘
                                    │
                             Bounding Boxes
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Obstacle Analysis   │
                         │ Position + Area     │
                         └──────────┬──────────┘
                                    │
                              Repulsive Force
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │        APF          │
                         │ Steering Computation│
                         └──────────┬──────────┘
                                    │
                              Vision Angle
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │      Steering Fusion        │
                    │                             │
                    │ Vision + IMU PD Correction │
                    │        + Smoothing          │
                    └──────────────┬──────────────┘
                                   │
                              Servo Command
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │     MG995 Servo     │
                         │    Tail Actuator    │
                         └─────────────────────┘

                    ┌─────────────────────┐
                    │     MPU6050 IMU     │
                    │    Yaw Estimation   │
                    └──────────┬──────────┘
                               │
                         Heading Feedback
                               │
                               ▼
                         PD Correction
```

The architecture shown in Figure 1 of the paper integrates **HPNN-AIE preprocessing, YOLOv8 detection, APF navigation, IMU-based PD stabilization, and actuator control** within a closed-loop perception–control framework.

---

# 3. Hardware Platform

The robotic fish platform consists of:

* **ESP32-CAM**
* **OV2640 camera**
* **MPU6050 IMU**
* **MG995 servo motor**
* Robotic fish tail-fin mechanism
* Host computer with GPU for computationally intensive processing

The ESP32 handles communication, the camera captures image frames, the MPU6050 provides inertial measurements, and the MG995 servo controls the tail fin.

The computationally intensive deep-learning processing is performed on a separate host device rather than directly on the ESP32.

---

# 4. HPNN-Guided Adaptive Image Enhancement (AIE)

## 4.1 Motivation

Underwater images are affected by:

* Light scattering
* Dimming
* Reduced contrast
* Color degradation
* Loss of visual details

These effects make underwater obstacle detection more difficult.

The proposed system therefore introduces an **Adaptive Image Enhancement (AIE)** module before YOLOv8 detection.

Instead of applying one fixed enhancement configuration to every frame, the system uses an **HPNN** to analyze the input frame and predict enhancement parameters for that particular frame.

---

## 4.2 HPNN

The paper describes HPNN as a neural network that receives a **downsampled version of the input frame** and predicts **five scalar parameters** for the enhancement process.

These parameters are:

* `Wr` – red-channel weight
* `Wg` – green-channel weight
* `Wb` – blue-channel weight
* `ω` – contrast coefficient
* `λ` – sharpening factor

The HPNN therefore acts as an adaptive parameter generator for the AIE module.

### HPNN-AIE Pipeline

```text
                    Input Frame
                         │
                         ▼
                 Downsampled Frame
                         │
                         ▼
                  ┌─────────────┐
                  │    HPNN     │
                  │ Hyperparameter
                  │   Network   │
                  └──────┬──────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
     Wr, Wg, Wb          ω              λ
   Channel Weights   Contrast       Sharpening
                     Coefficient      Factor
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                  Adaptive AIE
                         │
                         ▼
                  Enhanced Frame
                         │
                         ▼
                      YOLOv8
```

---

## 4.3 Channel-Wise Color Balancing

The first enhancement operation adjusts the RGB channel contributions:

```text
Iw = (Wr Ir, Wg Ig, Wb Ib)
```

where:

* `Ir` = red channel
* `Ig` = green channel
* `Ib` = blue channel
* `Wr`, `Wg`, `Wb` = channel weights generated by HPNN

This allows the enhancement process to adapt the color balance of each underwater frame.

---

## 4.4 Contrast Enhancement

The contrast enhancement stage is represented by:

```text
Ic = ω En(I) + (1 − ω)I
```

where:

* `I` = original image
* `En(I)` = enhanced image
* `ω` = HPNN-generated contrast coefficient

The parameter `ω` determines the contribution of the enhanced image relative to the original image.

---

## 4.5 Detail Sharpening

The sharpening operation enhances image edges and details:

```text
Is = I + λ(I − Gaussian(I))
```

where:

* `I` = input image
* `Gaussian(I)` = Gaussian-blurred image
* `λ` = sharpening factor predicted by HPNN

The resulting sharpened image `Is` is passed to the YOLOv8 detector.

---

## 4.6 Complete AIE Pipeline

```text
Raw Underwater Frame
        │
        ▼
      HPNN
        │
        ├───────────────┐
        │               │
        ▼               ▼
Channel Weights    Enhancement
(Wr, Wg, Wb)       Parameters
                    (ω, λ)
        │               │
        ▼               ▼
Color Balancing
        │
        ▼
Contrast Enhancement
        │
        ▼
Detail Sharpening
        │
        ▼
Enhanced Frame
        │
        ▼
      YOLOv8
        │
        ▼
Obstacle Detection
```

Figure 2 of the paper illustrates this HPNN-guided AIE pipeline, where the HPNN predicts channel weights, contrast coefficient, and sharpening factor before the enhanced image is passed to YOLOv8.

---

## 4.7 End-to-End Training

The paper specifies that **HPNN and the YOLOv8 detection backbone are trained end-to-end**.

A downsampled input frame is provided to HPNN, which predicts the enhancement parameters. The resulting enhancement operations are differentiable, allowing the detection loss to be backpropagated through the AIE module.

```text
Input Frame
     │
     ▼
    HPNN
     │
     ▼
Enhancement Parameters
     │
     ▼
Differentiable AIE
     │
     ▼
Enhanced Image
     │
     ▼
   YOLOv8
     │
     ▼
Detection Loss
     │
     └──────────────► Backpropagation
                          │
                    HPNN + YOLOv8
                       Parameters
```

This means that HPNN learns enhancement parameters according to their usefulness for the **downstream detection task**.

The paper also states that a separate clear-image ground truth is **not required** for training the HPNN-AIE module.

---

# 5. YOLOv8 Obstacle Detection

The enhanced image is passed to a **YOLOv8** object detector.

The paper uses the **YOLOv8n (Nano)** variant because its lightweight architecture provides a suitable balance between detection accuracy and inference speed for embedded robotic applications.

The detector returns bounding boxes represented by:

```text
(x1, y1, x2, y2)
```

These bounding boxes are then used by the obstacle-analysis and navigation modules.

## Object Classes

The detection dataset contains seven classes:

```text
fish
jellyfish
penguin
puffin
shark
starfish
stingray
```

---

# 6. Obstacle Position Analysis

For each detected obstacle, the horizontal distance from the image center is normalized as:

```text
dᵢ = (cᵢ − W/2) / (W/2)
```

where:

* `cᵢ` = center of the obstacle
* `W` = image width
* `dᵢ` = normalized horizontal displacement

This value represents the direction and magnitude of the steering contribution.

---

# 7. Obstacle Proximity Estimation

A monocular camera does not directly provide depth information.

Therefore, the system uses the normalized bounding-box area as an approximation of obstacle proximity:

```text
Aᵢ = ((x₂ − x₁)(y₂ − y₁)) / (H × W)
```

where:

* `H` = image height
* `W` = image width
* `x₁, y₁, x₂, y₂` = bounding-box coordinates

Under perspective projection:

```text
Aᵢ ∝ 1 / Dᵢ²
```

where `Dᵢ` represents the distance to the obstacle.

Thus, a larger bounding box indicates that an obstacle is closer or occupies a larger portion of the visual field.

Duplicate detections are filtered using the horizontal distance between bounding-box centers. If two centers are less than **50 pixels apart horizontally**, they are considered duplicates and only one representative detection is retained.

---

# 8. Artificial Potential Field (APF) Navigation

The detected obstacles are converted into repulsive steering forces using an **Artificial Potential Field**.

For each obstacle:

```text
Fᵢ =
    −kr dᵢ (Aᵢ − Ath),    if Aᵢ > Ath
     0,                     otherwise
```

where:

* `kr` = repulsion gain
* `dᵢ` = normalized horizontal displacement
* `Aᵢ` = normalized bounding-box area
* `Ath` = proximity threshold

The steering force is calculated for detected obstacles and temporally smoothed to reduce sudden changes caused by detection noise.

---

## 8.1 Adaptive Steering Gain

The system maintains a temporal risk metric based on recent obstacle bounding-box areas.

The adaptive gain is:

```text
Gs(t) = G0 (1 + ρ̄(t))
```

where:

* `G0` = base steering gain
* `ρ̄(t)` = smoothed obstacle-risk metric

The vision steering command is then:

```text
θv = θ0 + Gs Fs
```

where:

* `θ0` = servo neutral angle
* `Fs` = smoothed APF force

The adaptive gain allows the steering response to change according to the recent obstacle-risk history while maintaining smoother navigation in cluttered scenes.

---

# 9. IMU-Based Heading Stabilization

The robotic fish uses an **MPU6050** inertial measurement unit containing a three-axis accelerometer and three-axis gyroscope.

The initial yaw angle is stored as the reference heading:

```text
ψr = ψ(0)
```

The heading error is:

```text
eψ = ψr − ψ(t)
```

The PD correction is:

```text
Cimu = Kp eψ + Kd deψ/dt
```

The gains reported in the paper are:

```text
Kp = 0.6
Kd = 0.25
```

The IMU correction compensates for heading deviations caused by environmental disturbances and servo actuation.

---

# 10. Steering Command Fusion

The visual steering command and IMU correction are combined:

```text
θ(t) = θv(t) + Cimu(t)
```

The resulting command is smoothed using:

```text
θs(t) = α θ(t) + (1 − α) θs(t − 1)
```

where `α` controls the response speed of the steering system.

The reported smoothing factor is:

```text
α = 0.35
```

The final smoothed steering command is sent to the MG995 servo.

---

# 11. Complete Navigation Algorithm

The navigation procedure described in the paper is:

```text
1. Initialize camera and IMU
2. Store reference yaw ψr ← ψ(0)
3. Obtain camera frame
4. Apply HPNN-guided AIE
5. Run YOLOv8 detection
6. Extract bounding boxes
7. Calculate obstacle area Ai
8. Calculate horizontal displacement di
9. Remove duplicate detections
10. Calculate APF force Fi
11. Smooth APF force Fs
12. Calculate adaptive gain Gs
13. Calculate vision steering θv
14. Read IMU yaw ψ(t)
15. Calculate PD correction Cimu
16. Fuse vision and IMU commands
17. Smooth final steering command θs
18. Send θs to MG995 servo
19. Repeat
```

The paper notes that detection can be performed periodically, while the control loop continues independently. Intermediate frames can reuse the most recent detection, and actuator commands are updated every **0.12 seconds**.

---

# 12. Control Parameters

The parameters reported in Table I of the paper are:

| Parameter                     |  Value |
| ----------------------------- | -----: |
| Yaw proportional gain `Kp`    |    0.6 |
| Yaw derivative gain `Kd`      |   0.25 |
| Base steering gain `G0`       |     45 |
| Repulsion gain `kr`           |      8 |
| Proximity threshold `Ath`     |   0.12 |
| Steering smoothing factor `α` |   0.35 |
| Actuator update interval      | 0.12 s |

---

# 13. Dataset

The training dataset was collected using the **OV2640 camera at 640 × 480 resolution**.

The dataset was:

* Annotated with bounding boxes
* Divided into training, validation, and test subsets
* Split using an **80:10:10 ratio**
* Processed using normalization and augmentation techniques

The dataset structure is:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
```

---

# 14. Model Training

The YOLOv8 detection backbone was initialized using pretrained weights and trained on the underwater object-detection dataset.

The selected model is **YOLOv8n**, chosen for its lightweight architecture and suitability for real-time robotic applications.

The repository contains:

```text
models/
└── best.pt
```

for the trained detector.

Training configuration and results are stored under:

```text
results/
```

---

# 15. Experimental Results

The paper reports the following YOLOv8 detection results:

| Metric    | Result |
| --------- | -----: |
| mAP@50    |  0.733 |
| mAP@50–95 |  0.456 |
| Accuracy  |  ~0.78 |
| Recall    |   0.71 |

The results indicate reliable obstacle detection across the object classes used in the experiments.

---

# 16. Runtime Performance

The reported runtime results are:

| Measurement               |     Result |
| ------------------------- | ---------: |
| Detection processing rate |   ~5.0 FPS |
| Control update frequency  |    ~2.1 Hz |
| System latency            | 380–500 ms |

## The evaluation considered detection quality, navigation quality, servo behavior, latency, control frequency, and detection processing rate.

# 17. Obstacle Avoidance Results

The system was evaluated under four obstacle configurations:

| Scenario           |  APF Force | Steering | Motion      |
| ------------------ | ---------: | -------: | ----------- |
| No obstacle        |       0.00 |   88–92° | Forward     |
| Left barrier       | +0.40–0.60 | 105–118° | Right Avoid |
| Right obstacle     | −0.40–0.60 |   60–70° | Left Avoid  |
| Multiple obstacles |      ±0.30 |  85–108° | Adaptive    |

Positive APF force produces right steering, while negative force produces left steering. In free space, the servo returns approximately to its neutral position.

---

# 18. Experimental Observation

Initial testing showed that heading drift increased when steering was based entirely on visual information.

The addition of MPU6050 feedback helped correct heading deviations and stabilize the steering behavior.

The experiments therefore demonstrate the benefit of combining **visual obstacle perception with inertial feedback** for robotic fish navigation.

The paper's conclusion also reports that IMU-based heading stabilization reduced the servo oscillation that was observed under pure vision-based control.

---

# 19. Repository Structure

```text
.
├── .git/
│
├── configs/
│   └── data.yaml
│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── docs/
│   └── paper.pdf
│
├── models/
│   └── best.pt
│
├── results/
│   ├── results.csv
│   ├── args.yaml
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   └── ...
│
├── robo2.py
├── servo_new.ino
└── README.md
```

### `configs/`

Contains the YOLO dataset configuration.

### `dataset/`

Contains the underwater object-detection images and YOLO annotations.

### `models/`

Contains the trained YOLOv8 model.

### `results/`

Contains training history, evaluation results, confusion matrices, curves, and prediction outputs.

### `robo2.py`

Contains the Python-side robotic fish processing/control implementation.

### `servo_new.ino`

Contains the embedded Arduino/ESP32-side servo control code.

### `docs/paper.pdf`

Contains the IEEE-style research paper describing the proposed navigation system and experimental evaluation.

---

# 20. Software

The project uses:

* **Python**
* **PyTorch**
* **YOLOv8**
* **OpenCV**
* **Arduino IDE**
* **ESP32 firmware**
* **Wi-Fi communication**
* **HTTP communication**

The host computer performs computationally intensive visual processing, while the ESP32 provides camera acquisition and embedded communication/control.

---

# 21. Limitations

The reported experiments were conducted indoors using artificial barriers.

The current system has a reported processing rate of approximately **5 FPS** and a latency of **380–500 ms**.

Real underwater environments may introduce additional challenges, including:

* Stronger water disturbances
* Changing illumination
* Turbidity
* Dynamic obstacles
* Unstructured underwater environments
* Greater communication and processing delays

---

# 22. Future Work

The paper identifies several future directions:

* Fully autonomous marine exploration
* Open-water testing
* Integration of propulsion control
* Handling dynamic and unstructured underwater environments
* Reinforcement-learning-based navigation

---

# 23. Research Paper

**Title:**
*Vision-based Obstacle Tracking and IMU for the Robotic Fish Steering Navigation*

**Authors:**

* Abdul Azhar Jamesh
* Deepak Prabhu
* Pranesh M
* Sandhiya D
* Akhil VM

**Affiliation:**
Amrita School of Artificial Intelligence
Amrita Vishwa Vidyapeetham, India

The complete IEEE-style paper is available at:

```text
docs/paper.pdf
```

---

# 24. References

1. A. Prakash, A. R. Nair, H. Arunav, R. P. Rthuraj, V. M. Akhil, C. Tawk, and K. V. Shankar, “Bioinspiration and biomimetics in marine robotics: A review on current applications and future trends,” *Bioinspiration & Biomimetics*, vol. 19, no. 3, 2024.

2. J. Han, R. Huang, X. Yuan, Y. Liu, B. Yin, S. Zou, and Z. Ma, “Vision-Based Obstacle Avoidance and Formation Control for Underwater Robotic Fish,” *IEEE Robotics and Automation Letters*, vol. 10, no. 10, pp. 10442–10449, 2025.

3. H. Jian, “Angle Detection System Design Based on MPU6050,” *Proc. International Conference on Electrical Engineering and Mechatronics Control*, 2017.

4. R. Varghese and M. Sambath, “YOLOv8: An Innovative Object Detection Algorithm with Improved Robustness and Performance,” *Proc. International Conference on Advanced Data Engineering and Intelligent Computing Systems*, 2024.

5. O. Khatib, “Real-Time Obstacle Avoidance for Manipulators and Mobile Robots,” *Proc. IEEE International Conference on Robotics and Automation*, 1985.

6. C.-Y. Li, J.-C. Guo, R.-M. Cong, Y.-W. Pang, and B. Wang, “Underwater Image Enhancement by Dehazing With Minimum Information Loss and Histogram Distribution Prior,” *IEEE Transactions on Image Processing*, vol. 25, no. 12, pp. 5664–5677, 2016.

7. D. Akkaynak and T. Treibitz, “A Revised Underwater Image Formation Model,” *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2018.

8. T. Qin, P. Li, and S. Shen, “VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator,” *IEEE Transactions on Robotics*, vol. 34, no. 4, pp. 1004–1020, 2018.

---

# 25. Authors

**Abdul Azhar Jamesh**
**Deepak Prabhu**
**Pranesh M**
**Sandhiya D**
**Akhil VM**

**Amrita School of Artificial Intelligence**
**Amrita Vishwa Vidyapeetham, India**
