# ============================================================
# FINAL RESEARCH VERSION
# Dynamic Gain + Logging + Density + UDP IMU
# ============================================================

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import requests
import socket
import threading
import csv
from collections import deque


DEVICE="cuda" if torch.cuda.is_available() else "cpu"

ESP32_IP="172.20.10.9"
ESP32_CAPTURE_URL="http://172.20.10.8/capture"

ESP32_SERVO_URL=f"http://{ESP32_IP}/servo"

YOLO_WEIGHTS=r"C:\Users\azhar\Downloads\Robotics\run\content\runs\detect\underwater_detection\weights\best.pt"

YOLO_SIZE=320
YOLO_SKIP_FRAMES=6

# CONTROL PARAMETERS
Kp_yaw=0.6
Kd_yaw=0.25

STEER_GAIN=45
REPULSION_GAIN=8
DIST_THRESHOLD=0.12

# UDP IMU
UDP_IP="0.0.0.0"
UDP_PORT=5005
imu_yaw=0

# GLOBALS
initial_yaw=None
prev_error=0

prev_servo_angle=90
last_servo_time=0
SERVO_INTERVAL=0.12

servo_energy=0

risk_history=deque(maxlen=10)

# LOGGING
log_file=open("experiment_log.csv","w",newline="")
writer=csv.writer(log_file)
writer.writerow([
"time",
"risk",
"force",
"yaw_error",
"servo_angle",
"latency",
"detection_fps",
"control_fps",
"density"
])

# UDP IMU LISTENER
def imu_listener():

    global imu_yaw

    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind((UDP_IP,UDP_PORT))
    sock.setblocking(False)

    while True:

        try:
            data,_=sock.recvfrom(1024)
            imu_yaw=float(data.decode())
        except:
            pass

# CAMERA
def get_frame():

    try:
        r=requests.get(ESP32_CAPTURE_URL,timeout=1)
        img=np.frombuffer(r.content,np.uint8)
        return cv2.imdecode(img,cv2.IMREAD_COLOR)

    except:
        return None

# SERVO
def send_servo(angle):

    try:
        requests.get(
            ESP32_SERVO_URL,
            params={"angle":int(angle)},
            timeout=0.15)

    except:
        pass

# FILTER DUPLICATE BOXES
def filter_boxes(boxes):

    filtered=[]

    for box in boxes:

        x1,y1,x2,y2=box
        cx=(x1+x2)/2

        keep=True

        for fb in filtered:

            fx1,fy1,fx2,fy2=fb
            fcx=(fx1+fx2)/2

            if abs(cx-fcx)<50:
                keep=False

        if keep:
            filtered.append(box)

    return filtered

# APF FORCE
def compute_force(detections,H,W):

    global risk_history

    F=0
    risk=0

    for (x1,y1,x2,y2) in detections:

        cx=(x1+x2)/2
        dx=(cx-W/2)/(W/2)

        area=(x2-x1)*(y2-y1)/(W*H)

        risk+=area

        if area>DIST_THRESHOLD:

            F+=-REPULSION_GAIN*dx*(area-DIST_THRESHOLD)

    risk_history.append(risk)

    return np.clip(F,-0.6,0.6),np.mean(risk_history)

# MAIN
def run():

    global initial_yaw
    global prev_error
    global prev_servo_angle
    global last_servo_time
    global servo_energy

    model=YOLO(YOLO_WEIGHTS)
    model.fuse()

    imu_thread=threading.Thread(target=imu_listener,daemon=True)
    imu_thread.start()

    skip_counter=0
    last_detections=[]

    video_writer=None

    print("System Started")

    while True:

        loop_start=time.time()

        frame=get_frame()

        if frame is None:
            continue

        H,W,_=frame.shape

        if video_writer is None:

            video_writer=cv2.VideoWriter(
                "experiment_output.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (W,H))

        # YOLO DETECTION
        detection_start=time.time()

        if skip_counter==0:

            result=model(
                frame,
                imgsz=YOLO_SIZE,
                conf=0.35,
                device=DEVICE,
                verbose=False)[0]

            boxes=[]

            for b in result.boxes:

                x1,y1,x2,y2=b.xyxy[0].cpu().numpy()
                boxes.append((x1,y1,x2,y2))

            last_detections=filter_boxes(boxes)

            skip_counter=YOLO_SKIP_FRAMES

        else:
            skip_counter-=1

        detection_time=time.time()-detection_start
        detection_fps=1/detection_time if detection_time>0 else 0

        detections=last_detections

        # DRAW BOXES
        for (x1,y1,x2,y2) in detections:

            cv2.rectangle(
                frame,
                (int(x1),int(y1)),
                (int(x2),int(y2)),
                (0,255,0),
                2)

        # OBSTACLE DENSITY
        density=len(detections)/(W*H)

        # APF
        force,risk=compute_force(detections,H,W)

        # DYNAMIC STEERING GAIN
        dynamic_gain=STEER_GAIN*(1+risk)

        vision_angle=90+force*dynamic_gain

        # IMU PD CONTROL
        yaw=imu_yaw

        if initial_yaw is None:
            initial_yaw=yaw

        yaw_error=initial_yaw-yaw

        d_error=yaw_error-prev_error
        prev_error=yaw_error

        imu_corr=Kp_yaw*yaw_error+Kd_yaw*d_error

        target=np.clip(
            vision_angle+imu_corr,
            60,120)

        # SERVO SMOOTHING
        alpha=0.35

        angle=int(
            prev_servo_angle+
            alpha*(target-prev_servo_angle)
        )

        # SERVO ENERGY METRIC
        servo_energy+=abs(angle-prev_servo_angle)

        prev_servo_angle=angle

        now=time.time()

        if now-last_servo_time>SERVO_INTERVAL:

            send_servo(angle)
            last_servo_time=now

        # LATENCY
        latency=time.time()-loop_start
        control_fps=1/latency

        # TERMINAL LOG
        print(
        f"Det:{len(detections)} "
        f"Risk:{risk:.3f} "
        f"Density:{density:.6f} "
        f"Force:{force:.2f} "
        f"YawErr:{yaw_error:.2f} "
        f"Servo:{angle} "
        f"Latency:{latency*1000:.1f}ms "
        f"DetFPS:{detection_fps:.1f} "
        f"CtrlFPS:{control_fps:.1f}"
        )

        # CSV LOG
        writer.writerow([
        time.time(),
        risk,
        force,
        yaw_error,
        angle,
        latency,
        detection_fps,
        control_fps,
        density
        ])

        video_writer.write(frame)

        cv2.imshow("Experiment",frame)

        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    video_writer.release()
    log_file.close()
    cv2.destroyAllWindows()


if __name__=="__main__":
    run()