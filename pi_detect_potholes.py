import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import serial
import pynmea2
import threading
from datetime import datetime
import requests
import base64
import json

# Initialize Pygame mixer
pygame.mixer.init()

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
SOUND_DIR = os.path.join(BASE_DIR, "sounds")

# GPS Configuration
GPS_PORT = '/dev/serial0'
GPS_BAUD = 9600

# FIREBASE CONFIGURATION
FIREBASE_PROJECT_ID = "potholeproject-15a36"  
FIREBASE_STORAGE_BUCKET = "potholeproject-15a36.firebasestorage.app"  
FIREBASE_DATABASE_URL = "https://potholeproject-15a36-default-rtdb.firebaseio.com"  

# Global variables
sound_playing = False
sound_end_time = 0
prev_object_count = 0
current_gps = {"lat": None, "lon": None, "timestamp": None}
last_detection_time = 0
DETECTION_COOLDOWN = 5

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--source', help='Image source (file, folder, video, or usb0)', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold', default=0.25)
parser.add_argument('--resolution', help='Resolution WxH', default=None)
parser.add_argument('--record', help='Record results to video', action='store_true')
parser.add_argument('--no-gps', help='Disable GPS', action='store_true')
parser.add_argument('--save-local', help='Also save images locally', action='store_true')

args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record
use_gps = not args.no_gps
save_local = args.save_local

if save_local:
    LOCAL_SAVE_DIR = os.path.join(BASE_DIR, "pothole_detections")
    os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

# GPS Reader Thread
def gps_thread():
    global current_gps
    try:
        ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)
        print("GPS initialized")
        while True:
            try:
                line = ser.readline().decode('ascii', errors='replace')
                if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                    msg = pynmea2.parse(line)
                    if hasattr(msg, 'latitude') and msg.latitude:
                        current_gps["lat"] = msg.latitude
                        current_gps["lon"] = msg.longitude
                        current_gps["timestamp"] = datetime.utcnow().isoformat()
            except Exception as e:
                pass
    except Exception as e:
        print(f"GPS Error: {e}")
        print("Running without GPS")

# Upload Image to Firebase Storage + Save Metadata to Database
def upload_to_firebase(image, metadata):
    
    try:
        # Generate unique ID
        timestamp_id = str(int(time.time() * 1000))
        
        # Convert image to base64 (for simple upload via REST API)
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Upload metadata + base64 image to Realtime Database
        # (For production, use Firebase Storage SDK for better performance)
        data = {
            **metadata,
            "image_base64": image_base64,
            "image_id": timestamp_id
        }
        
        url = f"{FIREBASE_DATABASE_URL}/detections/{timestamp_id}.json"
        response = requests.put(url, json=data, timeout=10)
        
        if response.status_code == 200:
            print(f" Uploaded image {timestamp_id} to Firebase")
            return True
        else:
            print(f" Firebase upload failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f" Firebase error: {e}")
        return False

# Start GPS thread
if use_gps:
    gps_worker = threading.Thread(target=gps_thread, daemon=True)
    gps_worker.start()

# Audio Function
def play_sound(wav_file, duration):
    global sound_playing, sound_end_time

    if sound_playing:
        return 

    full_path = os.path.join(SOUND_DIR, wav_file)

    if not os.path.exists(full_path):
        print("Sound file not found:", full_path)
        return

    try:
        sound_playing = True
        sound_end_time = time.time() + duration
        pygame.mixer.music.load(full_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Audio error: {e}")

# Load Model
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid.')
    sys.exit(0)

print(f"Loading Model: {model_path}...")
model = YOLO(model_path, task='detect')
labels = model.names

# Handle Sources 
source_type = None
img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.avi','.mov','.mp4','.mkv']

if img_source.isnumeric():
    source_type = 'usb'
    usb_idx = int(img_source)
elif os.path.isdir(img_source):
    source_type = 'folder'
    imgs_list = glob.glob(img_source + '/*')
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list: source_type = 'image'
    elif ext in vid_ext_list: source_type = 'video'
else:
    source_type = 'video'

# Setup Capture
cap = None
if source_type == 'video':
    cap = cv2.VideoCapture(img_source)
elif source_type == 'usb':
    cap = cv2.VideoCapture(usb_idx)

if cap and not cap.isOpened():
    print("ERROR: Could not open video source")
    sys.exit(0)

# Set Resolution 
resW, resH = 640, 480 
if user_res and (source_type == 'video' or source_type == 'usb'):
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
    cap.set(3, resW)
    cap.set(4, resH)
elif cap:
    w = int(cap.get(3))
    h = int(cap.get(4))
    if w > 0 and h > 0:
        resW, resH = w, h
print(f"Resolution set to: {resW}x{resH}")

recorder = None
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video/camera.')
        sys.exit(0)
    record_name = 'demo_pi.avi'
    record_fps = 15
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = glob.glob(img_source + '/*')

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 20
img_count = 0

# Preprocessing Function
def preprocess_frame(frame):
    #Enhanced preprocessing for pothole detection
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    
    lab_eq = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

print("Starting detection... Press 'Q' to quit")
print(f"Images will be uploaded to Firebase when potholes are detected")

while True:
    # Audio Check
    if sound_playing and time.time() >= sound_end_time:
        sound_playing = False

    t_start = time.perf_counter()

    # Get Frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('Done.')
            break
        frame = cv2.imread(imgs_list[img_count])
        if frame is None:
            print(f"Could not read image: {imgs_list[img_count]}")
            img_count += 1
            continue
        img_count += 1
    else:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("End of video or camera error")
            break

    # Resize
    if frame.shape[1] != resW or frame.shape[0] != resH:
        frame = cv2.resize(frame, (resW, resH))

    # Keep original frame for saving
    original_frame = frame.copy()

    # Preprocessing 
    preprocessed_frame = frame #preprocessed_frame = preprocess_frame(frame) 
    
    # Run detection
    results = model(preprocessed_frame, verbose=False)
    detections = results[0].boxes

    object_count = 0
    max_confidence = 0
    detection_boxes = []
    
    for i in range(len(detections)):
        conf = detections[i].conf.item()
        
        if conf > min_thresh:
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy

            classidx = int(detections[i].cls.item())
            classname = labels[classidx] if hasattr(labels, '__getitem__') else 'pothole'
            object_count += 1
            
            if conf > max_confidence:
                max_confidence = conf
            
            detection_boxes.append({
                'class': classname,
                'confidence': float(conf),
                'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)]
            })
            
            # Draw RED Boxes
            color = (0, 0, 255)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw Labels
            label = f'{classname} {int(conf*100)}%'
            cv2.putText(frame, label, (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Sound logic
    current_time = time.time()
    if object_count > 0 and not sound_playing:
        if object_count == 1:
            play_sound("pothole_1.wav", duration=3.0)
        elif 2 <= object_count <= 3:
            play_sound("pothole_2.wav", duration=3.5)
        else:
            play_sound("pothole_3.wav", duration=4.0)

    # Save and Upload Image when pothole detected
    if object_count > 0 and (current_time - last_detection_time) > DETECTION_COOLDOWN:
        if current_gps["lat"] is not None and current_gps["lon"] is not None:
            
            # Prepare metadata
            metadata = {
                "latitude": current_gps["lat"],
                "longitude": current_gps["lon"],
                "timestamp": current_gps["timestamp"],
                "pothole_count": object_count,
                "max_confidence": float(max_confidence),
                "detections": detection_boxes,
                "device_id": "pi_001",
                "detected_at": datetime.utcnow().isoformat()
            }
            
            # Save locally if requested
            if save_local:
                timestamp_id = str(int(time.time() * 1000))
                local_img_path = os.path.join(LOCAL_SAVE_DIR, f"pothole_{timestamp_id}.jpg")
                local_json_path = os.path.join(LOCAL_SAVE_DIR, f"pothole_{timestamp_id}.json")
                
                cv2.imwrite(local_img_path, frame)
                with open(local_json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f" Saved locally: {local_img_path}")
            
            # Upload to Firebase in background
            threading.Thread(
                target=upload_to_firebase, 
                args=(frame, metadata), 
                daemon=True
            ).start()
            
            last_detection_time = current_time
            
        elif use_gps:
            print(" GPS not ready, skipping save")

    prev_object_count = object_count

    # Display info
    fps_text = f'FPS: {avg_frame_rate:.1f}'
    cv2.putText(frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f'Potholes: {object_count}', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    if use_gps and current_gps["lat"]:
        gps_text = f'GPS: {current_gps["lat"]:.6f}, {current_gps["lon"]:.6f}'
        cv2.putText(frame, gps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Pi Pothole Detector', frame)
    #cv2.imshow('Model View', preprocessed_frame)

    if record and recorder is not None:
        recorder.write(frame)

    if source_type in ['image', 'folder']:
        key = cv2.waitKey(0)
    else:
        key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q'): 
        break
    elif key == ord('p') or key == ord('P'): 
        cv2.imwrite('capture.png', frame)

    # FPS Calc
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start)) if (t_stop - t_start) > 0 else 0
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Cleanup
if cap: 
    cap.release()
if recorder: 
    recorder.release()
cv2.destroyAllWindows()
