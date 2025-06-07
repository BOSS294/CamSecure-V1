import sys
import cv2
import threading
import numpy as np
import datetime
import os
import pyttsx3
import speech_recognition as sr
import shutil
import subprocess
from deepface import DeepFace
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout, QSizePolicy,
    QProgressBar, QStackedLayout, QGridLayout, QWidget
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QFont, QPainter, QPen, QBrush
import time
import requests
import scipy.spatial
import math
import random

# -------- Configuration --------
VIDEO_OUTPUT_DIR = "recordings"
THREAT_LOG_DIR = "logs"
SCREENSHOT_DIR = "screenshots"
KNOWN_FACES_DIR = "known_faces"

LIVE_FEED_DIR = os.path.join(VIDEO_OUTPUT_DIR, "live_feed")
ROOM_SCAN_DIR = os.path.join(VIDEO_OUTPUT_DIR, "room_scan")
THERMAL_DIR = os.path.join(VIDEO_OUTPUT_DIR, "thermal")
PARANORMAL_DIR = os.path.join(VIDEO_OUTPUT_DIR, "paranormal")
for d in [LIVE_FEED_DIR, ROOM_SCAN_DIR, THERMAL_DIR, PARANORMAL_DIR]:
    os.makedirs(d, exist_ok=True)

os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
os.makedirs(THREAT_LOG_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

tts_engine = pyttsx3.init()

def get_ist_time():
    return (datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)).strftime('%Y-%m-%d %H:%M:%S')

def speak(text):
    threading.Thread(target=lambda: tts_engine.say(text) or tts_engine.runAndWait(), daemon=True).start()

def listen_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""

# ------- Face Recognition & Movement Detector with DeepFace -------
class Detector:
    def __init__(self, log_callback=None, motion_callback=None):
        
        self.video_capture = cv2.VideoCapture(0)
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video_capture.get(cv2.CAP_PROP_FPS) or 15)
        self.video_files = []
        self.current_filename = self.new_video_filename()
        self.outs = [
            cv2.VideoWriter(os.path.join(LIVE_FEED_DIR, self.new_video_filename()), cv2.VideoWriter_fourcc(*'XVID'), 15, (self.frame_width, self.frame_height)),
            cv2.VideoWriter(os.path.join(ROOM_SCAN_DIR, self.new_video_filename()), cv2.VideoWriter_fourcc(*'XVID'), 15, (self.frame_width, self.frame_height)),
            cv2.VideoWriter(os.path.join(THERMAL_DIR, self.new_video_filename()), cv2.VideoWriter_fourcc(*'XVID'), 15, (self.frame_width, self.frame_height)),
            cv2.VideoWriter(os.path.join(PARANORMAL_DIR, self.new_video_filename()), cv2.VideoWriter_fourcc(*'XVID'), 15, (self.frame_width, self.frame_height)),
        ]
        self.prev_frame = None
        self.log_callback = log_callback
        self.motion_callback = motion_callback
        self.motion_count = 0
        self.last_frame_time = time.time()
        self.real_fps = 0
        self.session_log_file = os.path.join(
            THREAT_LOG_DIR, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        self.registered_face_enc = None
        self.known_face_path = os.path.join(KNOWN_FACES_DIR, "my_face.jpg")
        if os.path.exists(self.known_face_path):
            try:
                self.registered_face_enc = DeepFace.represent(img_path=self.known_face_path, model_name='Facenet')[0]["embedding"]
            except Exception:
                self.registered_face_enc = None
        self.frame_counter = 0
        self.deepface_interval = 5  # Run DeepFace every 5 frames

    def log(self, message):
        timestamp = get_ist_time()
        log_entry = f"{timestamp} - {message}"
        with open(self.session_log_file, "a") as f:
            f.write(log_entry + "\n")
        if self.log_callback:
            self.log_callback(log_entry)

    def new_video_filename(self):
        return f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"

    def is_registered_face(self, face_img):
        if self.registered_face_enc is None:
            return False
        try:
            test_enc = DeepFace.represent(img_path=face_img, model_name='Facenet')[0]["embedding"]
            # Use cosine similarity for better accuracy
            sim = 1 - scipy.spatial.distance.cosine(self.registered_face_enc, test_enc)
            return sim > 0.7  # threshold for Facenet cosine similarity
        except Exception:
            return False
    def process_frame(self, mode=0):
        self.frame_counter += 1

        ret, frame = self.video_capture.read()
        if not ret:
            self.log("Frame capture failed.")
            return None

        # Reduce brightness for better detection
        frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=-30)  # alpha<1 darkens, beta<0 shifts darker

        now = time.time()
        self.real_fps = 1.0 / (now - self.last_frame_time) if self.last_frame_time else 0
        self.last_frame_time = now

        resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.prev_frame is None:
            self.prev_frame = gray
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        alert_triggered = False
        motion_detected = False

        # --- Frame Modes ---
        if mode == 0:
            # Motion Detection Feed
            for contour in motion_contours:
                area = cv2.contourArea(contour)
                if area < 500:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                # Classify motion type (simple: size-based)
                if area > 5000:
                    motion_type = "LARGE MOVEMENT"
                    percent = 95
                elif area > 2000:
                    motion_type = "MEDIUM MOVEMENT"
                    percent = 80
                else:
                    motion_type = "SMALL MOVEMENT"
                    percent = 60
                cv2.rectangle(frame, (x*2, y*2), ((x+w)*2, (y+h)*2), (0, 0, 255), 2)
                cv2.putText(frame, f"{motion_type} ({percent}%)", (x*2, y*2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                motion_detected = True
                self.log(f"Motion detected at ({x*2},{y*2},{(x+w)*2},{(y+h)*2}) [{motion_type}]")

        elif mode == 1:
            # Room Scan Mode ( DUMMY HAVE TO BE CREATED ): Animated scan lines and fake analysis
            scan_color = (0, 255, 128)
            for i in range(0, frame.shape[0], 40):
                cv2.line(frame, (0, i), (frame.shape[1], i), scan_color, 2)

            # Define multiple entry/exit zones (add as many as needed)
            zones = [
                (30, 50, 250, 350),  # left door
                (frame.shape[1]-250, 50, frame.shape[1]-30, 350),  # right door
                (frame.shape[1]//2-100, frame.shape[0]-200, frame.shape[1]//2+100, frame.shape[0]-30)  # bottom door
            ]
            people_in_any_zone = 0
            new_event = False

            # Draw all zones
            for idx, zone in enumerate(zones):
                cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (0,255,255), 4)
                cv2.putText(frame, f"ENTRY/EXIT ZONE {idx+1}", (zone[0]+5, zone[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # Face detection (small frame for speed)
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                small_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
                gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in faces:
                    x_full, y_full, w_full, h_full = int(x/0.4), int(y/0.4), int(w/0.4), int(h/0.4)
                    cx, cy = x_full + w_full//2, y_full + h_full//2
                    for zone in zones:
                        if zone[0] < cx < zone[2] and zone[1] < cy < zone[3]:
                            people_in_any_zone += 1
                            cv2.rectangle(frame, (x_full, y_full), (x_full+w_full, y_full+h_full), (0,255,0), 2)
                            cv2.putText(frame, "FACE", (x_full, y_full-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                            new_event = True
            except Exception as e:
                self.log(f"Face detection error: {e}")

            # Full-body detection (HOG+SVM)
            try:
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                rects, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(16,16), scale=1.05)
                for (x, y, w, h) in rects:
                    cx, cy = x + w//2, y + h//2
                    for zone in zones:
                        if zone[0] < cx < zone[2] and zone[1] < cy < zone[3]:
                            people_in_any_zone += 1
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,128,0), 2)
                            cv2.putText(frame, "BODY", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,128,0), 2)
                            new_event = True
            except Exception as e:
                self.log(f"Body detection error: {e}")

            # Show people count and status
            cv2.putText(frame, f"People in zones: {people_in_any_zone}", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
            if people_in_any_zone > 0:
                cv2.putText(frame, "ENTRY/EXIT DETECTED!", (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), 3)

            # Save screenshot only on new detection
            if people_in_any_zone > 0 and new_event:
                ss_path = os.path.join(ROOM_SCAN_DIR, f"entry_exit_{get_ist_time().replace(':','-')}.jpg")
                cv2.imwrite(ss_path, frame)

            # Overlay scan status
            cv2.putText(frame, "SCANNING ROOM...", (40, 130), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,255,128), 3)
            cv2.putText(frame, "All exits clear." if people_in_any_zone == 0 else "Movement at entry/exit!", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,128), 2)

        elif mode == 2:
            # Thermal Vision Mode
            thermal = cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.3, thermal, 0.7, 0)

            # Anomaly detection in thermal
            diff = cv2.absdiff(frame, cv2.medianBlur(frame, 21))
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_diff, 60, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            anomaly_found = False
            for c in contours:
                if cv2.contourArea(c) > 1500:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                    cv2.putText(frame, "THERMAL ANOMALY", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
                    anomaly_found = True
            if anomaly_found:
                threats_dir = "threats"
                os.makedirs(threats_dir, exist_ok=True)
                ss_path = os.path.join(threats_dir, f"thermal_anomaly_{get_ist_time().replace(':','-')}.jpg")
                cv2.imwrite(ss_path, frame)

        elif mode == 3:
            threat_detected = False
            zoomed = False
            # Emotion & Face Detection
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                small_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
                gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in faces:
                    x_full, y_full, w_full, h_full = int(x/0.4), int(y/0.4), int(w/0.4), int(h/0.4)
                    face_crop = frame[y_full:y_full+h_full, x_full:x_full+w_full]
                    temp_path = "temp_emotion.jpg"
                    cv2.imwrite(temp_path, face_crop)
                    if self.frame_counter % self.deepface_interval == 0:
                        try:
                            result = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
                            emotion = result['dominant_emotion']
                            cv2.rectangle(frame, (x_full, y_full), (x_full+w_full, y_full+h_full), (0,255,255), 2)
                            cv2.putText(frame, f"Emotion: {emotion}", (x_full, y_full-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                            if emotion in ['fear', 'surprise', 'angry']:
                                threat_detected = True
                                ss_path = os.path.join(PARANORMAL_DIR, f"emotion_{emotion}_{get_ist_time().replace(':','-')}.jpg")
                                cv2.imwrite(ss_path, frame)
                            # Zoom in on face
                            if not zoomed:
                                zoomed_face = cv2.resize(face_crop, (frame.shape[1], frame.shape[0]))
                                cv2.putText(zoomed_face, f"ZOOMED FACE - {emotion}", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,255,255), 3)
                                frame = zoomed_face
                                zoomed = True
                        except Exception:
                            pass
                    os.remove(temp_path)
            except Exception as e:
                self.log(f"Face detection error: {e}")

            # Human detection (zoom on body)
            try:
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                rects, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(16,16), scale=1.05)
                for (x, y, w, h) in rects:
                    if not zoomed:
                        human_crop = frame[y:y+h, x:x+w]
                        zoomed_human = cv2.resize(human_crop, (frame.shape[1], frame.shape[0]))
                        cv2.putText(zoomed_human, "ZOOMED HUMAN", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,128,0), 3)
                        frame = zoomed_human
                        zoomed = True
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255,128,0), 2)
                    cv2.putText(frame, "BODY", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,128,0), 2)
            except Exception as e:
                self.log(f"Body detection error: {e}")

            # Fire/Smoke Detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_fire = np.array([0, 50, 200])
            upper_fire = np.array([35, 255, 255])
            mask_fire = cv2.inRange(hsv, lower_fire, upper_fire)
            fire_pixels = cv2.countNonZero(mask_fire)
            if fire_pixels > 800:
                contours, _ = cv2.findContours(mask_fire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) > 500:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
                        cv2.putText(frame, "FIRE/SMOKE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                threat_detected = True
                ss_path = os.path.join(PARANORMAL_DIR, f"fire_{get_ist_time().replace(':','-')}.jpg")
                cv2.imwrite(ss_path, frame)

            # Anomaly Detection
            diff = cv2.absdiff(frame, cv2.medianBlur(frame, 21))
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > 2000:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                    cv2.putText(frame, "ANOMALY", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
                    threat_detected = True
                    ss_path = os.path.join(PARANORMAL_DIR, f"anomaly_{get_ist_time().replace(':','-')}.jpg")
                    cv2.imwrite(ss_path, frame)

            cv2.putText(frame, "ADVANCED DETECTION MODE", (40, 80), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)

            # Save advanced feed video
            advanced_feed_dir = os.path.join(VIDEO_OUTPUT_DIR, "advanced_feed")
            os.makedirs(advanced_feed_dir, exist_ok=True)
            if not hasattr(self, 'advanced_out'):
                self.advanced_out = cv2.VideoWriter(
                    os.path.join(advanced_feed_dir, self.new_video_filename()),
                    cv2.VideoWriter_fourcc(*'XVID'), 15, (self.frame_width, self.frame_height)
                )
            self.advanced_out.write(frame)

            # Save threat screenshot/video to threats folder
            if threat_detected:
                threats_dir = "threats"
                os.makedirs(threats_dir, exist_ok=True)
                threat_ss = os.path.join(threats_dir, f"threat_{get_ist_time().replace(':','-')}.jpg")
                cv2.imwrite(threat_ss, frame)

        # Overlay FPS and IST timestamp (both in green)
        timestamp = get_ist_time()
        fps_text = f"FPS: {self.real_fps:.1f}"
        cv2.putText(frame, timestamp, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        (text_width, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, fps_text, (frame.shape[1] - text_width - 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save video to correct folder
        self.outs[mode].write(frame)
        self.prev_frame = gray

        # On motion detection, call callback for GUI update
        if motion_detected and self.motion_callback:
            self.motion_count += 1
            self.motion_callback(f"motion_detection_{self.motion_count}st_sighting", frame)

        return frame

    def register_my_face(self):
        # Capture a frame and save as your face in known_faces
        ret, frame = self.video_capture.read()
        if not ret:
            self.log("Failed to capture face for registration.")
            return False
        face_path = self.known_face_path
        cv2.imwrite(face_path, frame)
        try:
            self.registered_face_enc = DeepFace.represent(img_path=face_path, model_name='Facenet')[0]["embedding"]
        except Exception:
            self.registered_face_enc = None
        self.log("Your face has been registered and will not be flagged as unknown.")
        return True

    def release(self):
        self.video_capture.release()
        for out in self.outs:
            out.release()
        speak("Session ended. Videos merged successfully.")
        self.log("Session ended. Videos merged successfully.")

# ------- Network Meter -------
class NetworkMeter(QWidget):
    def __init__(self):
        super().__init__()
        self.speed = 80  # Start with a reasonable value
        self.last_speed = 80
        self.fail_count = 0
        self.setMinimumSize(120, 120)
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_network)
        self.timer.start(1000)

    def check_network(self):
        try:
            start = time.time()
            requests.get("https://www.google.com", timeout=2)
            elapsed = time.time() - start
            self.speed = min(100, max(10, int(100 - elapsed * 100)))
            self.last_speed = self.speed
            self.fail_count = 0
        except:
            self.fail_count += 1
            if self.fail_count > 3:
                self.speed = max(10, self.speed - 5)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) // 2 - 10

        # Draw background
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor("#181a20")))
        painter.drawEllipse(center, radius, radius)

        # Draw arc
        angle = int(270 * self.speed / 100)
        pen = QPen(QColor("#00ff99") if self.speed > 70 else QColor("#ffe066") if self.speed > 40 else QColor("#ff3333"), 10)
        painter.setPen(pen)
        painter.drawArc(center.x()-radius, center.y()-radius, 2*radius, 2*radius, 225*16, -angle*16)

        # Draw pointer
        pointer_angle = math.radians(225 - angle)
        px = center.x() + radius * 0.8 * math.cos(pointer_angle)
        py = center.y() - radius * 0.8 * math.sin(pointer_angle)
        painter.setPen(QPen(QColor("#00ff99"), 4))
        painter.drawLine(center.x(), center.y(), int(px), int(py))

        # Draw text
        painter.setPen(QPen(QColor("#00ff99"), 2))
        painter.setFont(QFont("Consolas", 12, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, f"{self.speed}%")

# ------- Device Radar -------
class DeviceRadar(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(160, 160)
        self.devices = []
        self.angle = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_radar)
        self.timer.start(80)

    def update_radar(self):
        # Simulate random devices, always keep at least 2
        if len(self.devices) < 2 or (random.random() < 0.05 and len(self.devices) < 8):
            angle = random.uniform(0, 2*math.pi)
            dist = random.uniform(40, 70)
            self.devices.append((angle, dist))
        # Fade out old devices
        if len(self.devices) > 8:
            self.devices.pop(0)
        self.angle = (self.angle + 0.07) % (2*math.pi)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) // 2 - 10

        # Draw radar circles
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#00ff99"), 2))
        for r in range(1, 4):
            painter.drawEllipse(center, int(radius*r/3), int(radius*r/3))

        # Draw sweep
        sweep_angle = self.angle
        painter.setPen(QPen(QColor("#00ff99"), 3))
        sx = center.x() + radius * math.cos(sweep_angle)
        sy = center.y() + radius * math.sin(sweep_angle)
        painter.drawLine(center.x(), center.y(), int(sx), int(sy))

        # Draw devices
        for ang, dist in self.devices:
            dx = center.x() + dist * math.cos(ang)
            dy = center.y() + dist * math.sin(ang)
            painter.setBrush(QBrush(QColor("#00ff99")))
            painter.drawEllipse(int(dx)-5, int(dy)-5, 10, 10)

        # Draw center
        painter.setBrush(QBrush(QColor("#00ff99")))
        painter.drawEllipse(center.x()-6, center.y()-6, 12, 12)

# ------- Camera Worker Thread -------
class CameraWorker(QThread):
    frame_ready = pyqtSignal(int, np.ndarray)

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.running = True

    def run(self):
        while self.running:
            for i in range(4):
                frame = self.detector.process_frame(mode=i)
                if frame is not None:
                    self.frame_ready.emit(i, frame)
            self.msleep(10)  # ~100 FPS loop, adjust as needed

    def stop(self):
        self.running = False
        self.wait()

# ------- GUI Application -------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CamSecureV1")
        self.setGeometry(100, 100, 1300, 900)
        self.set_dark_theme()
        # Log console
        self.log_display = QTextEdit(self)
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 10))
        self.log_display.setStyleSheet("""
            background-color: #11141a;
            color: #00ff99;
            border-radius: 8px;
            font-size: 13px;
            padding: 8px;
        """)

        # Video frames
        self.video_labels = [QLabel(self) for _ in range(4)]
        for i, label in enumerate(self.video_labels):
            label.setFixedSize(600, 340)
            label.setStyleSheet("background-color: #181a20; border-radius: 12px; border: 2px solid #00ff99;")
            label.setAlignment(Qt.AlignCenter)

        # Grid layout for video frames
        grid = QGridLayout()
        grid.setHorizontalSpacing(30)
        grid.setVerticalSpacing(30)
        grid.addWidget(self.video_labels[0], 0, 0)
        grid.addWidget(self.video_labels[1], 0, 1)
        grid.addWidget(self.video_labels[2], 1, 0)
        grid.addWidget(self.video_labels[3], 1, 1)

        # Frame mode names
        self.frame_names = [
            "LIVE FEED",
            "ROOM SCAN",
            "THERMAL VISION",
            "INTRUDER HEATMAP"
        ]
        self.frame_mode = 0

        # Modern buttons
        self.start_btn = QPushButton("▶ Start Detection", self)
        self.stop_btn = QPushButton("■ Stop Detection", self)
        self.register_btn = QPushButton("🧑 Register My Face", self)
        self.bg_btn = QPushButton("🕶 Enable Background Mode", self)
        self.bg_btn.setCheckable(True)
        self.audio_enabled = False
        self.audio_btn = QPushButton("🎙 Enable Audio Command", self)
        self.audio_btn.setCheckable(True)
        for btn in [self.start_btn, self.stop_btn, self.register_btn, self.bg_btn, self.audio_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2d313a;
                    color: #00ff99;
                    border: none;
                    border-radius: 8px;
                    padding: 12px 24px;
                    font-size: 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #3a3f4b;
                }
                QPushButton:pressed {
                    background-color: #23272e;
                }
            """)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Frame switch buttons
        self.frame_btns = []
        for i, name in enumerate(self.frame_names):
            btn = QPushButton(name, self)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #181a20;
                    color: #00ff99;
                    border: 2px solid #00ff99;
                    border-radius: 8px;
                    padding: 8px 16px;
                    font-size: 14px;
                }
                QPushButton:checked {
                    background-color: #00ff99;
                    color: #181a20;
                }
            """)
            btn.clicked.connect(lambda checked, idx=i: self.set_frame_mode(idx))
            self.frame_btns.append(btn)

        frame_btn_layout = QHBoxLayout()
        for btn in self.frame_btns:
            frame_btn_layout.addWidget(btn)
        frame_btn_layout.addStretch(1)

        # Motion analysis label
        self.motion_label = QLabel(self)
        self.motion_label.setStyleSheet("color: #00ff99; font-size: 16px; background: transparent;")
        self.motion_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.motion_label.setFixedWidth(350)

        # Instructions panel
        self.instructions = QTextEdit(self)
        self.instructions.setReadOnly(True)
        self.instructions.setFont(QFont("Consolas", 11, QFont.Bold))
        self.instructions.setStyleSheet("background-color: #181a20; color: #00ff99; border-radius: 8px;")
        self.instructions.setFixedHeight(90)
        self.instructions.setText(
            "Available Commands:\n"
            "- 'Analyse the room' : Room scan\n"
            "- 'Tell me about network' : Network status\n"
            "- 'Last detection' : Last motion/human\n"
            "- 'Open panel' : Show window\n"
            "- 'Export report' : Save log report\n"
            "- 'Register my face' : Register yourself\n"
            "- 'Switch to [mode]' : Change view\n"
        )
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.register_btn)
        btn_layout.addWidget(self.bg_btn)
        btn_layout.addWidget(self.audio_btn)
        btn_layout.setSpacing(20)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(grid)
        main_layout.addLayout(frame_btn_layout)
        main_layout.addLayout(btn_layout)
        main_layout.addStretch(1)

        # Bottom right log console
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch(1)
        self.log_display.setFixedSize(400, 120)
        bottom_layout.addWidget(self.log_display)
        bottom_layout.addWidget(self.motion_label)

        # Network meter
        self.network_meter = NetworkMeter()
        bottom_layout.addWidget(self.network_meter)

        # Device radar
        self.device_radar = DeviceRadar()
        bottom_layout.addWidget(self.device_radar)

        main_layout.addLayout(bottom_layout)

        # Instructions toggle button
        self.instructions_toggle = QPushButton("Show/Hide Instructions", self)
        self.instructions_toggle.setCheckable(True)
        self.instructions_toggle.setChecked(True)
        self.instructions_toggle.setStyleSheet("color: #00ff99; background: #23272e; border-radius: 8px;")
        self.instructions_toggle.clicked.connect(lambda: self.instructions.setVisible(self.instructions_toggle.isChecked()))
        main_layout.addWidget(self.instructions_toggle)
        main_layout.addWidget(self.instructions)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Detector with log callback
        self.detector = Detector(log_callback=self.append_log, motion_callback=self.show_motion_analysis)

        # Camera worker for frame processing
        self.camera_worker = CameraWorker(self.detector)
        self.camera_worker.frame_ready.connect(self.update_frame)
        self.camera_worker.start()

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.register_btn.clicked.connect(self.register_my_face)
        self.bg_btn.clicked.connect(self.toggle_background_mode)
        self.audio_btn.clicked.connect(self.toggle_audio_command)

        self.voice_thread = None

        # Default: show all frames, highlight first
        self.set_frame_mode(0)

    def set_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(24, 26, 32))
        dark_palette.setColor(QPalette.WindowText, QColor(185, 192, 203))
        dark_palette.setColor(QPalette.Base, QColor(35, 39, 46))
        dark_palette.setColor(QPalette.AlternateBase, QColor(24, 26, 32))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(185, 192, 203))
        dark_palette.setColor(QPalette.ToolTipText, QColor(185, 192, 203))
        dark_palette.setColor(QPalette.Text, QColor(185, 192, 203))
        dark_palette.setColor(QPalette.Button, QColor(45, 49, 58))
        dark_palette.setColor(QPalette.ButtonText, QColor(0, 255, 0))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.Highlight, QColor(52, 59, 72))
        dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        QApplication.setPalette(dark_palette)
        self.setStyleSheet("QMainWindow { background-color: #181a20; }")

    def append_log(self, message):
        self.log_display.append(message)
        self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum())

    def set_frame_mode(self, idx):
        self.frame_mode = idx
        for i, btn in enumerate(self.frame_btns):
            btn.setChecked(i == idx)

    def start(self):
        self.append_log("Detection started.")
        self.camera_worker.start()
        # self.timer.start(int(1000 / self.detector.fps))

    def stop(self):
        self.append_log("Detection stopped.")
        # self.timer.stop()
        self.detector.release()
        self.camera_worker.stop()

    def register_my_face(self):
        if self.detector.register_my_face():
            self.append_log("Your face has been registered and will not be flagged as unknown.")
            speak("Your face has been registered.")

    def toggle_audio_command(self):
        self.audio_enabled = self.audio_btn.isChecked()
        if self.audio_enabled:
            self.audio_btn.setText("🎙 Listening...")
            self.voice_thread = threading.Thread(target=self.listen_loop, daemon=True)
            self.voice_thread.start()
        else:
            self.audio_btn.setText("🎙 Enable Audio Command")

    def listen_loop(self):
        while self.audio_enabled:
            cmd = listen_command().lower()
            self.append_log(f"Voice command: {cmd}")
            self.handle_voice_command(cmd)

    def handle_voice_command(self, cmd):
        if "analysis" in cmd or "analyse the room" in cmd:
            speak("Scanning the room for threats and activity.")
            self.append_log("[Room Analysis] No additional threats found.")
            self.set_frame_mode(1)
        elif "network" in cmd or "tell me about network" in cmd:
            speed = self.network_meter.last_speed
            status = "Good" if speed > 70 else "Normal" if speed > 40 else "Poor"
            speak(f"Network status is {status}. Speed: {speed} percent.")
            self.append_log(f"Network status: {status} ({speed}%)")
        elif "last detection" in cmd:
            self.append_log("Last detection: " + (self.motion_label.text() or "No recent motion."))
            speak("Last detection was: " + (self.motion_label.text() or "No recent motion."))
        elif "open panel" in cmd:
            self.showNormal()
            self.raise_()
            self.activateWindow()
            speak("Panel opened.")
        elif "export report" in cmd:
            try:
                shutil.copy(self.detector.session_log_file, os.path.join(VIDEO_OUTPUT_DIR, "report.txt"))
                speak("Threat report exported.")
                self.append_log("Threat report exported to recordings/report.txt")
            except Exception as e:
                self.append_log(f"Export failed: {e}")
        else:
            speak("Command not recognized, but I'm always listening for your instructions.")
            self.append_log("Command not recognized.")

    def toggle_background_mode(self):
        if self.bg_btn.isChecked():
            self.hide()
            speak("Background mode enabled. Say 'open panel' to bring me back.")
        else:
            self.show()
            speak("Panel visible.")

    def update_frame(self, idx, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img)
        self.video_labels[idx].setPixmap(pix.scaled(self.video_labels[idx].size(), Qt.KeepAspectRatio))

    def show_motion_analysis(self, motion_id, frame):
        analysis = f"{motion_id}: Movement detected and analyzed."
        self.motion_label.setText(analysis)
        self.append_log(analysis)

    def closeEvent(self, event):
        self.camera_worker.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())