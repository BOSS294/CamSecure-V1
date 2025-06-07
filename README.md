**CamSecure V1**

A comprehensive, multi-mode video surveillance and threat detection system built with Python, OpenCV, DeepFace, and PyQt5. CamSecure V1 supports live feed monitoring, room scanning, thermal vision, and advanced intruder heatmap analysis, with audio prompts and logging.

---

## Table of Contents

1. [Features](#features)
2. [Architecture & Modules](#architecture--modules)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Directory Structure](#directory-structure)
8. [Code Documentation](#code-documentation)
9. [Voice Commands](#voice-commands)
10. [License](#license)
11. [Repository](#repository)

---

## Features

* **Multi-mode Video Streams**:

  * Live Feed with motion detection overlay
  * Room Scan with entry/exit zone monitoring
  * Thermal Vision with anomaly detection
  * Intruder Heatmap with emotion, human, fire/smoke, and anomaly detection
* **Face Recognition**: Register a known face and skip false alarms
* **Automated Logging**: Timestamped logs of events and alerts
* **Audio Feedback & Voice Control**: Text-to-speech prompts and speech recognition commands
* **Network Meter & Device Radar**: Real-time network health and simulated radar display
* **Background Mode**: Hide GUI and continue monitoring
* **PyQt5 GUI**: Dark-themed interface with real-time grid display, controls, and log console

---

## Architecture & Modules

* **app.py / main.py**: Entry point; initializes PyQt5 `MainWindow` and starts the application loop.
* **Detector**: Core class handling camera capture, frame processing, motion detection, face recognition, thermal and advanced threat analysis.
* **CameraWorker (QThread)**: Continuously fetches and processes frames across modes in a separate thread.
* **MainWindow**: PyQt5 GUI, sets up layouts, buttons, panels, video frames, log console, and integrates `Detector` and `CameraWorker`.
* **NetworkMeter & DeviceRadar**: Custom QWidget classes for network speed gauge and device sweep radar.

---

## Prerequisites

* Python 3.8+
* OpenCV `cv2`
* DeepFace
* PyQt5
* pyttsx3
* SpeechRecognition
* NumPy
* SciPy
* Requests

Install dependencies via pip:

```bash
pip install opencv-python deepface pyqt5 pyttsx3 SpeechRecognition numpy scipy requests
```

Additionally, ensure you have:

* A working microphone for voice control
* A webcam for video capture

---

## Installation

1. Clone the repository:

   ```bash
   ```

git clone [https://github.com/BOSS294/CamSecure-V1.git](https://github.com/BOSS294/CamSecure-V1.git)
cd CamSecure-V1

````

2. Install the prerequisites as shown above.

3. Create necessary directories (if not auto-generated):

```bash
mkdir recordings logs screenshots known_faces recordings/live_feed recordings/room_scan recordings/thermal recordings/paranormal
mkdir recordings/advanced_feed threats
````

---

## Configuration

* **VIDEO\_OUTPUT\_DIR**: Base directory for recording videos. Default: `recordings`
* **LOG\_DIR**: Directory for session logs. Default: `logs`
* **SCREENSHOT\_DIR**: Directory for saved screenshots. Default: `screenshots`
* **KNOWN\_FACES\_DIR**: Directory to store registered face images.
* **DeepFace Threshold**: Cosine similarity threshold (0.7) for face matching.
* **Frame Modes**: Mode indices 0-3 correspond to live, room, thermal, and advanced respectively.

Edit constants at the top of `main.py` to modify paths or thresholds.

---

## Usage

```bash
python main.py
```

* Click **▶ Start Detection** to begin processing frames.
* **■ Stop Detection** stops recording and merges videos.
* **🧑 Register My Face** captures and registers your face so it isn't flagged as unknown.
* **🕶 Background Mode** hides the GUI; say "open panel" to restore.
* **🎙 Audio Command** toggles voice control for commands.
* Use the mode buttons (Live Feed, Room Scan, Thermal Vision, Intruder Heatmap) to switch view.

---

## Directory Structure

```
CamSecure-V1/
├── main.py
├── recordings/
│   ├── live_feed/
│   ├── room_scan/
│   ├── thermal/
│   ├── paranormal/
│   └── advanced_feed/
├── logs/
├── screenshots/
└── known_faces/
```

---

## Code Documentation

### Detector Class

* **`__init__`**: Initializes video capture, output writers, and loads registered face embedding.
* **`process_frame(mode)`**: Processes a single frame based on the selected mode:

  * **Mode 0**: Motion detection with bounding boxes and logging
  * **Mode 1**: Room scan with entry/exit zones, face/body detection, and screenshots
  * **Mode 2**: Thermal vision using color maps and anomaly detection
  * **Mode 3**: Advanced detection (emotion, human, fire/smoke, anomaly) with zoom and threat logging
* **`is_registered_face(face_img)`**: Compares detected face to registered face embedding.
* **`register_my_face()`**: Captures and saves a face image to `known_faces`.
* **`release()`**: Cleans up capture and writers; speaks session end and log merge.

### CameraWorker (QThread)

* Runs `Detector.process_frame` in a loop for each mode and emits frames.

### MainWindow Class

* Sets up PyQt5 dark-themed GUI, video labels, control buttons, and log console.
* Connects button events to start/stop detection, register face, toggle modes, and voice commands.
* **`update_frame`**: Receives frames from `CameraWorker` and displays them.
* **`show_motion_analysis`**: Updates motion summary label and log.

---

## Voice Commands

* **"Analyse the room"**: Switch to Room Scan mode
* **"Tell me about network"**: Reports network status
* **"Last detection"**: Reads last detection summary
* **"Open panel"**: Restores GUI in background mode
* **"Export report"**: Copies log report to `recordings/report.txt`
* **"Register my face"**: Captures and registers face

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Repository

For full source code, issues, and contribution guidelines, visit:

[https://github.com/BOSS294/CamSecure-V1](https://github.com/BOSS294/CamSecure-V1)
