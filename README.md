# AI Vision Assistant

A real-time webcam-based AI system that detects objects and describes scenes using advanced deep learning models. Features multi-threaded architecture for optimal CPU performance, object tracking, analytics logging, and text-to-speech output.

## Features

✅ Real-time Object Detection (YOLOv8 nano, CPU-optimized)  
✅ Scene Captioning (BLIP model)  
✅ Text-to-Speech (pyttsx3)  
✅ Object Tracking across frames  
✅ FPS Counter  
✅ Analytics Logging (detections + stats)  
✅ YAML Configuration system  
✅ Modular, FastAPI-ready architecture  

## Project Structure

```
OBJECT DETECTION/
├── main.py
├── config.py
├── config.yaml
├── vision.py
├── caption.py
├── voice.py
├── tracker.py
├── analytics.py
├── requirements.txt
├── logs/
└── README.md
```

## Quick Start

### 1) Install dependencies
```bash
cd "d:\PROJECTS\OBJECT DETECTION"
pip install -r requirements.txt
```

### 2) Run
```bash
python main.py
```

## Web App (Session-Based)

### Start the web server
```bash
uvicorn webapp:app --host 127.0.0.1 --port 8000
```

Open your browser at http://127.0.0.1:8000

### Web Features
- Start/Stop camera per session
- Session-specific object list
- Click object to get AI description

## Optional: OpenRouter AI Descriptions

Set your API key as an environment variable (do not hardcode it):

Windows PowerShell:
```bash
$env:OPENROUTER_API_KEY="YOUR_KEY_HERE"
```

Optional:
```bash
$env:OPENROUTER_MODEL="deepseek/deepseek-r1-0528:free"
$env:OPENROUTER_SITE_URL="http://127.0.0.1:8000"
$env:OPENROUTER_SITE_TITLE="EchoVision AI"
```

## Optional: MongoDB Session Storage

Set your MongoDB connection string in an environment variable (do not hardcode it):

Windows PowerShell:
```bash
$env:MONGODB_URI="YOUR_MONGODB_URI"
```

Optional database settings:
```bash
$env:MONGODB_DB="echovision"
$env:MONGODB_COLLECTION="sessions"
```

### Controls
- SPACE = Pause/Resume
- Q = Quit
- R = Reset tracking
- S = Save statistics

## Customize
Edit [config.yaml](config.yaml) to tune model, speed, logging, speech, and display.

## Notes
- First run downloads YOLO and BLIP models (~2GB)
- Optimized for CPU/integrated GPU
- To boost speed, increase `performance.frame_skip` in config

---
Enjoy! 🚀
