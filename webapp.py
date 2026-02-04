"""
Web Application for AI Vision Assistant
FastAPI-based web interface with session-based camera control
"""

import threading
import time
import os
from datetime import datetime
from typing import Dict, List, Optional

import cv2
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from config import get_config
from vision import ObjectDetector
from caption import ImageCaptioner
from voice import VoiceAssistant
from openrouter_client import OpenRouterClient

try:
    from pymongo import MongoClient
    from pymongo.server_api import ServerApi
except Exception:
    MongoClient = None
    ServerApi = None


class MongoSessionStore:
    """Optional MongoDB session storage"""

    def __init__(self):
        self.uri = os.getenv("MONGODB_URI", "")
        self.db_name = os.getenv("MONGODB_DB", "echovision")
        self.collection_name = os.getenv("MONGODB_COLLECTION", "sessions")
        self.enabled = bool(self.uri and MongoClient)

        self.client = None
        self.collection = None

        if self.enabled:
            self.client = MongoClient(self.uri, server_api=ServerApi("1"))
            self.collection = self.client[self.db_name][self.collection_name]

    def create_session(self, session_id: str, session_number: int) -> None:
        if not self.enabled:
            return
        self.collection.insert_one({
            "session_id": session_id,
            "session_number": session_number,
            "started_at": datetime.utcnow().isoformat(),
            "ended_at": None,
            "objects": [],
            "descriptions": {},
        })

    def update_snapshot(self, session_id: str, objects: List[Dict], descriptions: Dict) -> None:
        if not self.enabled:
            return
        self.collection.update_one(
            {"session_id": session_id},
            {"$set": {
                "objects": objects,
                "descriptions": descriptions,
                "last_updated": datetime.utcnow().isoformat(),
            }}
        )

    def end_session(self, session_id: str, objects: List[Dict], descriptions: Dict) -> None:
        if not self.enabled:
            return
        self.collection.update_one(
            {"session_id": session_id},
            {"$set": {
                "objects": objects,
                "descriptions": descriptions,
                "ended_at": datetime.utcnow().isoformat(),
            }}
        )


class CameraSessionManager:
    """Session-based camera controller"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.detector = ObjectDetector(
            model_name=self.config.get("model.yolo_model"),
            confidence_threshold=self.config.get("model.confidence_threshold")
        )
        self.captioner = ImageCaptioner(
            model_name=self.config.get("model.caption_model")
        )
        self.voice = VoiceAssistant(
            rate=self.config.get("speech.speech_rate"),
            volume=self.config.get("speech.volume")
        )
        self.llm = OpenRouterClient()
        self.store = MongoSessionStore()

        self.session_active = False
        self.camera_active = False
        self.session_id = None
        self.session_number = 0
        self.session_history = []
        self.described_objects = set()  # Track which objects have been auto-described
        self.capture_thread = None
        self.cap = None

        self.lock = threading.Lock()
        self.last_frame = None
        self.last_detections = {"boxes": [], "class_names": [], "confidences": [], "class_ids": []}
        self.last_caption = ""
        self.objects = {}
        self.descriptions = {}
        self.last_caption_time = 0.0
        self.fps_clock = cv2.getTickFrequency()
        self.frame_times = []

    def start_session(self) -> str:
        if self.session_active:
            return self.session_id

        self.session_number += 1
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.objects = {}
        self.descriptions = {}
        self.last_caption = ""
        self.last_frame = None
        self.last_detections = {"boxes": [], "class_names": [], "confidences": [], "class_ids": []}
        self.last_caption_time = 0.0
        self.frame_times = []

        self.session_active = True
        self.camera_active = False

        self.session_history.append({
            "number": self.session_number,
            "session_id": self.session_id,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": None,
        })

        if self.store.enabled:
            self.store.create_session(self.session_id, self.session_number)

        if self.capture_thread is None or not self.capture_thread.is_alive():
            self.capture_thread = threading.Thread(target=self._run_loop, daemon=True)
            self.capture_thread.start()

        return self.session_id

    def start_camera(self) -> str:
        if not self.session_active:
            self.start_session()

        if self.camera_active:
            return self.session_id

        self.cap = cv2.VideoCapture(self.config.get("webcam.camera_index"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get("webcam.frame_width"))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get("webcam.frame_height"))
        self.cap.set(cv2.CAP_PROP_FPS, self.config.get("webcam.fps"))

        if not self.cap.isOpened():
            self.cap = None
            raise RuntimeError("Failed to open webcam")

        self.camera_active = True
        return self.session_id

    def stop_camera(self) -> None:
        if not self.camera_active:
            return

        self.camera_active = False
        if self.cap:
            self.cap.release()
        self.cap = None

        if self.session_id:
            self.store.update_snapshot(self.session_id, self.get_objects(), self.descriptions)

    def end_session(self) -> None:
        if not self.session_active:
            return

        self.session_active = False
        self.camera_active = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        self.cap = None

        if self.session_id:
            for session in self.session_history:
                if session["session_id"] == self.session_id:
                    session["ended_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    break

            self.store.end_session(self.session_id, self.get_objects(), self.descriptions)
        self.session_id = None
        self.described_objects.clear()  # Clear auto-description tracking

    def _filter_detections(self, detections: Dict) -> Dict:
        if not self.config.get("detection.only_show_detected_classes"):
            return detections

        allowed = set(self.config.get("detection.classes_to_detect") or [])
        if not allowed:
            return detections

        filtered = {"boxes": [], "class_names": [], "confidences": [], "class_ids": []}
        for box, name, conf, cid in zip(
            detections["boxes"],
            detections["class_names"],
            detections["confidences"],
            detections["class_ids"]
        ):
            if name in allowed:
                filtered["boxes"].append(box)
                filtered["class_names"].append(name)
                filtered["confidences"].append(conf)
                filtered["class_ids"].append(cid)
        return filtered

    def _update_objects(self, class_names: List[str]) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        for name in class_names:
            if name not in self.objects:
                self.objects[name] = {"count": 0, "last_seen": now}
            self.objects[name]["count"] += 1
            self.objects[name]["last_seen"] = now

    def _draw_caption(self, frame, caption: str) -> None:
        if not caption:
            return
        max_width = 80
        words = caption.split()
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)
            if len(" ".join(current_line)) > max_width:
                lines.append(" ".join(current_line[:-1]))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        y_offset = frame.shape[0] - 20 - (len(lines) * 25)
        for i, line in enumerate(lines):
            text_y = y_offset + (i * 25)
            cv2.rectangle(
                frame,
                (8, text_y - 20),
                (frame.shape[1] - 8, text_y + 6),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                frame,
                line,
                (12, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    def _draw_fps(self, frame, fps: float) -> None:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    def _run_loop(self) -> None:
        frame_skip = max(1, int(self.config.get("performance.frame_skip")))
        iou_threshold = self.config.get("model.iou_threshold")
        caption_interval = 2.0
        frame_counter = 0

        while self.session_active:
            if not self.camera_active or self.cap is None:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue

            frame_start = cv2.getTickCount()
            detections = self.detector.detect(frame, iou_threshold=iou_threshold)
            detections = self._filter_detections(detections)

            self._update_objects(detections["class_names"])            
            # Auto-generate descriptions for new objects
            self.auto_describe_objects()
            now = time.time()
            if now - self.last_caption_time >= caption_interval:
                caption = self.captioner.generate_caption(frame, max_length=50, num_beams=1)
                summary = self.detector.get_detection_summary(detections)
                self.last_caption = ImageCaptioner.enhance_caption(caption, summary)
                self.last_caption_time = now

            display_frame = self.detector.draw_detections(
                frame,
                detections,
                box_color=tuple(self.config.get("display.box_color")),
                text_color=tuple(self.config.get("display.text_color")),
                thickness=self.config.get("display.box_thickness"),
                font_scale=self.config.get("display.font_size"),
                show_confidence=self.config.get("display.show_confidence")
            )

            frame_end = cv2.getTickCount()
            frame_time = (frame_end - frame_start) / self.fps_clock
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            fps = len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0

            if self.config.get("display.show_fps"):
                self._draw_fps(display_frame, fps)
            if self.config.get("display.show_description"):
                self._draw_caption(display_frame, self.last_caption)

            with self.lock:
                self.last_frame = display_frame
                self.last_detections = detections

        if self.cap:
            self.cap.release()

    def get_frame(self):
        with self.lock:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()

    def get_objects(self) -> List[Dict]:
        with self.lock:
            return [
                {"name": name, "count": data["count"], "last_seen": data["last_seen"]}
                for name, data in sorted(self.objects.items())
            ]

    def get_caption(self) -> str:
        with self.lock:
            return self.last_caption

    def get_sessions(self) -> List[Dict]:
        return list(self.session_history)

    def auto_describe_objects(self):
        """Automatically generate descriptions for newly detected objects"""
        if not self.session_active:
            return
        
        # Get current unique object classes
        current_objects = set(self.objects.keys())
        
        # Find objects that haven't been auto-described yet
        new_objects = current_objects - self.described_objects
        
        # Generate descriptions for new objects
        for obj_class in new_objects:
            self.logger.info(f"Auto-generating description for {obj_class}")
            # Trigger description generation (will be async in background)
            threading.Thread(target=self.describe_object, args=(obj_class,), daemon=True).start()
            self.described_objects.add(obj_class)

    def describe_object(self, object_name: str) -> str:
        with self.lock:
            frame = self.last_frame.copy() if self.last_frame is not None else None
            detections = self.last_detections

        if frame is None:
            llm_description = self.llm.describe_object(object_name)
            if llm_description:
                self.descriptions[object_name] = llm_description
                if self.session_id:
                    self.store.update_snapshot(self.session_id, self.get_objects(), self.descriptions)
                if self.config.get("speech.enable_tts"):
                    self.voice.speak_if_changed(llm_description)
                return llm_description
            return "No active frame available. Start the camera at least once."

        crop = None
        for box, name in zip(detections["boxes"], detections["class_names"]):
            if name == object_name:
                x1, y1, x2, y2 = box
                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)
                crop = frame[y1:y2, x1:x2]
                break

        prompt = f"a photo of a {object_name}"
        if crop is None:
            crop = frame

        llm_description = self.llm.describe_object(object_name)
        if llm_description:
            self.descriptions[object_name] = llm_description
            if self.session_id:
                self.store.update_snapshot(self.session_id, self.get_objects(), self.descriptions)
            if self.config.get("speech.enable_tts"):
                self.voice.speak_if_changed(llm_description)
            return llm_description

        caption = self.captioner.generate_caption_with_context(crop, context=prompt, max_length=40)
        self.descriptions[object_name] = caption
        if self.session_id:
            self.store.update_snapshot(self.session_id, self.get_objects(), self.descriptions)
        if self.config.get("speech.enable_tts"):
            self.voice.speak_if_changed(caption)
        return caption


app = FastAPI(title="AI Vision Assistant Webapp")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize manager immediately to avoid startup event issues
manager = CameraSessionManager()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/session/start")
def start_session():
    session_id = manager.start_session()
    return {"session_active": True, "camera_active": manager.camera_active, "session_id": session_id}


@app.post("/session/end")
def end_session():
    manager.end_session()
    return {"session_active": False, "camera_active": False, "session_id": None}


@app.post("/camera/start")
def start_camera():
    session_id = manager.start_camera()
    return {"session_active": manager.session_active, "camera_active": True, "session_id": session_id}


@app.post("/camera/stop")
def stop_camera():
    manager.stop_camera()
    return {"session_active": manager.session_active, "camera_active": False, "session_id": manager.session_id}


@app.get("/status")
def status():
    return {
        "session_active": manager.session_active,
        "camera_active": manager.camera_active,
        "session_id": manager.session_id,
        "caption": manager.get_caption()
    }


@app.get("/objects")
def objects():
    return {"objects": manager.get_objects()}


@app.get("/sessions")
def sessions():
    return {"sessions": manager.get_sessions()}

@app.get("/session", response_class=HTMLResponse)
def session_page(request: Request):
    """Session history page"""
    return templates.TemplateResponse("session.html", {
        "request": request,
        "sessions": manager.get_sessions()
    })


@app.get("/objects-page", response_class=HTMLResponse)
def objects_page(request: Request):
    """Objects page showing all detected objects with descriptions"""
    objects = manager.get_objects()
    # Add descriptions to objects
    for obj in objects:
        obj_name = obj.get('class', '')
        obj['description'] = manager.descriptions.get(obj_name, '')
    
    return templates.TemplateResponse("objects.html", {
        "request": request,
        "objects": objects
    })

@app.get("/describe")
def describe(object_name: str = Query(..., alias="object")):
    description = manager.describe_object(object_name)
    return {"object": object_name, "description": description}


@app.get("/stream")
def stream():
    def generate():
        while True:
            if not manager.camera_active:
                time.sleep(0.1)
                continue
            frame = manager.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/health")
def health():
    return {"status": "ok"}
