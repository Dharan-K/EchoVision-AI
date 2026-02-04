"""
AI Vision Assistant Main Application
Real-time webcam object detection with AI descriptions and text-to-speech
"""

import cv2
import time
import threading
import queue
import sys

from config import get_config
from vision import ObjectDetector
from caption import ImageCaptioner
from voice import VoiceAssistant
from tracker import ObjectTracker
from analytics import AnalyticsLogger


class VisionAssistant:
    """Main application controller with multi-threading"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.config.print_config()

        print("\n" + "="*60)
        print("INITIALIZING COMPONENTS")
        print("="*60)

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

        self.tracker = ObjectTracker()

        self.analytics = AnalyticsLogger(
            log_file=self.config.get("logging.log_file"),
            stats_file=self.config.get("logging.stats_file")
        )

        self.cap = cv2.VideoCapture(self.config.get("webcam.camera_index"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get("webcam.frame_width"))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get("webcam.frame_height"))
        self.cap.set(cv2.CAP_PROP_FPS, self.config.get("webcam.fps"))

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open webcam")

        print("✓ All components initialized")
        print("="*60 + "\n")

        self.running = True
        self.paused = False
        self.frame_count = 0
        self.last_caption = ""
        self.frame_skip_counter = 0

        self.frame_queue = queue.Queue(maxsize=2)
        self.detection_queue = queue.Queue(maxsize=1)
        self.caption_queue = queue.Queue(maxsize=1)

        self.fps_clock = cv2.getTickFrequency()
        self.frame_times = []
        self.current_fps = 0

    def capture_frames(self) -> None:
        print("Starting frame capture thread...")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam")
                break

            self.frame_skip_counter += 1
            if self.frame_skip_counter % self.config.get("performance.frame_skip") != 0:
                continue

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def detect_objects(self) -> None:
        print("Starting object detection thread...")
        iou_threshold = self.config.get("model.iou_threshold")

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                detections = self.detector.detect(frame, iou_threshold=iou_threshold)
                tracked_objects = self.tracker.update(detections, frame.shape[:2])

                try:
                    self.detection_queue.put_nowait({
                        "frame": frame,
                        "detections": detections,
                        "tracked_objects": tracked_objects
                    })
                except queue.Full:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in detection thread: {e}")

    def generate_captions(self) -> None:
        print("Starting caption generation thread...")

        while self.running:
            try:
                data = self.detection_queue.get(timeout=2)
                frame = data["frame"]
                detections = data["detections"]

                caption = self.captioner.generate_caption(
                    frame,
                    max_length=50,
                    num_beams=1
                )

                detection_summary = self.detector.get_detection_summary(detections)
                enhanced_caption = ImageCaptioner.enhance_caption(caption, detection_summary)

                try:
                    self.caption_queue.put_nowait(enhanced_caption)
                except queue.Full:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in caption thread: {e}")

    def display_frame(self, frame, detections, caption, fps):
        display_frame = self.detector.draw_detections(
            frame,
            detections,
            box_color=tuple(self.config.get("display.box_color")),
            text_color=tuple(self.config.get("display.text_color")),
            thickness=self.config.get("display.box_thickness"),
            font_scale=self.config.get("display.font_size"),
            show_confidence=self.config.get("display.show_confidence")
        )

        if self.config.get("display.show_fps"):
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        if self.config.get("display.show_description") and caption:
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

            y_offset = display_frame.shape[0] - 20 - (len(lines) * 25)

            for i, line in enumerate(lines):
                text_y = y_offset + (i * 25)
                cv2.rectangle(
                    display_frame,
                    (5, text_y - 20),
                    (display_frame.shape[1] - 5, text_y + 5),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(
                    display_frame,
                    line,
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        if self.paused:
            cv2.putText(
                display_frame,
                "PAUSED",
                (display_frame.shape[1]//2 - 50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                2
            )

        return display_frame

    def run(self) -> None:
        print("Starting AI Vision Assistant...")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  Q: Quit")
        print("  R: Reset tracking")
        print("  S: Save statistics")
        print("="*60 + "\n")

        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        detection_thread = threading.Thread(target=self.detect_objects, daemon=True)
        caption_thread = threading.Thread(target=self.generate_captions, daemon=True)

        capture_thread.start()
        detection_thread.start()
        caption_thread.start()

        frame_display = None
        current_detections = {"boxes": [], "class_names": [], "confidences": []}

        try:
            while self.running:
                frame_start = cv2.getTickCount()

                try:
                    detection_data = self.detection_queue.get_nowait()
                    frame_display = detection_data["frame"]
                    current_detections = detection_data["detections"]
                except queue.Empty:
                    pass

                try:
                    new_caption = self.caption_queue.get_nowait()
                    self.last_caption = new_caption

                    if self.config.get("speech.enable_tts"):
                        if self.config.get("speech.speak_on_change_only"):
                            self.voice.speak_if_changed(self.last_caption)
                        else:
                            self.voice.speak(self.last_caption)
                except queue.Empty:
                    pass

                if self.config.get("logging.enable_logging"):
                    if self.frame_count % self.config.get("logging.log_interval") == 0:
                        self.analytics.log_detection(self.frame_count, current_detections)

                if frame_display is not None:
                    frame_end = cv2.getTickCount()
                    frame_time = (frame_end - frame_start) / self.fps_clock
                    self.frame_times.append(frame_time)

                    if len(self.frame_times) > 30:
                        self.frame_times.pop(0)

                    self.current_fps = len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0
                    self.analytics.log_fps(self.current_fps)
                    self.analytics.log_processing_time(frame_time)

                    display_frame = self.display_frame(
                        frame_display,
                        current_detections,
                        self.last_caption,
                        self.current_fps
                    )

                    cv2.imshow("AI Vision Assistant", display_frame)
                    self.frame_count += 1

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nQuitting application...")
                    self.running = False
                elif key == ord(' '):
                    self.paused = not self.paused
                    print(f"{'Paused' if self.paused else 'Resumed'}")
                elif key == ord('r'):
                    self.tracker.reset()
                    print("Tracking reset")
                elif key == ord('s'):
                    self.analytics.save_statistics()
                    print("Statistics saved")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        print("\nCleaning up...")
        self.running = False
        time.sleep(1)
        self.cap.release()
        cv2.destroyAllWindows()
        self.voice.cleanup()

        if self.config.get("logging.save_statistics"):
            self.analytics.save_statistics()

        self.analytics.print_statistics()
        print("✓ Application terminated successfully")


def main():
    try:
        app = VisionAssistant(config_path="config.yaml")
        app.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
