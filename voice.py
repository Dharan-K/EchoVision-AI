"""
Text-to-Speech Module
Converts text descriptions to speech using pyttsx3
"""

import pyttsx3
import threading
import logging

logger = logging.getLogger("VoiceAssistant")


class VoiceAssistant:
    """Text-to-speech voice output"""

    def __init__(self, rate: int = 150, volume: float = 1.0):
        """
        Initialize text-to-speech engine

        Args:
            rate: Speech rate (words per minute)
            volume: Volume (0.0 to 1.0)
        """
        self.rate = rate
        self.volume = volume
        self.last_spoken_text = None
        self.is_speaking = False
        self.lock = threading.Lock()
        self.engine = None

        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            logger.info(f"✓ Voice assistant initialized (rate: {rate}, volume: {volume})")
        except Exception as e:
            logger.warning(f"⚠ Voice assistant initialization failed: {e}. TTS will be disabled.")
            self.engine = None

    def speak(self, text: str, wait: bool = False) -> None:
        """Speak text"""
        if not self.engine or not text or text.strip() == "":
            return

        try:
            with self.lock:
                self.engine.say(text)
                self.is_speaking = True

            if wait:
                self.engine.runAndWait()
                with self.lock:
                    self.is_speaking = False
            else:
                threading.Thread(
                    target=self._speak_async,
                    args=(text,),
                    daemon=True
                ).start()

        except Exception as e:
            logger.warning(f"⚠ Error speaking: {e}")

    def _speak_async(self, text: str) -> None:
        """Speak text asynchronously"""
        if not self.engine:
            return
        try:
            with self.lock:
                self.engine.runAndWait()
                self.is_speaking = False
        except Exception as e:
            logger.warning(f"⚠ Error in async speech: {e}")

    def speak_if_changed(self, text: str, wait: bool = False) -> bool:
        """Only speak if text changed"""
        if text != self.last_spoken_text:
            self.speak(text, wait=wait)
            self.last_spoken_text = text
            return True
        return False

    def set_rate(self, rate: int) -> None:
        """Change speech rate"""
        if not self.engine:
            return
        self.engine.setProperty('rate', rate)
        self.rate = rate

    def set_volume(self, volume: float) -> None:
        """Change volume"""
        if not self.engine or not (0.0 <= volume <= 1.0):
            return
        self.engine.setProperty('volume', volume)
        self.volume = volume

    def get_voices(self) -> list:
        """Get available voices"""
        if not self.engine:
            return []
        try:
            voices = self.engine.getProperty('voices')
            return [(v.id, v.name) for v in voices]
        except Exception:
            return []

    def set_voice(self, voice_id: str) -> None:
        """Set voice by ID"""
        if not self.engine:
            return
        try:
            self.engine.setProperty('voice', voice_id)
        except Exception as e:
            logger.warning(f"⚠ Error setting voice: {e}")

    def stop(self) -> None:
        """Stop current speech"""
        if not self.engine:
            return
        try:
            with self.lock:
                self.engine.stop()
                self.is_speaking = False
        except Exception as e:
            logger.warning(f"⚠ Error stopping speech: {e}")

    def cleanup(self) -> None:
        """Clean up resources"""
        if not self.engine:
            return
        try:
            self.stop()
            del self.engine
            self.engine = None
            logger.info("✓ Voice assistant cleaned up")
        except Exception as e:
            logger.warning(f"⚠ Error cleaning up voice: {e}")

