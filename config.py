"""
Configuration Manager
Handles loading and validating configuration settings from YAML file
"""

import yaml
import os
from typing import Dict, Any


class ConfigManager:
    """Load and manage application configuration"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Returns:
            Dictionary containing configuration
        """
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                print(f"✓ Configuration loaded from {self.config_path}")
                return config
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "model": {
                "yolo_model": "yolov8n",
                "caption_model": "Salesforce/blip-image-captioning-base",
                "confidence_threshold": 0.5,
                "iou_threshold": 0.4
            },
            "performance": {
                "frame_skip": 2,
                "target_fps": 15,
                "enable_multithreading": True,
                "thread_pool_size": 4,
                "model_precision": "fp32"
            },
            "detection": {
                "classes_to_detect": ["person", "pen", "pencil", "bottle", "watch", "phone", "laptop"],
                "only_show_detected_classes": False,
                "max_objects_track": 50
            },
            "display": {
                "show_fps": True,
                "show_confidence": True,
                "show_labels": True,
                "show_description": True,
                "box_thickness": 2,
                "font_size": 0.6,
                "text_color": [0, 255, 0],
                "box_color": [0, 255, 0]
            },
            "speech": {
                "enable_tts": True,
                "speak_on_change_only": True,
                "speech_rate": 150,
                "volume": 1.0
            },
            "logging": {
                "enable_logging": True,
                "log_file": "logs/detections.log",
                "save_statistics": True,
                "stats_file": "logs/statistics.json",
                "log_interval": 100
            },
            "webcam": {
                "camera_index": 0,
                "frame_width": 640,
                "frame_height": 480,
                "fps": 30
            },
            "api": {
                "enable_api": True,
                "api_host": "127.0.0.1",
                "api_port": 8000,
                "api_debug": False
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation (e.g., 'model.yolo_model')

        Args:
            key: Configuration key with dot notation
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by dot notation

        Args:
            key: Configuration key with dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def print_config(self) -> None:
        """Print current configuration"""
        print("\n" + "="*50)
        print("CURRENT CONFIGURATION")
        print("="*50)
        for key, value in self.config.items():
            print(f"\n[{key}]")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"  {value}")
        print("="*50 + "\n")


config = None


def get_config(config_path: str = "config.yaml") -> ConfigManager:
    """Get or create global config instance"""
    global config
    if config is None:
        config = ConfigManager(config_path)
    return config
