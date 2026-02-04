"""
Image Captioning Module
Generates natural language descriptions of scenes using BLIP model
"""

import cv2
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from typing import Optional


class ImageCaptioner:
    """Generate captions for images using BLIP model"""

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize BLIP captioner

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = "cpu"

        print(f"Loading BLIP captioning model on {self.device}...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        print("✓ BLIP model loaded successfully")

    def generate_caption(
        self,
        frame: np.ndarray,
        max_length: int = 50,
        num_beams: int = 1
    ) -> str:
        """Generate caption for an image frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    do_sample=False
                )

            caption = self.processor.decode(output[0], skip_special_tokens=True)
            return caption

        except Exception as e:
            print(f"Error generating caption: {e}")
            return "Unable to generate description"

    def generate_caption_with_context(
        self,
        frame: np.ndarray,
        context: Optional[str] = None,
        max_length: int = 50
    ) -> str:
        """Generate caption with optional context prompt"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            text = context if context else "a photo of"
            inputs = self.processor(pil_image, text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=1,
                    do_sample=False
                )

            caption = self.processor.decode(output[0], skip_special_tokens=True)
            return caption

        except Exception as e:
            print(f"Error generating contextual caption: {e}")
            return "Unable to generate description"

    @staticmethod
    def enhance_caption(caption: str, detected_objects: str) -> str:
        """Enhance caption with detected objects summary"""
        if detected_objects and "No objects" not in detected_objects:
            return f"{caption}. {detected_objects}."
        return caption
