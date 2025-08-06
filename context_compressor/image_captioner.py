#!/usr/bin/env python3
"""
Image captioner for Context Compressor using BLIP.
Provides AI-generated captions for extracted images.
Implements the recommended model from test.md blueprint.
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from typing import List, Optional, Dict, Any
import re


class ImageCaptioner:
    """Image captioner using BLIP for AI-generated captions (as per test.md recommendations)."""
    
    def __init__(self, 
                 model_name: str = "Salesforce/blip-image-captioning-large",  # Updated to large model
                 device: str = "auto"):
        """
        Initialize the image captioner.
        
        Args:
            model_name: BLIP model name (large as recommended in test.md)
            device: Device to use
        """
        self.model_name = model_name
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load BLIP model with recommended configuration."""
        try:
            print(f"Loading caption model: {self.model_name}")
            
            # Load processor and model with recommended settings
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device != "cuda" and self.model is not None:
                self.model = self.model.to(self.device)
            
            print(f"✓ Loaded caption model: {self.model_name}")
            print(f"  Device: {self.device}")
            print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            print(f"✗ Failed to load caption model: {e}")
            print("  Falling back to basic image processing...")
            self.processor = None
            self.model = None
    
    def generate_caption(self, 
                        image_path: str, 
                        max_length: int = 50,
                        num_beams: int = 5) -> Optional[str]:
        """
        Generate caption for an image.
        
        Args:
            image_path: Path to image file
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            
        Returns:
            Generated caption or None if failed
        """
        if self.model is None or self.processor is None:
            return None
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    do_sample=False  # Use deterministic generation for consistency
                )
            
            # Decode caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Clean caption
            caption = self._clean_caption(caption)
            
            return caption
            
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return None
    
    def generate_captions_batch(self, 
                               image_paths: List[str], 
                               max_length: int = 50,
                               num_beams: int = 5) -> List[Optional[str]]:
        """
        Generate captions for multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            
        Returns:
            List of generated captions
        """
        captions = []
        
        # Process images in smaller batches to avoid memory issues
        batch_size = 4 if self.device == "cuda" else 2
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_captions = []
            
            for image_path in batch_paths:
                caption = self.generate_caption(image_path, max_length, num_beams)
                batch_captions.append(caption)
            
            captions.extend(batch_captions)
        
        return captions
    
    def generate_detailed_caption(self, 
                                 image_path: str, 
                                 context: Optional[str] = None) -> Optional[str]:
        """
        Generate a more detailed caption with optional context.
        
        Args:
            image_path: Path to image file
            context: Optional context about the image
            
        Returns:
            Detailed caption or None if failed
        """
        if self.model is None or self.processor is None:
            return None
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare text prompt if context is provided
            if context:
                text = f"Describe this image in detail: {context}"
                inputs = self.processor(image, text, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate detailed caption
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=5,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Clean caption
            caption = self._clean_caption(caption)
            
            return caption
            
        except Exception as e:
            print(f"Error generating detailed caption for {image_path}: {e}")
            return None
    
    def generate_figure_caption(self, 
                               image_path: str, 
                               figure_number: Optional[int] = None) -> Optional[str]:
        """
        Generate a caption specifically for figures in academic papers (as per test.md section 2B).
        
        Args:
            image_path: Path to image file
            figure_number: Figure number if known
            
        Returns:
            Figure caption or None if failed
        """
        if self.model is None or self.processor is None:
            return None
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Create figure-specific prompt (as per test.md recommendations)
            if figure_number:
                text = f"Describe this figure {figure_number} from an academic paper in detail"
            else:
                text = "Describe this figure from an academic paper in detail"
            
            inputs = self.processor(image, text, return_tensors="pt").to(self.device)
            
            # Generate figure caption
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=150,
                    num_beams=5,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Clean caption
            caption = self._clean_caption(caption)
            
            # Add figure number if provided
            if figure_number and not caption.startswith(f"Figure {figure_number}"):
                caption = f"Figure {figure_number}: {caption}"
            
            return caption
            
        except Exception as e:
            print(f"Error generating figure caption for {image_path}: {e}")
            return None
    
    def _clean_caption(self, caption: str) -> str:
        """
        Clean and format the generated caption (as per test.md section 8).
        
        Args:
            caption: Raw caption from model
            
        Returns:
            Cleaned caption
        """
        # Remove extra whitespace
        caption = re.sub(r'\s+', ' ', caption.strip())
        
        # Remove common artifacts
        caption = re.sub(r'^a\s+', '', caption, flags=re.IGNORECASE)
        caption = re.sub(r'^an\s+', '', caption, flags=re.IGNORECASE)
        caption = re.sub(r'^the\s+', '', caption, flags=re.IGNORECASE)
        
        # Capitalize first letter
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        # Ensure proper ending
        if caption and not caption.endswith(('.', '!', '?')):
            caption += '.'
        
        # Limit length to avoid token bloat (as per test.md section 8)
        if len(caption.split()) > 30:
            words = caption.split()[:30]
            caption = ' '.join(words) + '...'
        
        return caption
    
    def analyze_image_content(self, 
                             image_path: str) -> Dict[str, Any]:
        """
        Analyze image content and generate multiple descriptions.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with various image descriptions
        """
        result = {
            "basic_caption": None,
            "detailed_caption": None,
            "figure_caption": None,
            "content_type": None,
            "confidence": 0.0
        }
        
        try:
            # Generate basic caption
            result["basic_caption"] = self.generate_caption(image_path, max_length=30)
            
            # Generate detailed caption
            result["detailed_caption"] = self.generate_detailed_caption(image_path)
            
            # Generate figure caption
            result["figure_caption"] = self.generate_figure_caption(image_path)
            
            # Determine content type
            result["content_type"] = self._classify_content_type(result["basic_caption"])
            
            # Calculate confidence based on caption quality
            result["confidence"] = self._calculate_confidence(result)
            
        except Exception as e:
            print(f"Error analyzing image content for {image_path}: {e}")
        
        return result
    
    def _classify_content_type(self, caption: Optional[str]) -> str:
        """
        Classify the type of content in the image based on caption.
        
        Args:
            caption: Image caption
            
        Returns:
            Content type classification
        """
        if not caption:
            return "unknown"
        
        caption_lower = caption.lower()
        
        # Check for different content types
        if any(word in caption_lower for word in ['chart', 'graph', 'plot', 'diagram']):
            return "chart"
        elif any(word in caption_lower for word in ['table', 'data', 'matrix']):
            return "table"
        elif any(word in caption_lower for word in ['photo', 'image', 'picture']):
            return "photo"
        elif any(word in caption_lower for word in ['figure', 'illustration']):
            return "figure"
        elif any(word in caption_lower for word in ['text', 'document', 'page']):
            return "text"
        else:
            return "general"
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the analysis.
        
        Args:
            result: Analysis result dictionary
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.0
        
        # Base confidence from having captions
        if result["basic_caption"]:
            confidence += 0.3
        if result["detailed_caption"]:
            confidence += 0.3
        if result["figure_caption"]:
            confidence += 0.2
        
        # Bonus for content type classification
        if result["content_type"] != "unknown":
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "status": "loaded"
        }
