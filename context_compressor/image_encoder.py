#!/usr/bin/env python3
"""
Image encoder for Context Compressor using OpenCLIP.
Provides true image similarity for multimodal compression.
Implements the recommended model from test.md blueprint.
"""

import torch
import open_clip
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from pathlib import Path


class ImageEncoder:
    """Image encoder using OpenCLIP for multimodal similarity (as per test.md recommendations)."""
    
    def __init__(self, 
                 model_name: str = "ViT-H-14",  # Updated to recommended H-14 model
                 pretrained: str = "laion2b_s32b_b79k",  # Updated to recommended weights
                 device: str = "auto"):
        """
        Initialize the image encoder.
        
        Args:
            model_name: OpenCLIP model name (ViT-H-14 as recommended in test.md)
            pretrained: Pretrained weights (laion2b_s32b_b79k as recommended)
            device: Device to use
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self._load_model()
    
    def _load_model(self):
        """Load OpenCLIP model with recommended configuration."""
        try:
            print(f"Loading image model: {self.model_name}/{self.pretrained}")
            
            # Load model with recommended configuration
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained,
                device=self.device,
                precision="fp16" if self.device == "cuda" else "fp32"  # Use fp16 for GPU efficiency
            )
            
            # Get tokenizer for text encoding
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            
            print(f"✓ Loaded image model: {self.model_name}/{self.pretrained}")
            print(f"  Device: {self.device}")
            print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            print(f"✗ Failed to load image model: {e}")
            print("  Falling back to basic image processing...")
            self.model = None
            self.tokenizer = None
            self.preprocess = None
    
    def encode_image(self, image_path: str) -> Optional[List[float]]:
        """
        Encode an image to embedding vector.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image embedding vector or None if failed
        """
        if self.model is None or self.preprocess is None:
            return None
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Ensure tensor type matches model precision
            if self.device == "cuda" and image_tensor.dtype != torch.float16:
                image_tensor = image_tensor.half()
            
            # Encode image
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                
                # Normalize features for better similarity computation
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.cpu().numpy()[0]
            
            return image_features.tolist()
            
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def encode_text(self, text: str) -> Optional[List[float]]:
        """
        Encode text to embedding vector for text→image similarity.
        
        Args:
            text: Text to encode
            
        Returns:
            Text embedding vector or None if failed
        """
        if self.model is None or self.tokenizer is None:
            return None
        
        try:
            # Tokenize and encode text
            text_tokens = self.tokenizer([text]).to(self.device)
            
            # Tokenizer output should remain as Long/Int tensors, not converted to half
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize features for better similarity computation
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.cpu().numpy()[0]
            
            return text_features.tolist()
            
        except Exception as e:
            print(f"Error encoding text: {e}")
            return None
    
    def compute_similarity(self, 
                          image_path: str, 
                          text: str) -> Optional[float]:
        """
        Compute similarity between image and text (text→image retrieval).
        
        Args:
            image_path: Path to image file
            text: Text to compare against
            
        Returns:
            Similarity score or None if failed
        """
        if self.model is None:
            return None
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Ensure tensor type matches model precision
            if self.device == "cuda" and image_tensor.dtype != torch.float16:
                image_tensor = image_tensor.half()
            
            # Tokenize text
            text_tokens = self.tokenizer([text]).to(self.device)
            
            # Tokenizer output should remain as Long/Int tensors, not converted to half
            
            # Compute similarity
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = (image_features @ text_features.T).item()
            
            return similarity
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return None
    
    def batch_encode_images(self, 
                           image_paths: List[str]) -> List[Optional[List[float]]]:
        """
        Encode multiple images in batch for efficiency.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of image embedding vectors
        """
        if self.model is None or self.preprocess is None:
            return [None] * len(image_paths)
        
        embeddings = []
        
        # Process images in smaller batches to avoid memory issues
        batch_size = 8 if self.device == "cuda" else 4
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_embeddings = []
            
            try:
                # Load and preprocess batch
                images = []
                valid_indices = []
                
                for j, image_path in enumerate(batch_paths):
                    try:
                        image = Image.open(image_path).convert('RGB')
                        images.append(image)
                        valid_indices.append(j)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                        batch_embeddings.append(None)
                
                if images:
                    # Process valid images in batch
                    image_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
                    
                    with torch.no_grad():
                        image_features = self.model.encode_image(image_tensors)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        image_features = image_features.cpu().numpy()
                    
                    # Map back to original indices
                    for j, embedding in enumerate(image_features):
                        batch_embeddings.insert(valid_indices[j], embedding.tolist())
                
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error in batch encoding: {e}")
                # Fill with None for failed batch
                embeddings.extend([None] * len(batch_paths))
        
        return embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "status": "loaded"
        }
