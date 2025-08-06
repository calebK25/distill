#!/usr/bin/env python3
"""
Multimodal PDF extractor for Context Compressor.
Extracts text, images, and tables from PDFs with proper modality handling.
"""

import re
import os
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import io
import subprocess
import tempfile

from .multimodal_schemas import (
    MultimodalCandidate, ModalityType, BoundingBox, 
    ImageReference, QueryClassifier
)


class MultimodalExtractor:
    """Extracts multimodal content from PDFs."""
    
    def __init__(self, output_dir: str = "extracted_images"):
        """
        Initialize the multimodal extractor.
        
        Args:
            output_dir: Directory to save extracted images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Image patterns
        self.image_patterns = [
            r'figure\s+\d+',
            r'fig\.\s*\d+',
            r'image\s+\d+',
            r'img\.\s*\d+',
            r'chart\s+\d+',
            r'plot\s+\d+',
            r'diagram\s+\d+'
        ]
        
        # Table patterns
        self.table_patterns = [
            r'table\s+\d+',
            r'tab\.\s*\d+',
            r'data\s+table',
            r'results\s+table'
        ]
        
        # Check for CLI tools
        self.has_pdfimages = self._check_pdfimages()
        self.has_mutool = self._check_mutool()
    
    def _check_pdfimages(self) -> bool:
        """Check if pdfimages CLI tool is available."""
        try:
            subprocess.run(['pdfimages', '-h'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_mutool(self) -> bool:
        """Check if mutool CLI tool is available."""
        try:
            subprocess.run(['mutool', 'draw', '-h'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _diagnose_pdf_images(self, pdf_path: str) -> Dict[str, Any]:
        """Diagnose PDF image content using CLI tools."""
        diagnosis = {
            'has_raster_images': False,
            'image_count': 0,
            'vector_figures': True,  # Assume vector by default
            'extraction_method': 'render'
        }
        
        if self.has_pdfimages:
            try:
                # List all images in PDF
                result = subprocess.run(
                    ['pdfimages', '-list', pdf_path], 
                    capture_output=True, text=True, check=True
                )
                
                # Parse output to count images
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header line
                    image_count = len(lines) - 1
                    diagnosis['image_count'] = image_count
                    diagnosis['has_raster_images'] = image_count > 0
                    diagnosis['vector_figures'] = image_count == 0
                    diagnosis['extraction_method'] = 'extract' if image_count > 0 else 'render'
                    
            except subprocess.CalledProcessError:
                pass
        
        return diagnosis
    
    def classify_query(self, query: str) -> QueryClassifier:
        """
        Classify query to determine which modalities to search.
        
        Args:
            query: User query
            
        Returns:
            QueryClassifier with modality preferences
        """
        query_lower = query.lower()
        
        # Check for image terms
        has_image_terms = any(
            re.search(pattern, query_lower) 
            for pattern in self.image_patterns
        ) or any(term in query_lower for term in [
            'figure', 'image', 'diagram', 'chart', 'plot', 'see figure',
            'what does the image', 'show me the', 'visual'
        ])
        
        # Check for table terms
        has_table_terms = any(
            re.search(pattern, query_lower) 
            for pattern in self.table_patterns
        ) or any(term in query_lower for term in [
            'table', 'data', 'value', '%', '$', 'increase', 'decrease',
            'statistics', 'numbers', 'results'
        ])
        
        # Check for numeric terms
        has_numeric_terms = bool(re.search(r'\d+', query))
        
        # Determine primary modality
        if has_image_terms:
            primary_modality = ModalityType.IMAGE
        elif has_table_terms:
            primary_modality = ModalityType.TABLE
        else:
            primary_modality = ModalityType.TEXT
        
        return QueryClassifier(
            has_image_terms=has_image_terms,
            has_table_terms=has_table_terms,
            has_numeric_terms=has_numeric_terms,
            primary_modality=primary_modality,
            confidence=0.8 if (has_image_terms or has_table_terms) else 1.0
        )
    
    def extract_from_pdf(self, pdf_path: str, doc_id: str) -> List[MultimodalCandidate]:
        """
        Extract multimodal content from PDF.
        
        Args:
            pdf_path: Path to PDF file
            doc_id: Document ID
            
        Returns:
            List of multimodal candidates
        """
        candidates = []
        
        # Diagnose PDF image content
        diagnosis = self._diagnose_pdf_images(pdf_path)
        print(f"PDF diagnosis: {diagnosis}")
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text blocks
                text_candidates = self._extract_text_blocks(page, doc_id, page_num)
                candidates.extend(text_candidates)
                
                # Extract images (uses robust methods based on diagnosis)
                image_candidates = self._extract_images(page, doc_id, page_num)
                candidates.extend(image_candidates)
                
                # Extract tables (basic detection)
                table_candidates = self._extract_tables(page, doc_id, page_num)
                candidates.extend(table_candidates)
            
            doc.close()
            
            # If no images were extracted and CLI tools are available, try CLI fallback
            image_count = len([c for c in candidates if c.modality == ModalityType.IMAGE])
            if image_count == 0 and (self.has_pdfimages or self.has_mutool):
                print("   ⚠️  No images extracted via PyMuPDF, trying CLI fallback...")
                cli_candidates = self._extract_images_cli_fallback(pdf_path, doc_id)
                candidates.extend(cli_candidates)
                print(f"   ✓ CLI fallback extracted {len(cli_candidates)} images")
            
        except Exception as e:
            print(f"Error extracting from PDF {pdf_path}: {e}")
            return []
        
        return candidates
    
    def _extract_text_blocks(self, page, doc_id: str, page_num: int) -> List[MultimodalCandidate]:
        """Extract text blocks from a page."""
        candidates = []
        
        # Get text blocks with coordinates
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" not in block:  # Skip non-text blocks
                continue
            
            # Extract text from block
            text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    text += span["text"] + " "
            
            text = text.strip()
            if not text or len(text) < 10:
                continue
            
            # Get bounding box
            bbox = BoundingBox(
                x0=block["bbox"][0],
                y0=block["bbox"][1],
                x1=block["bbox"][2],
                y1=block["bbox"][3]
            )
            
            # Determine section
            section = self._classify_section(text, page_num)
            
            # Create candidate
            candidate = MultimodalCandidate(
                id=f"{doc_id}_p{page_num}_text_{len(candidates):04d}",
                doc_id=doc_id,
                modality=ModalityType.TEXT,
                section=section,
                page=page_num,
                text=text,
                tokens=len(text.split()),  # Rough token count
                bbox=bbox,
                bm25=0.0,  # Will be computed later
                dense_sim=0.0,  # Will be computed later
                metadata={"font_size": block.get("size", 12)}
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _extract_images(self, page, doc_id: str, page_num: int) -> List[MultimodalCandidate]:
        """Extract images from a page using robust methods."""
        candidates = []
        
        # Method 1: Try to extract embedded raster images
        candidates.extend(self._extract_raster_images(page, doc_id, page_num))
        
        # Method 2: If no raster images found, try to detect and crop vector figures
        if not candidates:
            candidates.extend(self._extract_vector_figures(page, doc_id, page_num))
        
        return candidates
    
    def _extract_images_cli_fallback(self, pdf_path: str, doc_id: str) -> List[MultimodalCandidate]:
        """Extract images using CLI tools as a fallback method."""
        candidates = []
        
        # Try CLI-based raster extraction
        candidates.extend(self._extract_raster_images_cli(pdf_path, doc_id))
        
        return candidates
    
    def _extract_raster_images(self, page, doc_id: str, page_num: int) -> List[MultimodalCandidate]:
        """Extract embedded raster images from page."""
        candidates = []
        
        # Get image list with full info
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            # Handle different return formats from get_images(full=True)
            if isinstance(img_info, tuple):
                if len(img_info) >= 9:
                    xref, smask, width, height, bpc, colorspace, alt_colorspace, name, filter = img_info[:9]
                else:
                    xref = img_info[0] if img_info else None
            else:
                xref = img_info
                
            if xref is None:
                continue
                
            try:
                # Get image rectangle
                img_rect = page.get_image_bbox(xref)
                if img_rect is None:
                    # Fallback: try to get rectangle from image info
                    try:
                        img_info = page.get_image_info(xref)
                        if img_info and 'bbox' in img_info:
                            img_rect = fitz.Rect(img_info['bbox'])
                        else:
                            continue
                    except:
                        continue
                
                # Extract image data
                try:
                    img_data = page.extract_image(xref)["image"]
                except:
                    # Fallback: render the image region
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=img_rect)
                    img_data = pix.tobytes("png")
                
                # Save image
                img_filename = f"{doc_id}_p{page_num}_raster{img_index}.png"
                img_path = self.output_dir / img_filename
                
                with open(img_path, "wb") as f:
                    f.write(img_data)
                
                # Create bounding box
                bbox = BoundingBox(
                    x0=img_rect.x0,
                    y0=img_rect.y0,
                    x1=img_rect.x1,
                    y1=img_rect.y1
                )
                
                # Create image reference
                image_ref = ImageReference(
                    doc_id=doc_id,
                    page=page_num,
                    bbox=bbox,
                    image_path=str(img_path)
                )
                
                # Try to find caption
                caption = self._find_image_caption(page, img_rect)
                
                # Create candidate
                candidate = MultimodalCandidate(
                    id=f"{doc_id}_p{page_num}_raster{img_index}",
                    doc_id=doc_id,
                    modality=ModalityType.IMAGE,
                    section="figure",
                    page=page_num,
                    text=caption or f"Raster image on page {page_num + 1}",
                    tokens=len((caption or "").split()),
                    bbox=bbox,
                    image_ref=image_ref,
                    caption=caption,
                    caption_generated=caption is None,
                    bm25=0.0,
                    dense_sim=0.0,
                    image_sim=0.0,
                    metadata={"image_index": img_index, "type": "raster"}
                )
                
                candidates.append(candidate)
                
            except Exception as e:
                print(f"Error extracting raster image {img_index} from page {page_num}: {e}")
                continue
        
        return candidates
    
    def _extract_raster_images_cli(self, pdf_path: str, doc_id: str) -> List[MultimodalCandidate]:
        """Extract raster images using CLI tools as fallback."""
        candidates = []
        
        if not self.has_pdfimages:
            return candidates
        
        try:
            # Create temporary directory for extracted images
            temp_dir = self.output_dir / f"{doc_id}_cli_images"
            temp_dir.mkdir(exist_ok=True)
            
            # Extract images using pdfimages
            result = subprocess.run(
                ['pdfimages', '-all', '-p', pdf_path, str(temp_dir / "img")],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Find extracted image files
                image_files = list(temp_dir.glob("img-*"))
                
                for img_file in image_files:
                    try:
                        # Get image info
                        img_index = int(img_file.stem.split('-')[-1])
                        
                        # Create candidate
                        candidate = MultimodalCandidate(
                            id=f"{doc_id}_cli_raster{img_index}",
                            doc_id=doc_id,
                            modality=ModalityType.IMAGE,
                            section="figure",
                            page=0,  # CLI extraction doesn't provide page info
                            text=f"CLI-extracted image {img_index}",
                            tokens=5,
                            bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100),  # Default bbox
                            image_ref=ImageReference(
                                doc_id=doc_id,
                                page=0,
                                bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100),
                                image_path=str(img_file)
                            ),
                            caption=None,
                            caption_generated=True,
                            bm25=0.0,
                            dense_sim=0.0,
                            image_sim=0.0,
                            metadata={"type": "cli_raster", "source": "pdfimages"}
                        )
                        
                        candidates.append(candidate)
                        
                    except Exception as e:
                        print(f"Error processing CLI-extracted image {img_file}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error in CLI image extraction: {e}")
        
        return candidates
    
    def _extract_vector_figures(self, page, doc_id: str, page_num: int) -> List[MultimodalCandidate]:
        """Extract vector figures by detecting captions and cropping regions."""
        candidates = []
        
        # Get text blocks to find figure captions
        blocks = page.get_text("dict")["blocks"]
        figure_regions = []
        
        # Find figure captions and their regions
        for block in blocks:
            if "lines" not in block:
                continue
                
            text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    text += span["text"] + " "
            
            text = text.strip()
            
            # Check if this block contains a figure caption
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.image_patterns):
                # Get caption bounding box
                bbox = fitz.Rect(block["bbox"])
                
                # Crop region above caption (typical figure layout)
                # Adjust margins based on typical LaTeX paper layout
                page_rect = page.rect
                margin = 36  # 0.5 inch margin
                
                # Calculate figure region above caption
                figure_rect = fitz.Rect(
                    margin,  # left margin
                    margin,  # top margin  
                    page_rect.width - margin,  # right margin
                    max(margin, bbox.y0 - 8)  # just above caption
                )
                
                figure_regions.append((figure_rect, text))
        
        # Render and save each figure region
        for fig_index, (fig_rect, caption) in enumerate(figure_regions):
            try:
                # Render the figure region at high DPI
                matrix = fitz.Matrix(4, 4)  # ~288 DPI
                pix = page.get_pixmap(matrix=matrix, clip=fig_rect, alpha=False)
                
                # Save image
                img_filename = f"{doc_id}_p{page_num}_vector{fig_index}.png"
                img_path = self.output_dir / img_filename
                
                pix.save(str(img_path))
                
                # Create bounding box
                bbox = BoundingBox(
                    x0=fig_rect.x0,
                    y0=fig_rect.y0,
                    x1=fig_rect.x1,
                    y1=fig_rect.y1
                )
                
                # Create image reference
                image_ref = ImageReference(
                    doc_id=doc_id,
                    page=page_num,
                    bbox=bbox,
                    image_path=str(img_path)
                )
                
                # Create candidate
                candidate = MultimodalCandidate(
                    id=f"{doc_id}_p{page_num}_vector{fig_index}",
                    doc_id=doc_id,
                    modality=ModalityType.IMAGE,
                    section="figure",
                    page=page_num,
                    text=caption,
                    tokens=len(caption.split()),
                    bbox=bbox,
                    image_ref=image_ref,
                    caption=caption,
                    caption_generated=False,
                    bm25=0.0,
                    dense_sim=0.0,
                    image_sim=0.0,
                    metadata={"type": "vector", "caption": caption}
                )
                
                candidates.append(candidate)
                
            except Exception as e:
                print(f"Error extracting vector figure {fig_index} from page {page_num}: {e}")
                continue
        
        return candidates
    
    def _extract_tables(self, page, doc_id: str, page_num: int) -> List[MultimodalCandidate]:
        """Extract tables from a page (basic detection)."""
        candidates = []
        
        # Get text blocks that might be tables
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" not in block:
                continue
            
            # Check if block looks like a table
            if self._is_table_block(block):
                text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        text += span["text"] + " "
                
                text = text.strip()
                if not text:
                    continue
                
                # Create markdown table representation
                table_md = self._text_to_markdown_table(text)
                
                # Create bounding box
                bbox = BoundingBox(
                    x0=block["bbox"][0],
                    y0=block["bbox"][1],
                    x1=block["bbox"][2],
                    y1=block["bbox"][3]
                )
                
                # Create candidate
                candidate = MultimodalCandidate(
                    id=f"{doc_id}_p{page_num}_table_{len(candidates):04d}",
                    doc_id=doc_id,
                    modality=ModalityType.TABLE,
                    section="table",
                    page=page_num,
                    text=text,
                    tokens=len(text.split()),
                    bbox=bbox,
                    table_md=table_md,
                    bm25=0.0,
                    dense_sim=0.0,
                    metadata={"table_detected": True}
                )
                
                candidates.append(candidate)
        
        return candidates
    
    def _classify_section(self, text: str, page_num: int) -> str:
        """Classify text into document sections."""
        text_lower = text.lower()
        
        if page_num == 0 and len(text) < 200:
            return "title"
        elif "abstract" in text_lower:
            return "abstract"
        elif "introduction" in text_lower:
            return "introduction"
        elif "method" in text_lower or "approach" in text_lower:
            return "methods"
        elif "result" in text_lower or "experiment" in text_lower:
            return "results"
        elif "discussion" in text_lower or "conclusion" in text_lower:
            return "discussion"
        elif "reference" in text_lower or "bibliography" in text_lower:
            return "references"
        else:
            return "main"
    
    def _find_image_caption(self, page, img_rect) -> Optional[str]:
        """Find caption for an image."""
        # Look for text near the image
        expanded_rect = fitz.Rect(
            img_rect.x0 - 50,
            img_rect.y0 - 50,
            img_rect.x1 + 50,
            img_rect.y1 + 50
        )
        
        text = page.get_text("text", clip=expanded_rect)
        
        # Look for figure caption patterns
        caption_patterns = [
            r'figure\s+\d+[:\s]+([^.\n]+)',
            r'fig\.\s*\d+[:\s]+([^.\n]+)',
            r'image\s+\d+[:\s]+([^.\n]+)'
        ]
        
        for pattern in caption_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _is_table_block(self, block) -> bool:
        """Check if a text block looks like a table."""
        if "lines" not in block:
            return False
        
        # Count lines and check for tabular structure
        lines = block["lines"]
        if len(lines) < 2:
            return False
        
        # Check for consistent spacing (table-like structure)
        line_lengths = []
        for line in lines:
            spans = line.get("spans", [])
            if spans:
                line_lengths.append(len(spans))
        
        # If we have consistent number of columns, it might be a table
        if len(set(line_lengths)) <= 2 and max(line_lengths) >= 2:
            return True
        
        return False
    
    def _text_to_markdown_table(self, text: str) -> str:
        """Convert text to markdown table format."""
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return text
        
        # Simple conversion - assume first line is header
        header = lines[0]
        data_lines = lines[1:]
        
        # Split by common delimiters
        delimiters = ['\t', '  ', ' | ', ';', ',']
        for delim in delimiters:
            if delim in header:
                break
        else:
            delim = '  '  # Default to double space
        
        # Create markdown table
        md_table = []
        
        # Header
        header_cells = [cell.strip() for cell in header.split(delim)]
        md_table.append('| ' + ' | '.join(header_cells) + ' |')
        
        # Separator
        md_table.append('|' + '|'.join(['---'] * len(header_cells)) + '|')
        
        # Data rows
        for line in data_lines[:10]:  # Limit to 10 rows
            cells = [cell.strip() for cell in line.split(delim)]
            if len(cells) == len(header_cells):
                md_table.append('| ' + ' | '.join(cells) + ' |')
        
        return '\n'.join(md_table)
