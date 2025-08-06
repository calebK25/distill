#!/usr/bin/env python3
"""
Table extractor for Context Compressor using Camelot.
Provides better structured table extraction from PDFs.
Implements the recommended configuration from test.md blueprint.
"""

import camelot
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import re
from pathlib import Path


class TableExtractor:
    """Table extractor using Camelot for structured table extraction (as per test.md recommendations)."""
    
    def __init__(self,
                 flavor: str = "lattice",  # Recommended for structured tables
                 edge_tol: int = 500,      # Recommended tolerance
                 row_tol: int = 10):       # Recommended row tolerance
        """
        Initialize the table extractor.
        
        Args:
            flavor: Camelot flavor ('lattice' as recommended in test.md)
            edge_tol: Edge tolerance for table detection
            row_tol: Row tolerance for table detection
        """
        self.flavor = flavor
        self.edge_tol = edge_tol
        self.row_tol = row_tol
        
        # Quality thresholds (as per test.md recommendations)
        self.min_accuracy = 50  # Minimum accuracy for table extraction
        self.max_whitespace = 30  # Maximum whitespace percentage
        
        print(f"✓ Initialized table extractor (Camelot {flavor})")
    
    def extract_tables_from_pdf(self,
                               pdf_path: str,
                               pages: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF using Camelot (as per test.md section 2B).
        
        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to extract from (None for all)
            
        Returns:
            List of extracted table dictionaries
        """
        try:
            print(f"Extracting tables from {pdf_path}...")
            
            # Extract tables using Camelot with recommended settings
            tables = camelot.read_pdf(
                pdf_path,
                pages=','.join(map(str, pages)) if pages else 'all',
                flavor=self.flavor,
                edge_tol=self.edge_tol,
                row_tol=self.row_tol,
                strip_text='\n'  # Clean up text
            )
            
            print(f"Found {len(tables)} potential tables")
            
            extracted_tables = []
            
            for i, table in enumerate(tables):
                # Filter by quality (as per test.md recommendations)
                if self._is_table_quality_acceptable(table):
                    table_dict = self._process_table(table, i)
                    extracted_tables.append(table_dict)
                else:
                    print(f"Skipping table {i} due to low quality (accuracy: {table.parsing_report['accuracy']:.1f}%)")
            
            print(f"✓ Extracted {len(extracted_tables)} quality tables")
            return extracted_tables
            
        except Exception as e:
            print(f"Error extracting tables from PDF {pdf_path}: {e}")
            return []
    
    def _is_table_quality_acceptable(self, table) -> bool:
        """
        Check if table quality meets minimum standards (as per test.md section 8).
        
        Args:
            table: Camelot table object
            
        Returns:
            True if table quality is acceptable
        """
        report = table.parsing_report
        
        # Check accuracy threshold
        if report['accuracy'] < self.min_accuracy:
            return False
        
        # Check whitespace threshold
        if report['whitespace'] > self.max_whitespace:
            return False
        
        # Check if table has reasonable structure
        if len(table.df) < 2 or len(table.df.columns) < 2:
            return False
        
        return True
    
    def _process_table(self, table, table_index: int) -> Dict[str, Any]:
        """
        Process a Camelot table into our format (as per test.md section 2B).
        
        Args:
            table: Camelot table object
            table_index: Index of the table
            
        Returns:
            Processed table dictionary
        """
        # Get table data
        df = table.df
        
        # Clean the dataframe
        df = self._clean_dataframe(df)
        
        # Convert to markdown (as per test.md section 4)
        markdown_table = self._dataframe_to_markdown(df)
        
        # Extract text representation
        text = self._dataframe_to_text(df)
        
        # Get table metadata
        metadata = {
            "accuracy": table.parsing_report['accuracy'],
            "whitespace": table.parsing_report['whitespace'],
            "order": table.parsing_report['order'],
            "page": table.parsing_report['page'],
            "table_index": table_index,
            "shape": df.shape,
            "flavor": self.flavor,
            "quality_score": self.get_table_quality_score({
                "dataframe": df,
                "metadata": metadata
            })
        }
        
        return {
            "text": text,
            "markdown": markdown_table,
            "dataframe": df,
            "metadata": metadata,
            "bbox": table._bbox
        }
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataframe by removing empty rows/columns and merging headers.
        
        Args:
            df: Raw dataframe from Camelot
            
        Returns:
            Cleaned dataframe
        """
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean cell values
        df = df.applymap(lambda x: self._clean_cell(x) if pd.notna(x) else "")
        
        # Try to identify and merge header rows
        df = self._merge_headers(df)
        
        return df
    
    def _clean_cell(self, cell: str) -> str:
        """
        Clean a single cell value.
        
        Args:
            cell: Raw cell value
            
        Returns:
            Cleaned cell value
        """
        if not isinstance(cell, str):
            return str(cell)
        
        # Remove extra whitespace
        cell = re.sub(r'\s+', ' ', cell.strip())
        
        # Remove common PDF artifacts but keep important characters
        cell = re.sub(r'[^\w\s\-\.,%$+\-*/\()\[\]]', '', cell)
        
        return cell
    
    def _merge_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge multi-row headers into single header (as per test.md section 2B).
        
        Args:
            df: Dataframe with potential multi-row headers
            
        Returns:
            Dataframe with merged headers
        """
        if len(df) < 2:
            return df
        
        # Check if first few rows look like headers
        header_rows = 0
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            # Check if row contains mostly text and few numbers
            text_count = sum(1 for cell in row if isinstance(cell, str) and cell.strip())
            if text_count > len(row) * 0.7:
                header_rows += 1
            else:
                break
        
        if header_rows > 1:
            # Merge header rows
            headers = []
            for col in df.columns:
                header_parts = []
                for i in range(header_rows):
                    cell = df.iloc[i][col]
                    if cell and cell.strip():
                        header_parts.append(cell.strip())
                headers.append(' '.join(header_parts) if header_parts else f'Column_{col}')
            
            # Create new dataframe with merged headers
            new_df = df.iloc[header_rows:].copy()
            new_df.columns = headers
            return new_df
        
        return df
    
    def _dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """
        Convert dataframe to markdown table format (as per test.md section 4).
        
        Args:
            df: Dataframe to convert
            
        Returns:
            Markdown table string
        """
        if df.empty:
            return ""
        
        # Create markdown table
        markdown_lines = []
        
        # Header
        header = '| ' + ' | '.join(str(col) for col in df.columns) + ' |'
        markdown_lines.append(header)
        
        # Separator
        separator = '|' + '|'.join(['---'] * len(df.columns)) + '|'
        markdown_lines.append(separator)
        
        # Data rows (limit to avoid token bloat as per test.md section 8)
        max_rows = 10
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= max_rows:
                markdown_lines.append("| ... |" + "|".join(["..."] * (len(df.columns) - 1)) + "|")
                break
            data_row = '| ' + ' | '.join(str(cell) for cell in row) + ' |'
            markdown_lines.append(data_row)
        
        return '\n'.join(markdown_lines)
    
    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """
        Convert dataframe to text representation.
        
        Args:
            df: Dataframe to convert
            
        Returns:
            Text representation
        """
        if df.empty:
            return ""
        
        # Create text representation
        lines = []
        
        # Header
        header = ' | '.join(str(col) for col in df.columns)
        lines.append(header)
        
        # Separator
        separator = '-' * len(header)
        lines.append(separator)
        
        # Data rows (limit to avoid token bloat)
        max_rows = 5
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= max_rows:
                lines.append("... (table truncated)")
                break
            data_row = ' | '.join(str(cell) for cell in row)
            lines.append(data_row)
        
        return '\n'.join(lines)
    
    def extract_tables_from_page(self,
                                pdf_path: str,
                                page: int) -> List[Dict[str, Any]]:
        """
        Extract tables from a specific page.
        
        Args:
            pdf_path: Path to PDF file
            page: Page number (1-indexed)
            
        Returns:
            List of extracted table dictionaries
        """
        return self.extract_tables_from_pdf(pdf_path, pages=[page])
    
    def get_table_quality_score(self, table_dict: Dict[str, Any]) -> float:
        """
        Calculate quality score for extracted table (as per test.md section 8).
        
        Args:
            table_dict: Table dictionary from extraction
            
        Returns:
            Quality score (0-1)
        """
        metadata = table_dict['metadata']
        df = table_dict['dataframe']
        
        # Base score from Camelot accuracy
        base_score = metadata['accuracy'] / 100.0
        
        # Penalize for whitespace issues
        whitespace_penalty = metadata['whitespace'] / 100.0
        
        # Bonus for good structure
        structure_bonus = 0.0
        if len(df) > 1 and len(df.columns) > 1:
            structure_bonus = 0.1
        
        # Penalize for empty cells
        total_cells = len(df) * len(df.columns)
        empty_cells = sum(1 for cell in df.values.flatten() if not str(cell).strip())
        empty_penalty = empty_cells / total_cells if total_cells > 0 else 0
        
        # Bonus for numeric content (tables with numbers are often more valuable)
        numeric_bonus = 0.0
        numeric_cells = 0
        for cell in df.values.flatten():
            if isinstance(cell, str) and re.search(r'\d+', cell):
                numeric_cells += 1
        if total_cells > 0:
            numeric_ratio = numeric_cells / total_cells
            numeric_bonus = numeric_ratio * 0.1
        
        final_score = base_score - whitespace_penalty + structure_bonus - empty_penalty + numeric_bonus
        return max(0.0, min(1.0, final_score))
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about the table extractor."""
        return {
            "flavor": self.flavor,
            "edge_tol": self.edge_tol,
            "row_tol": self.row_tol,
            "min_accuracy": self.min_accuracy,
            "max_whitespace": self.max_whitespace
        }
