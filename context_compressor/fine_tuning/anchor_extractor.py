"""
Anchor extraction utilities for identifying key entities, numbers, and phrases.
"""

import re
from typing import List, Dict, Any, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AnchorExtractor:
    """Extracts anchors (entities, numbers, keyphrases) from text for coverage optimization."""
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize anchor extractor.
        
        Args:
            use_spacy: Whether to use spaCy for NER (requires spacy model)
        """
        self.use_spacy = use_spacy
        self.nlp = None
        
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for NER")
            except (ImportError, OSError):
                logger.warning("spaCy not available, falling back to regex extraction")
                self.use_spacy = False
    
    def extract_anchors(self, text: str) -> Dict[str, List[str]]:
        """
        Extract anchors from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with anchor types and values
        """
        anchors = {
            "entities": [],
            "numbers": [],
            "keyphrases": [],
            "dates": []
        }
        
        # Extract entities using spaCy or regex
        if self.use_spacy and self.nlp:
            anchors["entities"] = self._extract_entities_spacy(text)
        else:
            anchors["entities"] = self._extract_entities_regex(text)
        
        # Extract numbers and units
        anchors["numbers"] = self._extract_numbers(text)
        
        # Extract dates
        anchors["dates"] = self._extract_dates(text)
        
        # Extract keyphrases
        anchors["keyphrases"] = self._extract_keyphrases(text)
        
        return anchors
    
    def _extract_entities_spacy(self, text: str) -> List[str]:
        """Extract entities using spaCy NER."""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]:
                entities.append(ent.text.strip())
        
        return list(set(entities))
    
    def _extract_entities_regex(self, text: str) -> List[str]:
        """Extract entities using regex patterns."""
        entities = []
        
        # Company names (capitalized words followed by Inc, Corp, etc.)
        company_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Co)\b'
        companies = re.findall(company_pattern, text)
        entities.extend(companies)
        
        # Product names (quoted strings or capitalized phrases)
        product_pattern = r'["\']([^"\']+)["\']'
        products = re.findall(product_pattern, text)
        entities.extend(products)
        
        # Acronyms (3+ capital letters)
        acronym_pattern = r'\b[A-Z]{3,}\b'
        acronyms = re.findall(acronym_pattern, text)
        entities.extend(acronyms)
        
        return list(set(entities))
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numbers and units from text."""
        numbers = []
        
        # Percentage patterns
        percent_pattern = r'\b\d+(?:\.\d+)?%\b'
        percentages = re.findall(percent_pattern, text)
        numbers.extend(percentages)
        
        # Currency patterns
        currency_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
        currencies = re.findall(currency_pattern, text)
        numbers.extend(currencies)
        
        # Number with units
        unit_pattern = r'\b\d+(?:\.\d+)?\s*(?:million|billion|thousand|M|B|K)\b'
        units = re.findall(unit_pattern, text, re.IGNORECASE)
        numbers.extend(units)
        
        # Plain numbers (significant ones)
        number_pattern = r'\b\d{2,}(?:\.\d+)?\b'
        plain_numbers = re.findall(number_pattern, text)
        # Filter out small numbers that are likely not significant
        significant_numbers = [n for n in plain_numbers if float(n) >= 10]
        numbers.extend(significant_numbers)
        
        return list(set(numbers))
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text."""
        dates = []
        
        # Year patterns
        year_pattern = r'\b(?:19|20)\d{2}\b'
        years = re.findall(year_pattern, text)
        dates.extend(years)
        
        # Month patterns
        month_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
        months = re.findall(month_pattern, text, re.IGNORECASE)
        dates.extend(months)
        
        # Quarter patterns
        quarter_pattern = r'\bQ[1-4]\s+\d{4}\b'
        quarters = re.findall(quarter_pattern, text)
        dates.extend(quarters)
        
        return list(set(dates))
    
    def _extract_keyphrases(self, text: str) -> List[str]:
        """Extract keyphrases from text."""
        keyphrases = []
        
        # Technical terms (camelCase or snake_case)
        tech_pattern = r'\b[a-z]+(?:[A-Z][a-z]+)*\b'
        tech_terms = re.findall(tech_pattern, text)
        keyphrases.extend([term for term in tech_terms if len(term) > 5])
        
        # Multi-word phrases (3+ words, capitalized)
        phrase_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,}\b'
        phrases = re.findall(phrase_pattern, text)
        keyphrases.extend(phrases)
        
        # Important keywords (revenue, growth, performance, etc.)
        important_words = [
            'revenue', 'growth', 'performance', 'efficiency', 'optimization',
            'improvement', 'increase', 'decrease', 'margin', 'profit',
            'cost', 'investment', 'return', 'market', 'customer',
            'product', 'service', 'technology', 'innovation', 'strategy'
        ]
        
        text_lower = text.lower()
        for word in important_words:
            if word in text_lower:
                keyphrases.append(word)
        
        return list(set(keyphrases))
    
    def calculate_coverage(self, anchors: Dict[str, List[str]], context: str) -> float:
        """
        Calculate anchor coverage in context.
        
        Args:
            anchors: Dictionary of anchor types and values
            context: Context text to check coverage
            
        Returns:
            Coverage score (0.0 to 1.0)
        """
        if not anchors:
            return 0.0
        
        context_lower = context.lower()
        total_anchors = 0
        covered_anchors = 0
        
        for anchor_type, anchor_values in anchors.items():
            for anchor in anchor_values:
                total_anchors += 1
                if anchor.lower() in context_lower:
                    covered_anchors += 1
        
        return covered_anchors / total_anchors if total_anchors > 0 else 0.0
    
    def find_anchor_rich_passages(
        self, 
        anchors: Dict[str, List[str]], 
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find passages that are rich in anchors.
        
        Args:
            anchors: Dictionary of anchor types and values
            candidates: List of candidate passages
            
        Returns:
            List of candidates with coverage scores
        """
        scored_candidates = []
        
        for candidate in candidates:
            text = candidate.get('text', '')
            coverage = self.calculate_coverage(anchors, text)
            
            scored_candidate = candidate.copy()
            scored_candidate['anchor_coverage'] = coverage
            scored_candidates.append(scored_candidate)
        
        # Sort by coverage score (descending)
        scored_candidates.sort(key=lambda x: x['anchor_coverage'], reverse=True)
        
        return scored_candidates
    
    def create_anchor_summary(self, anchors: Dict[str, List[str]]) -> str:
        """
        Create a summary of extracted anchors.
        
        Args:
            anchors: Dictionary of anchor types and values
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        for anchor_type, values in anchors.items():
            if values:
                summary_parts.append(f"{anchor_type.title()}: {', '.join(values[:5])}")
                if len(values) > 5:
                    summary_parts[-1] += f" (+{len(values) - 5} more)"
        
        return "; ".join(summary_parts) if summary_parts else "No anchors found"
