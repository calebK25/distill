"""
Sentence trimming module for Context Compressor.
"""

from typing import List, Dict, Tuple, Optional
import re
from .schemas import Candidate, MappingItem
from .utils import (
    split_sentences, 
    extract_anchors, 
    count_tokens, 
    cosine_similarity,
    stable_sort_with_tie_breaker
)


class SentenceTrimmer:
    """Implements sentence-level trimming with anchor awareness."""
    
    def __init__(self, use_cross_encoder: bool = False):
        """
        Initialize sentence trimmer.
        
        Args:
            use_cross_encoder: Whether to use cross-encoder for sentence scoring
        """
        self.use_cross_encoder = use_cross_encoder
    
    def score_sentence(
        self, 
        sentence: str, 
        query: str,
        candidate_embedding: Optional[List[float]] = None,
        query_embedding: Optional[List[float]] = None
    ) -> float:
        """
        Score a sentence based on relevance and anchor elements.
        
        Args:
            sentence: Sentence to score
            query: Query text
            candidate_embedding: Candidate passage embedding (optional)
            query_embedding: Query embedding (optional)
            
        Returns:
            Sentence score (higher is better)
        """
        score = 0.0
        
        # 1. Sentence-query similarity
        if query_embedding and candidate_embedding:
            # Use embedding similarity if available
            # For simplicity, we'll use the candidate embedding as proxy for sentence
            # In a more sophisticated implementation, you'd compute sentence embeddings
            sim_score = cosine_similarity(candidate_embedding, query_embedding)
            score += sim_score * 0.6  # Weight for similarity
        else:
            # Fallback to simple text overlap
            query_words = set(query.lower().split())
            sentence_words = set(sentence.lower().split())
            if query_words:
                overlap = len(query_words.intersection(sentence_words)) / len(query_words)
                score += overlap * 0.6
        
        # 2. Anchor hits (numbers, entities)
        anchors = extract_anchors(sentence)
        anchor_score = min(len(anchors) * 0.1, 0.3)  # Cap anchor contribution
        score += anchor_score
        
        # 3. Optional cross-encoder scoring
        if self.use_cross_encoder:
            try:
                from .reranker import get_reranker
                reranker = get_reranker("BAAI/bge-reranker-large")  # Default model
                cross_encoder_score = reranker.score_single(query, sentence)
                score += cross_encoder_score * 0.2
            except Exception as e:
                # Fallback if reranker fails
                print(f"Warning: Cross-encoder scoring failed: {e}")
                score += 0.0
        
        return score
    
    def trim_candidate(
        self, 
        candidate: Candidate, 
        target_tokens: int,
        query: str,
        query_embedding: Optional[List[float]] = None
    ) -> Tuple[str, int, bool]:
        """
        Trim a candidate passage to fit within target token budget.
        
        Args:
            candidate: Candidate passage to trim
            target_tokens: Target token count
            query: Query text
            query_embedding: Query embedding (optional)
            
        Returns:
            Tuple of (trimmed_text, actual_tokens, was_trimmed)
        """
        if candidate.tokens <= target_tokens:
            return candidate.text, candidate.tokens, False
        
        # Split into sentences
        sentences = split_sentences(candidate.text)
        if not sentences:
            return candidate.text, candidate.tokens, False
        
        # Score sentences
        scored_sentences = []
        for sentence in sentences:
            score = self.score_sentence(
                sentence, 
                query, 
                candidate.embedding, 
                query_embedding
            )
            scored_sentences.append((sentence, score))
        
        # Sort by score (descending) with stable tie-breaking
        def sort_key(item):
            sentence, score = item
            return score
        
        sorted_sentences = stable_sort_with_tie_breaker(
            scored_sentences, 
            sort_key, 
            reverse=True
        )
        
        # Select sentences while preserving order and staying within budget
        selected_sentences = []
        used_tokens = 0
        
        # First, add sentences in original order if they fit
        original_sentences = [s for s, _ in scored_sentences]
        for sentence in original_sentences:
            sentence_tokens = count_tokens(sentence)
            if used_tokens + sentence_tokens <= target_tokens:
                selected_sentences.append(sentence)
                used_tokens += sentence_tokens
            else:
                break
        
        # If we still have budget, add more sentences from the sorted list
        # but maintain original order
        if used_tokens < target_tokens:
            remaining_sentences = [s for s, _ in sorted_sentences if s not in selected_sentences]
            for sentence in remaining_sentences:
                sentence_tokens = count_tokens(sentence)
                if used_tokens + sentence_tokens <= target_tokens:
                    # Find the right position to insert (maintain original order)
                    insert_pos = 0
                    for i, orig_sentence in enumerate(original_sentences):
                        if orig_sentence == sentence:
                            insert_pos = i
                            break
                    
                    # Insert at the appropriate position
                    if insert_pos < len(selected_sentences):
                        selected_sentences.insert(insert_pos, sentence)
                    else:
                        selected_sentences.append(sentence)
                    
                    used_tokens += sentence_tokens
                else:
                    break
        
        # Join sentences
        trimmed_text = " ".join(selected_sentences)
        actual_tokens = count_tokens(trimmed_text)
        was_trimmed = actual_tokens < candidate.tokens
        
        return trimmed_text, actual_tokens, was_trimmed
    
    def trim_candidates(
        self, 
        candidates: List[Candidate], 
        budget: int,
        query: str,
        query_embedding: Optional[List[float]] = None,
        soft_overflow: float = 0.15
    ) -> Tuple[List[str], List[MappingItem], int]:
        """
        Trim multiple candidates to fit within budget.
        
        Args:
            candidates: List of candidate passages
            budget: Token budget
            query: Query text
            query_embedding: Query embedding (optional)
            soft_overflow: Allow soft overflow before trimming
            
        Returns:
            Tuple of (trimmed_texts, mapping_items, total_tokens)
        """
        if not candidates:
            return [], [], 0
        
        # Calculate target tokens per candidate (simple equal distribution)
        # In a more sophisticated approach, you might weight by relevance
        target_per_candidate = budget // len(candidates)
        soft_limit = int(budget * (1 + soft_overflow))
        
        trimmed_texts = []
        mapping_items = []
        total_tokens = 0
        
        for candidate in candidates:
            # Trim candidate
            trimmed_text, actual_tokens, was_trimmed = self.trim_candidate(
                candidate, 
                target_per_candidate,
                query,
                query_embedding
            )
            
            trimmed_texts.append(trimmed_text)
            total_tokens += actual_tokens
            
            # Create mapping item
            mapping_item = MappingItem(
                id=candidate.id,
                doc_id=candidate.doc_id,
                section=candidate.section,
                page=candidate.page,
                tokens=actual_tokens,
                trimmed=was_trimmed
            )
            mapping_items.append(mapping_item)
        
        # If we exceed soft limit, apply additional trimming
        if total_tokens > soft_limit:
            # Simple approach: trim proportionally
            scale_factor = soft_limit / total_tokens
            new_total = 0
            
            for i, candidate in enumerate(candidates):
                new_target = int(mapping_items[i].tokens * scale_factor)
                trimmed_text, actual_tokens, was_trimmed = self.trim_candidate(
                    candidate,
                    new_target,
                    query,
                    query_embedding
                )
                
                trimmed_texts[i] = trimmed_text
                mapping_items[i].tokens = actual_tokens
                mapping_items[i].trimmed = was_trimmed or mapping_items[i].trimmed
                new_total += actual_tokens
            
            total_tokens = new_total
        
        return trimmed_texts, mapping_items, total_tokens
