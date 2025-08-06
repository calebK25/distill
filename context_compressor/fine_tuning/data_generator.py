"""
Training data generation for fine-tuning models.
"""

import json
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """Generates training data for fine-tuning from documents and QA pairs."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
    
    def generate_qa_pairs_from_document(
        self,
        document_text: str,
        num_questions: int = 10,
        question_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate QA pairs from document text.
        
        Args:
            document_text: Document text to generate questions from
            num_questions: Number of questions to generate
            question_types: Types of questions to generate
            
        Returns:
            List of QA pairs
        """
        if question_types is None:
            question_types = ['factual', 'numerical', 'comparative']
        
        qa_pairs = []
        
        # Split document into sentences
        sentences = self._split_into_sentences(document_text)
        
        # Generate different types of questions
        for question_type in question_types:
            type_questions = self._generate_questions_by_type(
                sentences, question_type, num_questions // len(question_types)
            )
            qa_pairs.extend(type_questions)
        
        # Shuffle and limit to requested number
        random.shuffle(qa_pairs)
        return qa_pairs[:num_questions]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _generate_questions_by_type(
        self, 
        sentences: List[str], 
        question_type: str, 
        num_questions: int
    ) -> List[Dict[str, Any]]:
        """Generate questions of specific type."""
        qa_pairs = []
        
        if question_type == 'factual':
            qa_pairs = self._generate_factual_questions(sentences, num_questions)
        elif question_type == 'numerical':
            qa_pairs = self._generate_numerical_questions(sentences, num_questions)
        elif question_type == 'comparative':
            qa_pairs = self._generate_comparative_questions(sentences, num_questions)
        
        return qa_pairs
    
    def _generate_factual_questions(self, sentences: List[str], num_questions: int) -> List[Dict[str, Any]]:
        """Generate factual questions from sentences."""
        qa_pairs = []
        
        # Simple template-based question generation
        templates = [
            "What is {entity}?",
            "How does {entity} work?",
            "What are the benefits of {entity}?",
            "What is the purpose of {entity}?",
            "What are the key features of {entity}?"
        ]
        
        for sentence in sentences[:num_questions]:
            # Extract potential entities (capitalized words)
            import re
            entities = re.findall(r'\b[A-Z][a-z]+\b', sentence)
            
            if entities:
                entity = random.choice(entities)
                template = random.choice(templates)
                question = template.format(entity=entity)
                
                qa_pairs.append({
                    'question': question,
                    'answer': sentence,
                    'context': sentence,
                    'type': 'factual'
                })
        
        return qa_pairs
    
    def _generate_numerical_questions(self, sentences: List[str], num_questions: int) -> List[Dict[str, Any]]:
        """Generate numerical questions from sentences."""
        qa_pairs = []
        
        templates = [
            "What is the value of {metric}?",
            "How much {metric} was achieved?",
            "What percentage {metric} was reported?",
            "What is the {metric} figure?"
        ]
        
        for sentence in sentences[:num_questions]:
            # Look for numbers and metrics
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?%?', sentence)
            metrics = ['revenue', 'growth', 'performance', 'efficiency', 'cost', 'profit']
            
            if numbers:
                number = random.choice(numbers)
                metric = random.choice(metrics)
                template = random.choice(templates)
                question = template.format(metric=metric)
                
                qa_pairs.append({
                    'question': question,
                    'answer': f"{number}",
                    'context': sentence,
                    'type': 'numerical'
                })
        
        return qa_pairs
    
    def _generate_comparative_questions(self, sentences: List[str], num_questions: int) -> List[Dict[str, Any]]:
        """Generate comparative questions from sentences."""
        qa_pairs = []
        
        templates = [
            "How does {entity1} compare to {entity2}?",
            "What is the difference between {entity1} and {entity2}?",
            "Which is better: {entity1} or {entity2}?"
        ]
        
        # Find sentences with comparative words
        comparative_words = ['better', 'worse', 'higher', 'lower', 'more', 'less', 'improved', 'decreased']
        
        for sentence in sentences[:num_questions]:
            if any(word in sentence.lower() for word in comparative_words):
                import re
                entities = re.findall(r'\b[A-Z][a-z]+\b', sentence)
                
                if len(entities) >= 2:
                    entity1, entity2 = random.sample(entities, 2)
                    template = random.choice(templates)
                    question = template.format(entity1=entity1, entity2=entity2)
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': sentence,
                        'context': sentence,
                        'type': 'comparative'
                    })
        
        return qa_pairs
    
    def create_training_examples(
        self,
        qa_pairs: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        num_negatives: int = 3,
        hard_negative_ratio: float = 0.3
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create training examples from QA pairs and candidates.
        
        Args:
            qa_pairs: List of QA pairs
            candidates: List of candidate passages
            num_negatives: Number of negative examples per positive
            hard_negative_ratio: Ratio of hard negatives to random negatives
            
        Returns:
            Tuple of (bi_encoder_examples, cross_encoder_examples)
        """
        bi_encoder_examples = []
        cross_encoder_examples = []
        
        for qa in qa_pairs:
            question = qa['question']
            answer = qa['answer']
            
            # Find positive examples (passages that contain answer)
            positive_passages = []
            for candidate in candidates:
                if self._contains_answer(candidate.get('text', ''), answer):
                    positive_passages.append(candidate)
            
            # Find negative examples
            negative_passages = []
            for candidate in candidates:
                if not self._contains_answer(candidate.get('text', ''), answer):
                    negative_passages.append(candidate)
            
            # Create bi-encoder examples
            for positive in positive_passages[:2]:  # Limit positives
                # Select negatives
                num_hard_negatives = int(num_negatives * hard_negative_ratio)
                num_random_negatives = num_negatives - num_hard_negatives
                
                # Hard negatives: similar but wrong
                hard_negatives = self._select_hard_negatives(
                    positive, negative_passages, num_hard_negatives
                )
                
                # Random negatives
                random_negatives = random.sample(
                    negative_passages, 
                    min(num_random_negatives, len(negative_passages))
                )
                
                all_negatives = hard_negatives + random_negatives
                
                bi_encoder_examples.append({
                    'query': question,
                    'positive': positive.get('text', ''),
                    'negatives': [n.get('text', '') for n in all_negatives]
                })
            
            # Create cross-encoder examples
            for positive in positive_passages[:1]:
                for negative in negative_passages[:num_negatives]:
                    cross_encoder_examples.append({
                        'query': question,
                        'positive': positive.get('text', ''),
                        'negative': negative.get('text', '')
                    })
        
        return bi_encoder_examples, cross_encoder_examples
    
    def _contains_answer(self, text: str, answer: str) -> bool:
        """Check if text contains the answer."""
        text_lower = text.lower()
        answer_lower = answer.lower()
        
        # Simple keyword matching
        answer_words = answer_lower.split()
        if len(answer_words) <= 2:
            return answer_lower in text_lower
        
        # For longer answers, check if most words are present
        matches = sum(1 for word in answer_words if word in text_lower)
        return matches >= len(answer_words) * 0.7
    
    def _select_hard_negatives(
        self, 
        positive: Dict[str, Any], 
        negative_passages: List[Dict[str, Any]], 
        num_hard_negatives: int
    ) -> List[Dict[str, Any]]:
        """Select hard negative examples (similar but wrong)."""
        if not negative_passages:
            return []
        
        # Simple similarity based on shared words
        positive_text = positive.get('text', '').lower()
        positive_words = set(positive_text.split())
        
        scored_negatives = []
        for negative in negative_passages:
            negative_text = negative.get('text', '').lower()
            negative_words = set(negative_text.split())
            
            # Calculate word overlap
            overlap = len(positive_words.intersection(negative_words))
            similarity = overlap / len(positive_words.union(negative_words)) if positive_words.union(negative_words) else 0
            
            scored_negatives.append((similarity, negative))
        
        # Sort by similarity (descending) and take top
        scored_negatives.sort(key=lambda x: x[0], reverse=True)
        return [negative for _, negative in scored_negatives[:num_hard_negatives]]
    
    def save_training_data(
        self,
        bi_encoder_examples: List[Dict[str, Any]],
        cross_encoder_examples: List[Dict[str, Any]],
        output_dir: str = "training_data"
    ) -> Tuple[str, str]:
        """
        Save training data to files.
        
        Args:
            bi_encoder_examples: Bi-encoder training examples
            cross_encoder_examples: Cross-encoder training examples
            output_dir: Output directory
            
        Returns:
            Tuple of (bi_encoder_path, cross_encoder_path)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save bi-encoder data
        bi_encoder_path = Path(output_dir) / "bi_encoder_training.json"
        with open(bi_encoder_path, 'w') as f:
            json.dump(bi_encoder_examples, f, indent=2)
        
        # Save cross-encoder data
        cross_encoder_path = Path(output_dir) / "cross_encoder_training.json"
        with open(cross_encoder_path, 'w') as f:
            json.dump(cross_encoder_examples, f, indent=2)
        
        logger.info(f"Saved {len(bi_encoder_examples)} bi-encoder examples to {bi_encoder_path}")
        logger.info(f"Saved {len(cross_encoder_examples)} cross-encoder examples to {cross_encoder_path}")
        
        return str(bi_encoder_path), str(cross_encoder_path)
    
    def load_training_data(self, bi_encoder_path: str, cross_encoder_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load training data from files.
        
        Args:
            bi_encoder_path: Path to bi-encoder training data
            cross_encoder_path: Path to cross-encoder training data
            
        Returns:
            Tuple of (bi_encoder_examples, cross_encoder_examples)
        """
        with open(bi_encoder_path, 'r') as f:
            bi_encoder_examples = json.load(f)
        
        with open(cross_encoder_path, 'r') as f:
            cross_encoder_examples = json.load(f)
        
        return bi_encoder_examples, cross_encoder_examples
