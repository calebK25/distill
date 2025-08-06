"""
Fine-tuning implementation for bi-encoder and cross-encoder models.
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from sentence_transformers import losses
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


class FineTuner:
    """Fine-tunes bi-encoder and cross-encoder models for context compression."""
    
    def __init__(self, device: str = None):
        """
        Initialize fine-tuner.
        
        Args:
            device: Device to use for training (auto-detect if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def train_bi_encoder(
        self,
        training_data: List[Dict[str, Any]],
        model_name: str = "BAAI/bge-large-en-v1.5",
        output_path: str = "models/bge-finetuned",
        batch_size: int = 128,
        epochs: int = 2,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        max_seq_length: int = 512
    ) -> str:
        """
        Train bi-encoder model with contrastive learning.
        
        Args:
            training_data: List of training examples with 'query', 'positive', 'negatives'
            model_name: Base model to fine-tune
            output_path: Path to save fine-tuned model
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            max_seq_length: Maximum sequence length
            
        Returns:
            Path to saved model
        """
        logger.info(f"Starting bi-encoder training with {len(training_data)} examples")
        
        # Load base model
        model = SentenceTransformer(model_name, device=self.device)
        model.max_seq_length = max_seq_length
        
        # Prepare training examples
        train_examples = []
        for example in training_data:
            query = example['query']
            positive = example['positive']
            negatives = example.get('negatives', [])
            
            # Add positive example
            train_examples.append(InputExample(texts=[query, positive], label=1.0))
            
            # Add negative examples
            for negative in negatives:
                train_examples.append(InputExample(texts=[query, negative], label=0.0))
        
        logger.info(f"Created {len(train_examples)} training examples")
        
        # Create data loader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Define loss function (MultipleNegativesRankingLoss for contrastive learning)
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        # Train model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            use_amp=True,  # Mixed precision training
            show_progress_bar=True,
            checkpoint_path=os.path.join(output_path, "checkpoints"),
            checkpoint_save_steps=1000
        )
        
        # Save model
        os.makedirs(output_path, exist_ok=True)
        model.save(output_path)
        
        # Save training config
        config = {
            "model_name": model_name,
            "training_examples": len(train_examples),
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "max_seq_length": max_seq_length
        }
        
        with open(os.path.join(output_path, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Bi-encoder training completed. Model saved to {output_path}")
        return output_path
    
    def train_cross_encoder(
        self,
        training_data: List[Dict[str, Any]],
        model_name: str = "BAAI/bge-reranker-large",
        output_path: str = "models/reranker-finetuned",
        batch_size: int = 32,
        epochs: int = 1,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        margin: float = 0.1
    ) -> str:
        """
        Train cross-encoder model with margin ranking loss.
        
        Args:
            training_data: List of training examples with 'query', 'positive', 'negative'
            model_name: Base model to fine-tune
            output_path: Path to save fine-tuned model
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            margin: Margin for ranking loss
            
        Returns:
            Path to saved model
        """
        logger.info(f"Starting cross-encoder training with {len(training_data)} examples")
        
        # Load base model
        model = CrossEncoder(model_name, num_labels=1, device=self.device)
        
        # Prepare training examples
        train_examples = []
        for example in training_data:
            query = example['query']
            positive = example['positive']
            negative = example['negative']
            
            # Create triplet: (query, positive, negative)
            train_examples.append(InputExample(
                texts=[query, positive, negative],
                label=1.0  # Positive should rank higher than negative
            ))
        
        logger.info(f"Created {len(train_examples)} training examples")
        
        # Train model
        model.fit(
            train_examples,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
            checkpoint_path=os.path.join(output_path, "checkpoints"),
            checkpoint_save_steps=500
        )
        
        # Save model
        os.makedirs(output_path, exist_ok=True)
        model.save(output_path)
        
        # Save training config
        config = {
            "model_name": model_name,
            "training_examples": len(train_examples),
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "margin": margin
        }
        
        with open(os.path.join(output_path, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Cross-encoder training completed. Model saved to {output_path}")
        return output_path
    
    def evaluate_model(
        self,
        model_path: str,
        test_data: List[Dict[str, Any]],
        model_type: str = "bi_encoder"
    ) -> Dict[str, float]:
        """
        Evaluate fine-tuned model on test data.
        
        Args:
            model_path: Path to fine-tuned model
            test_data: Test examples
            model_type: Type of model ('bi_encoder' or 'cross_encoder')
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {model_type} model from {model_path}")
        
        if model_type == "bi_encoder":
            model = SentenceTransformer(model_path, device=self.device)
        else:
            model = CrossEncoder(model_path, device=self.device)
        
        # Calculate metrics
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
        
        # Simple evaluation - can be extended with more sophisticated metrics
        correct = 0
        total = len(test_data)
        
        for example in test_data:
            query = example['query']
            positive = example['positive']
            negative = example['negative']
            
            if model_type == "bi_encoder":
                # Encode texts
                query_emb = model.encode(query)
                pos_emb = model.encode(positive)
                neg_emb = model.encode(negative)
                
                # Calculate similarities
                pos_sim = np.dot(query_emb, pos_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(pos_emb))
                neg_sim = np.dot(query_emb, neg_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(neg_emb))
                
                # Check if positive is ranked higher
                if pos_sim > neg_sim:
                    correct += 1
            else:
                # Cross-encoder scoring
                pos_score = model.predict([query, positive])
                neg_score = model.predict([query, negative])
                
                if pos_score > neg_score:
                    correct += 1
        
        metrics["accuracy"] = correct / total if total > 0 else 0.0
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def create_training_data_from_qa(
        self,
        qa_pairs: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        num_negatives: int = 3
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create training data from QA pairs for both bi-encoder and cross-encoder.
        
        Args:
            qa_pairs: List of QA pairs with 'question', 'answer', 'context'
            candidates: List of candidate passages
            num_negatives: Number of negative examples per positive
            
        Returns:
            Tuple of (bi_encoder_data, cross_encoder_data)
        """
        logger.info(f"Creating training data from {len(qa_pairs)} QA pairs")
        
        bi_encoder_data = []
        cross_encoder_data = []
        
        for qa in qa_pairs:
            question = qa['question']
            answer = qa['answer']
            context = qa['context']
            
            # Find positive examples (passages that contain answer)
            positive_passages = []
            for candidate in candidates:
                if self._contains_answer(candidate['text'], answer):
                    positive_passages.append(candidate['text'])
            
            # Find negative examples (passages that don't contain answer)
            negative_passages = []
            for candidate in candidates:
                if not self._contains_answer(candidate['text'], answer):
                    negative_passages.append(candidate['text'])
            
            # Create bi-encoder examples
            for positive in positive_passages[:2]:  # Limit positives
                negatives = np.random.choice(
                    negative_passages, 
                    size=min(num_negatives, len(negative_passages)), 
                    replace=False
                ).tolist()
                
                bi_encoder_data.append({
                    'query': question,
                    'positive': positive,
                    'negatives': negatives
                })
            
            # Create cross-encoder examples
            for positive in positive_passages[:1]:
                for negative in negative_passages[:num_negatives]:
                    cross_encoder_data.append({
                        'query': question,
                        'positive': positive,
                        'negative': negative
                    })
        
        logger.info(f"Created {len(bi_encoder_data)} bi-encoder examples")
        logger.info(f"Created {len(cross_encoder_data)} cross-encoder examples")
        
        return bi_encoder_data, cross_encoder_data
    
    def _contains_answer(self, text: str, answer: str) -> bool:
        """
        Check if text contains the answer (simple implementation).
        
        Args:
            text: Passage text
            answer: Answer text
            
        Returns:
            True if text contains answer
        """
        # Simple keyword matching - can be improved with NLI or semantic similarity
        text_lower = text.lower()
        answer_lower = answer.lower()
        
        # Check for key phrases from answer
        answer_words = answer_lower.split()
        if len(answer_words) <= 2:
            return answer_lower in text_lower
        
        # For longer answers, check if most words are present
        matches = sum(1 for word in answer_words if word in text_lower)
        return matches >= len(answer_words) * 0.7
