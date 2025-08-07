#!/usr/bin/env python3
"""
Production logging configuration for Context Compressor.
Provides structured logging, monitoring, and error tracking.
"""

import logging
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for production logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        self.metrics = {
            "start_time": time.time(),
            "operations": {},
            "errors": [],
            "performance": {
                "total_processing_time": 0,
                "compression_time": 0,
                "extraction_time": 0,
                "llm_time": 0,
                "memory_usage": []
            }
        }
    
    def record_operation(self, operation: str, duration: float, success: bool = True, **kwargs):
        """Record an operation with timing and metadata."""
        if operation not in self.metrics["operations"]:
            self.metrics["operations"][operation] = {
                "count": 0,
                "total_time": 0,
                "success_count": 0,
                "error_count": 0,
                "avg_time": 0
            }
        
        op_metrics = self.metrics["operations"][operation]
        op_metrics["count"] += 1
        op_metrics["total_time"] += duration
        op_metrics["avg_time"] = op_metrics["total_time"] / op_metrics["count"]
        
        if success:
            op_metrics["success_count"] += 1
        else:
            op_metrics["error_count"] += 1
            self.metrics["errors"].append({
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat(),
                "duration": duration,
                "error": kwargs.get("error", "Unknown error")
            })
    
    def record_error(self, operation: str, error: Exception, duration: float = 0):
        """Record an error with full context."""
        self.record_operation(operation, duration, success=False, error=str(error))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_time = time.time() - self.metrics["start_time"]
        return {
            "total_runtime": total_time,
            "operations": self.metrics["operations"],
            "error_count": len(self.metrics["errors"]),
            "success_rate": self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        total_ops = sum(op["count"] for op in self.metrics["operations"].values())
        if total_ops == 0:
            return 1.0
        total_success = sum(op["success_count"] for op in self.metrics["operations"].values())
        return total_success / total_ops


class ProductionLogger:
    """Production-ready logger with structured logging and monitoring."""
    
    def __init__(self, name: str = "context_compressor", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add structured formatter
        self._setup_handlers()
        
        # Initialize performance monitor
        self.monitor = PerformanceMonitor()
    
    def _setup_handlers(self):
        """Setup logging handlers."""
        # Console handler with structured output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler for all logs
        file_handler = logging.FileHandler(self.log_dir / "context_compressor.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.FileHandler(self.log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(error_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with extra fields."""
        self._log_with_extra(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with extra fields."""
        self._log_with_extra(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with exception context."""
        if error:
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)
            kwargs["traceback"] = traceback.format_exc()
        
        self._log_with_extra(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with extra fields."""
        self._log_with_extra(logging.DEBUG, message, **kwargs)
    
    def _log_with_extra(self, level: int, message: str, **kwargs):
        """Log message with extra fields."""
        record = self.logger.makeRecord(
            self.name, level, "", 0, message, (), None
        )
        record.extra_fields = kwargs
        self.logger.handle(record)
    
    @contextmanager
    def operation_timer(self, operation: str, **kwargs):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            self.info(f"Starting operation: {operation}", operation=operation, **kwargs)
            yield
            duration = time.time() - start_time
            self.monitor.record_operation(operation, duration, success=True, **kwargs)
            self.info(f"Completed operation: {operation}", 
                     operation=operation, duration=duration, **kwargs)
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.record_error(operation, e, duration)
            self.error(f"Failed operation: {operation}", error=e, 
                      operation=operation, duration=duration, **kwargs)
            raise
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.monitor.get_summary()


# Global logger instance
production_logger = ProductionLogger()

