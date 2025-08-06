"""
Performance benchmarks and stress tests for Context Compressor.
"""

import pytest
import time
import statistics
from context_compressor.compressor import ContextCompressor
from context_compressor.schemas import (
    Candidate,
    CompressionRequest,
    CompressionParams
)


class TestBenchmarks:
    """Performance benchmarks and stress tests."""

    def setup_method(self):
        """Set up test data."""
        self.compressor = ContextCompressor()
        
        # Create test candidates with varying characteristics
        self.small_candidates = self._create_candidates(10, 20, 30)
        self.medium_candidates = self._create_candidates(50, 20, 30)
        self.large_candidates = self._create_candidates(200, 20, 30)
        self.very_large_candidates = self._create_candidates(500, 20, 30)

    def _create_candidates(self, count, min_tokens, max_tokens):
        """Create test candidates with specified characteristics."""
        candidates = []
        for i in range(count):
            tokens = min_tokens + (i % (max_tokens - min_tokens))
            text = f"This is test candidate {i} with {tokens} tokens. " * (tokens // 10)
            
            candidate = Candidate(
                id=f"c_{i:03d}",
                doc_id=f"d_{i // 10:02d}",
                section=f"Section_{i % 5}",
                page=i + 1,
                text=text,
                tokens=tokens,
                bm25=5.0 + (i % 5) * 0.5,
                dense_sim=0.5 + (i % 10) * 0.05,
                embedding=[0.1 + i * 0.01] * 5
            )
            candidates.append(candidate)
        return candidates

    @pytest.mark.benchmark
    def test_small_scale_performance(self):
        """Benchmark small-scale compression (10 candidates)."""
        request = CompressionRequest(
            q="What are the main findings?",
            B=100,
            candidates=self.small_candidates
        )
        
        times = []
        for _ in range(10):
            start_time = time.time()
            response = self.compressor.compress(request)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Verify response quality
            assert response.stats.used <= 100
            assert len(response.context) > 0
        
        # Performance metrics
        avg_time = statistics.mean(times)
        p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile
        
        print(f"Small scale (10 candidates):")
        print(f"  Average time: {avg_time:.2f} ms")
        print(f"  95th percentile: {p95_time:.2f} ms")
        
        # Performance assertions
        assert avg_time < 100  # Average < 100ms
        assert p95_time < 200  # 95th percentile < 200ms

    @pytest.mark.benchmark
    def test_medium_scale_performance(self):
        """Benchmark medium-scale compression (50 candidates)."""
        request = CompressionRequest(
            q="Analyze the methodology and results",
            B=150,
            candidates=self.medium_candidates
        )
        
        times = []
        for _ in range(5):
            start_time = time.time()
            response = self.compressor.compress(request)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
            
            assert response.stats.used <= 150
        
        avg_time = statistics.mean(times)
        p95_time = statistics.quantiles(times, n=20)[18]
        
        print(f"Medium scale (50 candidates):")
        print(f"  Average time: {avg_time:.2f} ms")
        print(f"  95th percentile: {p95_time:.2f} ms")
        
        assert avg_time < 200  # Average < 200ms
        assert p95_time < 400  # 95th percentile < 400ms

    @pytest.mark.benchmark
    def test_large_scale_performance(self):
        """Benchmark large-scale compression (200 candidates)."""
        request = CompressionRequest(
            q="Summarize the comprehensive study findings",
            B=200,
            candidates=self.large_candidates
        )
        
        times = []
        for _ in range(3):
            start_time = time.time()
            response = self.compressor.compress(request)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
            
            assert response.stats.used <= 200
        
        avg_time = statistics.mean(times)
        p95_time = statistics.quantiles(times, n=20)[18]
        
        print(f"Large scale (200 candidates):")
        print(f"  Average time: {avg_time:.2f} ms")
        print(f"  95th percentile: {p95_time:.2f} ms")
        
        assert avg_time < 500  # Average < 500ms
        assert p95_time < 1000  # 95th percentile < 1s

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_very_large_scale_performance(self):
        """Benchmark very large-scale compression (500 candidates)."""
        request = CompressionRequest(
            q="Comprehensive analysis of all research findings",
            B=300,
            candidates=self.very_large_candidates
        )
        
        times = []
        for _ in range(2):
            start_time = time.time()
            response = self.compressor.compress(request)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
            
            assert response.stats.used <= 300
        
        avg_time = statistics.mean(times)
        
        print(f"Very large scale (500 candidates):")
        print(f"  Average time: {avg_time:.2f} ms")
        
        assert avg_time < 2000  # Average < 2s

    @pytest.mark.benchmark
    def test_throughput_benchmark(self):
        """Benchmark throughput (requests per second)."""
        request = CompressionRequest(
            q="Quick analysis",
            B=50,
            candidates=self.small_candidates
        )
        
        # Warm up
        for _ in range(3):
            self.compressor.compress(request)
        
        # Measure throughput
        start_time = time.time()
        request_count = 20
        
        for _ in range(request_count):
            self.compressor.compress(request)
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = request_count / total_time
        
        print(f"Throughput benchmark:")
        print(f"  Requests: {request_count}")
        print(f"  Total time: {total_time:.2f} s")
        print(f"  Throughput: {throughput:.2f} req/s")
        
        assert throughput > 10  # At least 10 req/s

    @pytest.mark.benchmark
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many requests to test memory usage
        requests = []
        for i in range(10):
            request = CompressionRequest(
                q=f"Query {i}",
                B=100,
                candidates=self.medium_candidates
            )
            requests.append(request)
        
        # Process requests
        for request in requests:
            self.compressor.compress(request)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage benchmark:")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Final memory: {final_memory:.2f} MB")
        print(f"  Memory increase: {memory_increase:.2f} MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 500  # Less than 500MB increase

    @pytest.mark.benchmark
    def test_cache_performance_impact(self):
        """Benchmark cache performance impact."""
        request = CompressionRequest(
            q="Cache performance test",
            B=100,
            candidates=self.medium_candidates
        )
        
        # First request (no cache)
        start_time = time.time()
        response1 = self.compressor.compress(request)
        first_request_time = (time.time() - start_time) * 1000
        
        # Second request (with cache)
        start_time = time.time()
        response2 = self.compressor.compress(request)
        second_request_time = (time.time() - start_time) * 1000
        
        # Verify responses are identical
        assert response1.context == response2.context
        
        speedup = first_request_time / second_request_time
        
        print(f"Cache performance impact:")
        print(f"  First request: {first_request_time:.2f} ms")
        print(f"  Second request: {second_request_time:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Cache should provide significant speedup
        assert speedup > 2  # At least 2x speedup

    @pytest.mark.benchmark
    def test_reranker_performance_impact(self):
        """Benchmark reranker performance impact."""
        # Without reranker
        request_no_reranker = CompressionRequest(
            q="Performance comparison",
            B=100,
            candidates=self.medium_candidates,
            params=CompressionParams(use_reranker=False)
        )
        
        start_time = time.time()
        response1 = self.compressor.compress(request_no_reranker)
        time_no_reranker = (time.time() - start_time) * 1000
        
        # With reranker
        request_with_reranker = CompressionRequest(
            q="Performance comparison",
            B=100,
            candidates=self.medium_candidates,
            params=CompressionParams(use_reranker=True)
        )
        
        start_time = time.time()
        response2 = self.compressor.compress(request_with_reranker)
        time_with_reranker = (time.time() - start_time) * 1000
        
        print(f"Reranker performance impact:")
        print(f"  Without reranker: {time_no_reranker:.2f} ms")
        print(f"  With reranker: {time_with_reranker:.2f} ms")
        print(f"  Overhead: {time_with_reranker - time_no_reranker:.2f} ms")
        
        # Reranker should not cause excessive overhead
        assert time_with_reranker < time_no_reranker * 5  # Less than 5x overhead

    @pytest.mark.benchmark
    def test_budget_scaling_performance(self):
        """Benchmark performance with different budget sizes."""
        budgets = [25, 50, 100, 200, 400]
        times = []
        
        for budget in budgets:
            request = CompressionRequest(
                q="Budget scaling test",
                B=budget,
                candidates=self.medium_candidates
            )
            
            start_time = time.time()
            response = self.compressor.compress(request)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)
            assert response.stats.used <= budget
        
        print(f"Budget scaling performance:")
        for budget, time_ms in zip(budgets, times):
            print(f"  Budget {budget}: {time_ms:.2f} ms")
        
        # Performance should scale reasonably with budget
        # Larger budgets should not cause exponential slowdown
        assert times[-1] < times[0] * 10  # 400 tokens < 10x 25 tokens

    @pytest.mark.benchmark
    def test_concurrent_requests_simulation(self):
        """Simulate concurrent requests performance."""
        import threading
        import queue
        
        def worker(request_queue, result_queue):
            """Worker function for concurrent processing."""
            while True:
                try:
                    request = request_queue.get_nowait()
                except queue.Empty:
                    break
                
                start_time = time.time()
                response = self.compressor.compress(request)
                end_time = time.time()
                
                result_queue.put((end_time - start_time) * 1000)
                request_queue.task_done()
        
        # Create requests
        request_queue = queue.Queue()
        result_queue = queue.Queue()
        
        for i in range(10):
            request = CompressionRequest(
                q=f"Concurrent request {i}",
                B=100,
                candidates=self.small_candidates
            )
            request_queue.put(request)
        
        # Start worker threads
        threads = []
        for _ in range(4):  # 4 concurrent workers
            thread = threading.Thread(target=worker, args=(request_queue, result_queue))
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Collect results
        times = []
        while not result_queue.empty():
            times.append(result_queue.get())
        
        avg_time = statistics.mean(times)
        max_time = max(times)
        
        print(f"Concurrent requests simulation:")
        print(f"  Average time: {avg_time:.2f} ms")
        print(f"  Max time: {max_time:.2f} ms")
        print(f"  Threads: 4")
        
        # Concurrent processing should not cause excessive slowdown
        assert max_time < avg_time * 3  # Max time < 3x average

    @pytest.mark.benchmark
    def test_compression_ratio_benchmark(self):
        """Benchmark compression ratios."""
        request = CompressionRequest(
            q="Compression ratio test",
            B=100,
            candidates=self.large_candidates
        )
        
        response = self.compressor.compress(request)
        
        original_tokens = sum(c.tokens for c in self.large_candidates)
        compressed_tokens = response.stats.used
        compression_ratio = (original_tokens - compressed_tokens) / original_tokens * 100
        
        print(f"Compression ratio benchmark:")
        print(f"  Original tokens: {original_tokens}")
        print(f"  Compressed tokens: {compressed_tokens}")
        print(f"  Compression ratio: {compression_ratio:.1f}%")
        
        # Should achieve significant compression
        assert compression_ratio > 30  # At least 30% compression
        assert response.stats.used <= 100  # Within budget
