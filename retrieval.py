"""
Multimodal Retrieval System for VLM Comparison Tool
Based on byaldi and ColPali-style models
"""

import base64
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from byaldi import RAGMultiModalModel


class PerformanceMonitor:
    """Monitor GPU memory usage and timing during indexing"""

    def __init__(self):
        self.start_time = None
        self.indexing_start_time = None
        self.download_complete_time = None
        self.indexing_complete_time = None
        self.gpu_memory_usage = []

    def start_timing(self):
        """Start the overall timing"""
        self.start_time = time.time()

    def start_indexing(self):
        """Mark when indexing actually starts (after model download)"""
        self.indexing_start_time = time.time()
        self.download_complete_time = time.time()

    def complete_indexing(self):
        """Mark when indexing is complete"""
        self.indexing_complete_time = time.time()

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                1024**2
            )  # MB
            gpu_used = torch.cuda.memory_allocated(0) / (1024**2)  # MB
            gpu_free = gpu_memory - gpu_used

            return {
                "total_mb": gpu_memory,
                "used_mb": gpu_used,
                "free_mb": gpu_free,
                "utilization_percent": (gpu_used / gpu_memory) * 100,
            }
        return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "utilization_percent": 0}

    def record_gpu_usage(self):
        """Record current GPU usage with timestamp"""
        usage = self.get_gpu_memory_usage()
        usage["timestamp"] = time.time()
        self.gpu_memory_usage.append(usage)
        return usage

    def get_timing_summary(self) -> Dict[str, float]:
        """Get timing summary"""
        summary = {}
        if self.start_time and self.download_complete_time:
            summary["download_time"] = self.download_complete_time - self.start_time
        if self.indexing_start_time and self.indexing_complete_time:
            summary["indexing_time"] = (
                self.indexing_complete_time - self.indexing_start_time
            )
        if self.start_time and self.indexing_complete_time:
            summary["total_time"] = self.indexing_complete_time - self.start_time
        return summary


class IndexingLogger:
    """Logger for indexing process with file output"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create timestamped log file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"indexing_{timestamp}.log"

        # Setup logger
        self.logger = logging.getLogger(f"indexing_{timestamp}")
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def log_gpu_usage(self, usage: Dict[str, float]):
        self.logger.info(
            f"GPU Memory: {usage['used_mb']:.1f}MB/{usage['total_mb']:.1f}MB ({usage['utilization_percent']:.1f}%)"
        )


class MultimodalRetriever:
    """Multimodal retrieval system using ColPali-style models"""

    SUPPORTED_MODELS = ["vidore/colqwen2-v1.0", "vidore/colpali-v1.3"]

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.model = None
        self.index_name = None
        self.temp_dir = None
        self.monitor = PerformanceMonitor()
        self.logger = None

    def _check_gpu_safety(self) -> Tuple[bool, str]:
        """Check if GPU is safe for use"""
        try:
            if not torch.cuda.is_available():
                return False, "CUDA not available"

            # Check GPU memory using torch
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory / (
                1024**3
            )  # GB
            allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # GB

            memory_usage = allocated_memory / total_memory if total_memory > 0 else 0

            # Be conservative with smaller GPUs
            if total_memory <= 6.5:  # 6GB GPU
                if memory_usage > 0.8:
                    return False, f"6GB GPU memory high: {memory_usage*100:.1f}%"
            else:
                if memory_usage > 0.9:
                    return False, f"GPU memory critically high: {memory_usage*100:.1f}%"

            return True, "GPU safe for use"

        except Exception as e:
            return False, f"Error checking GPU: {e}"

    def initialize_model(self, progress_callback=None) -> bool:
        """Initialize the RAG model with progress tracking"""
        if not self.model_name or self.model_name not in self.SUPPORTED_MODELS:
            return False

        try:
            self.logger = IndexingLogger()
            self.logger.info(f"Starting model initialization: {self.model_name}")

            self.monitor.start_timing()

            if progress_callback:
                progress_callback(
                    "Downloading model on GPU... This may take a few minutes on first use.",
                    "downloading",
                )

            self.logger.info("Loading RAG model on GPU...")

            self.model = RAGMultiModalModel.from_pretrained(self.model_name, verbose=1)

            self.monitor.start_indexing()  # Model download complete, indexing about to start
            self.logger.info("Model loaded successfully on GPU")

            if progress_callback:
                progress_callback(
                    "Model downloaded successfully on GPU. Ready for indexing.",
                    "ready",
                )

            return True

        except Exception as e:
            error_msg = f"Error initializing model: {str(e)}"
            self.logger.error(error_msg) if self.logger else print(error_msg)
            if progress_callback:
                progress_callback(f"Error: {error_msg}", "error")
            return False

    def index_pdf(self, pdf_data: bytes, progress_callback=None) -> bool:
        """Index a PDF for retrieval with GPU OOM handling"""
        if not self.model:
            return False

        try:
            # Create temporary directory for PDF
            self.temp_dir = tempfile.mkdtemp()
            pdf_path = Path(self.temp_dir) / "document.pdf"

            # Save PDF to temp file
            with open(pdf_path, "wb") as f:
                f.write(pdf_data)

            self.logger.info(f"PDF saved to: {pdf_path}")

            # Generate unique index name
            timestamp = int(time.time())
            self.index_name = f"pdf_index_{timestamp}"

            if progress_callback:
                device_mode = "CPU" if self.using_cpu else "GPU"
                progress_callback(
                    f"Starting PDF indexing on {device_mode}...", "indexing"
                )

            self.logger.info(
                f"Creating index '{self.index_name}' from PDF on {'CPU' if self.using_cpu else 'GPU'}"
            )

            # Record initial GPU usage (if available)
            initial_usage = self.monitor.record_gpu_usage()
            if initial_usage.get("total_mb", 0) > 0:
                self.logger.log_gpu_usage(initial_usage)

            try:
                # Index the PDF
                self.model.index(
                    input_path=str(pdf_path),
                    index_name=self.index_name,
                    store_collection_with_index=True,
                    overwrite=True,
                )
            except Exception as e:
                error_msg = f"Error indexing PDF: {str(e)}"
                self.logger.error(error_msg)
                if progress_callback:
                    progress_callback(f"Error: {error_msg}", "error")
                return False

            self.monitor.complete_indexing()

            # Record final GPU usage (if available)
            final_usage = self.monitor.record_gpu_usage()
            if final_usage.get("total_mb", 0) > 0:
                self.logger.log_gpu_usage(final_usage)

            # Log timing summary
            timing = self.monitor.get_timing_summary()
            self.logger.info(
                f"Indexing completed on GPU - Total time: {timing.get('total_time', 0):.2f}s, Indexing time: {timing.get('indexing_time', 0):.2f}s"
            )

            if progress_callback:
                progress_callback(
                    "PDF indexing completed successfully on GPU!", "complete"
                )

            return True

        except Exception as e:
            error_msg = f"Error indexing PDF: {str(e)}"
            self.logger.error(error_msg) if self.logger else print(error_msg)
            if progress_callback:
                progress_callback(f"Error: {error_msg}", "error")
            return False

    def search(self, query: str, k: int = 5) -> List[Tuple[bytes, float]]:
        """Search for relevant pages"""
        if not self.model or not self.index_name:
            return []

        try:
            self.logger.info(f"Searching for: {query}")
            results = self.model.search(query, k=k)
            self.logger.info(f"Found {len(results)} results")

            # Convert results to (image_bytes, score) tuples
            processed_results = []
            for result in results:
                try:
                    image_bytes = base64.b64decode(result.base64)
                    score = getattr(result, "score", 0.0)
                    processed_results.append((image_bytes, score))
                except Exception as e:
                    self.logger.error(f"Error processing result: {e}")
                    continue

            return processed_results

        except Exception as e:
            error_msg = f"Error during search: {str(e)}"
            self.logger.error(error_msg) if self.logger else print(error_msg)
            return []

    def get_status(self) -> Dict[str, Any]:
        """Get current status and performance metrics"""
        status = {
            "model_loaded": self.model is not None,
            "index_ready": self.index_name is not None,
            "model_name": self.model_name,
            "device_info": {
                "device_mode": "GPU",
            },
        }

        if self.monitor:
            status["timing"] = self.monitor.get_timing_summary()
            status["gpu_usage"] = self.monitor.get_gpu_memory_usage()

        return status

    def cleanup(self):
        """Clean up temporary files and resources"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

        if self.logger:
            self.logger.info("Cleanup completed")

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()


def create_retriever(model_name: str) -> Optional[MultimodalRetriever]:
    """Factory function to create a retriever"""
    if model_name not in MultimodalRetriever.SUPPORTED_MODELS:
        return None
    return MultimodalRetriever(model_name)
