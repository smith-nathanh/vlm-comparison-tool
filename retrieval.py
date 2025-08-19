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


def check_high_gpu_usage():
    """Check GPU usage and warn if it nears 90%"""
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory / (
                1024**3
            )  # GB
            allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # GB
            memory_usage = allocated_memory / total_memory if total_memory > 0 else 0

            if memory_usage >= 0.9:
                logging.warning(f"GPU memory usage high: {memory_usage*100:.1f}%")
    except Exception:
        pass  # Silently ignore GPU check errors


class MultimodalRetriever:
    """Multimodal retrieval system using ColPali-style models"""

    SUPPORTED_MODELS = ["vidore/colqwen2-v1.0", "vidore/colpali-v1.3"]

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.model = None
        self.index_name = None
        self.temp_dir = None

    def initialize_model(self, progress_callback=None) -> bool:
        """Initialize the RAG model with progress tracking"""
        if not self.model_name or self.model_name not in self.SUPPORTED_MODELS:
            return False

        try:
            logging.info(f"Starting model initialization: {self.model_name}")

            if progress_callback:
                progress_callback(
                    "Downloading model... This may take a few minutes on first use.",
                    "downloading",
                )

            logging.info("Loading RAG model...")
            self.model = RAGMultiModalModel.from_pretrained(self.model_name, verbose=1)

            check_high_gpu_usage()
            logging.info("Model loaded successfully")

            if progress_callback:
                progress_callback(
                    "Model downloaded successfully. Ready for indexing.", "ready"
                )

            return True

        except Exception as e:
            error_msg = f"Error initializing model: {str(e)}"
            logging.error(error_msg)
            if progress_callback:
                progress_callback(f"Error: {error_msg}", "error")
            return False

    def index_pdf(self, pdf_data: bytes, progress_callback=None) -> bool:
        """Index a PDF for retrieval"""
        if not self.model:
            return False

        try:
            # Create temporary directory for PDF
            self.temp_dir = tempfile.mkdtemp()
            pdf_path = Path(self.temp_dir) / "document.pdf"

            # Save PDF to temp file
            with open(pdf_path, "wb") as f:
                f.write(pdf_data)

            # Generate unique index name
            timestamp = int(time.time())
            self.index_name = f"pdf_index_{timestamp}"

            if progress_callback:
                progress_callback("Starting PDF indexing...", "indexing")

            logging.info("Starting PDF indexing...")

            # Index the PDF
            self.model.index(
                input_path=str(pdf_path),
                index_name=self.index_name,
                store_collection_with_index=True,
                overwrite=True,
            )

            logging.info("PDF indexing completed")

            if progress_callback:
                progress_callback("PDF indexing completed successfully!", "complete")

            return True

        except Exception as e:
            # Enhanced error detection
            error_str = str(e).lower()

            # Check if it's a GPU OOM issue
            if any(
                oom_keyword in error_str
                for oom_keyword in [
                    "out of memory",
                    "cuda out of memory",
                    "oom",
                    "memory",
                ]
            ):
                logging.error(f"GPU OUT OF MEMORY during indexing: {str(e)}")
                if progress_callback:
                    progress_callback("GPU out of memory during indexing", "error")
            # Check for CUDA errors
            elif "cuda" in error_str or "device" in error_str:
                logging.error(f"GPU/CUDA error during indexing: {str(e)}")
                if progress_callback:
                    progress_callback(f"GPU error: {str(e)}", "error")
            else:
                logging.error(f"Error indexing PDF: {str(e)}")
                if progress_callback:
                    progress_callback(f"Indexing error: {str(e)}", "error")

            return False

    def search(self, query: str, k: int = 5) -> List[Tuple[bytes, float, int]]:
        """Search for relevant pages"""
        if not self.model or not self.index_name:
            return []

        try:
            results = self.model.search(query, k=k)

            # Convert results to (image_bytes, score, page_number) tuples
            processed_results = []
            for i, result in enumerate(results):
                try:
                    image_bytes = base64.b64decode(result.base64)
                    score = result.score
                    page_number = result.page_num  # Use the actual page number from byaldi
                    processed_results.append((image_bytes, score, page_number))
                except Exception as e:
                    logging.error(f"Error processing result: {e}")
                    continue

            return processed_results

        except Exception as e:
            error_msg = f"Error during search: {str(e)}"
            logging.error(error_msg)
            return []

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "model_loaded": self.model is not None,
            "index_ready": self.index_name is not None,
            "model_name": self.model_name,
            "device_info": {"device_mode": "GPU"},
        }

    def cleanup(self):
        """Clean up temporary files and resources"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()


def create_retriever(model_name: str) -> Optional[MultimodalRetriever]:
    """Factory function to create a retriever"""
    if model_name not in MultimodalRetriever.SUPPORTED_MODELS:
        return None
    return MultimodalRetriever(model_name)
