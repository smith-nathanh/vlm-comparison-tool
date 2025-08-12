import base64
import io
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
import gradio as gr
import requests
from dotenv import load_dotenv
from PIL import Image

# Setup comprehensive logging
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),  # Also log to console
    ],
)
logger = logging.getLogger(__name__)

# Try to import GPU monitoring
try:
    import GPUtil
    import psutil

    GPU_MONITORING_AVAILABLE = True
    logger.info("GPU and system monitoring available")
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    logger.info(
        "GPU monitoring not available - install GPUtil and psutil for detailed system metrics"
    )


def log_system_stats():
    """Log current system and GPU statistics"""
    if not GPU_MONITORING_AVAILABLE:
        return

    try:
        # CPU and Memory stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        logger.info(
            f"System Stats - CPU: {cpu_percent}%, Memory: {memory.percent}% ({memory.used/1024/1024/1024:.1f}GB used / {memory.total/1024/1024/1024:.1f}GB total)"
        )

        # GPU stats
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            logger.info(
                f"GPU {i} ({gpu.name}) - Utilization: {gpu.load*100:.1f}%, Memory: {gpu.memoryUtil*100:.1f}% ({gpu.memoryUsed}MB / {gpu.memoryTotal}MB), Temp: {gpu.temperature}°C"
            )
    except Exception as e:
        logger.warning(f"Failed to collect system stats: {e}")


def check_gpu_safety() -> Tuple[bool, str]:
    """Check if GPU is safe for use and detect OOM conditions"""
    if not GPU_MONITORING_AVAILABLE:
        return False, "GPU monitoring not available"

    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA not available"

        gpus = GPUtil.getGPUs()
        if not gpus:
            return False, "No GPUs detected"

        # Check primary GPU
        gpu = gpus[0]
        memory_total_gb = gpu.memoryTotal / 1024

        # Check for critical memory usage that could lead to OOM
        if gpu.memoryUtil > 0.9:
            return (
                False,
                f"GPU memory critically high: {gpu.memoryUtil*100:.1f}% - OOM risk",
            )

        # Special handling for 6GB GPUs - be more conservative
        if memory_total_gb <= 6.5:
            if gpu.memoryUtil > 0.8:
                return (
                    False,
                    f"6GB GPU memory high: {gpu.memoryUtil*100:.1f}% - OOM risk on smaller GPU",
                )

        return True, "GPU safe for use"

    except Exception as e:
        return False, f"Error checking GPU safety: {e}"


def force_cpu_mode():
    """Force PyTorch to use CPU mode"""
    try:
        # Set CUDA_VISIBLE_DEVICES to empty to hide GPUs from PyTorch
        import os

        import torch

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Also set torch to explicitly use CPU
        if hasattr(torch, "set_default_tensor_type"):
            torch.set_default_tensor_type(torch.FloatTensor)

        logger.info("🔄 FORCED CPU MODE: GPU disabled for processing")
        return True

    except Exception as e:
        logger.error(f"Failed to force CPU mode: {e}")
        return False


def log_detailed_gpu_stats(operation_name="operation"):
    """Log detailed GPU statistics for a specific operation"""
    if not GPU_MONITORING_AVAILABLE:
        logger.info(f"{operation_name} - GPU monitoring not available")
        return

    try:
        import torch

        logger.info(f"--- {operation_name.upper()} GPU DETAILS ---")

        # PyTorch CUDA info
        if torch.cuda.is_available():
            logger.info("CUDA Available: True")
            logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
            current_device = torch.cuda.current_device()
            logger.info(f"Current CUDA Device: {current_device}")

            # Memory stats for current device
            allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(current_device) / 1024**3  # GB
            max_allocated = (
                torch.cuda.max_memory_allocated(current_device) / 1024**3
            )  # GB

            logger.info(
                f"PyTorch GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max Allocated: {max_allocated:.2f}GB"
            )

            # Device properties
            props = torch.cuda.get_device_properties(current_device)
            logger.info(
                f"GPU Device: {props.name}, Total Memory: {props.total_memory / 1024**3:.2f}GB"
            )
        else:
            logger.info("CUDA Not Available - Running on CPU")

        # GPUtil stats
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            memory_used_gb = gpu.memoryUsed / 1024
            memory_total_gb = gpu.memoryTotal / 1024
            memory_free_gb = memory_total_gb - memory_used_gb

            logger.info(f"GPUtil GPU {i} ({gpu.name}):")
            logger.info(f"  - Utilization: {gpu.load*100:.1f}%")
            logger.info(
                f"  - Memory: {memory_used_gb:.2f}GB / {memory_total_gb:.2f}GB ({gpu.memoryUtil*100:.1f}%) - Free: {memory_free_gb:.2f}GB"
            )
            logger.info(f"  - Temperature: {gpu.temperature}°C")

            # Check for OOM risk with 6GB GPU
            if memory_total_gb <= 6.5:  # Account for some system overhead
                logger.info(f"*** DETECTED 6GB GPU: {gpu.name} ***")
                if gpu.memoryUtil > 0.9:
                    logger.warning(
                        f"🚨 CRITICAL: GPU {i} memory usage is {gpu.memoryUtil*100:.1f}% - OOM RISK!"
                    )
                elif gpu.memoryUtil > 0.8:
                    logger.warning(
                        f"⚠️  HIGH: GPU {i} memory usage is {gpu.memoryUtil*100:.1f}% - Monitor closely"
                    )
                elif gpu.memoryUtil > 0.6:
                    logger.info(
                        f"📊 MODERATE: GPU {i} memory usage is {gpu.memoryUtil*100:.1f}%"
                    )
                else:
                    logger.info(
                        f"✅ SAFE: GPU {i} memory usage is {gpu.memoryUtil*100:.1f}%"
                    )
            else:
                # Warn if memory usage is high on larger GPUs
                if gpu.memoryUtil > 0.9:
                    logger.warning(
                        f"GPU {i} memory usage is critically high: {gpu.memoryUtil*100:.1f}%"
                    )
                elif gpu.memoryUtil > 0.8:
                    logger.warning(
                        f"GPU {i} memory usage is high: {gpu.memoryUtil*100:.1f}%"
                    )

        logger.info(f"--- END {operation_name.upper()} GPU DETAILS ---")

    except Exception as e:
        logger.error(f"Failed to collect detailed GPU stats for {operation_name}: {e}")


# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Import our custom retrieval system
try:
    from retrieval import MultimodalRetriever, create_retriever

    RETRIEVAL_AVAILABLE = True
    logger.info("Retrieval system imported successfully")
except ImportError:
    RETRIEVAL_AVAILABLE = False
    logger.warning(
        "Retrieval system not available - multimodal retrieval will be disabled"
    )

# Global retriever instance
current_retriever = None

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")


def get_available_models() -> List[Dict]:
    """Fetch available models from OpenRouter API"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:7860",
            "X-Title": "VLM Comparison Tool",
        }

        response = requests.get(
            f"{OPENROUTER_BASE_URL}/models", headers=headers, timeout=15
        )

        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])

            # Filter for vision-capable models using OpenRouter's architecture field
            vlm_models = []
            for model in models:
                # Check architecture.input_modalities for "image" support
                architecture = model.get("architecture", {})
                input_modalities = architecture.get("input_modalities", [])
                if "image" in input_modalities:
                    vlm_models.append(model)

            # Remove the 50 model limit - return all VLMs
            return vlm_models
        else:
            logger.error(f"Error fetching models: HTTP {response.status_code}")
            return []

    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        return []


def extract_model_info(model: Dict) -> Dict:
    """Extract key model information for filtering and display"""
    model_id = model.get("id", "")
    name = model.get("name", model_id)

    # Extract pricing - OpenRouter returns strings in per-token format
    pricing = model.get("pricing", {})
    prompt_price_per_token = float(pricing.get("prompt", "0"))
    completion_price_per_token = float(pricing.get("completion", "0"))

    # Convert from per-token to per-million-tokens
    prompt_price = prompt_price_per_token * 1_000_000
    completion_price = completion_price_per_token * 1_000_000

    # Extract context length
    context_length = model.get("context_length", 0)
    if not context_length:
        # Try alternative fields
        context_length = model.get("max_context_tokens", 0)
        if not context_length:
            # Default fallback based on model patterns
            model_lower = model_id.lower()
            if any(pattern in model_lower for pattern in ["gpt-4", "claude-3"]):
                context_length = 128000  # Common for newer models
            elif "gemini" in model_lower:
                context_length = 200000
            else:
                context_length = 32000  # Reasonable default

    # Extract provider
    provider = model_id.split("/")[0] if "/" in model_id else "unknown"

    return {
        "id": model_id,
        "name": name,
        "provider": provider,
        "prompt_price": prompt_price,
        "completion_price": completion_price,
        "context_length": context_length,
        "model_data": model,
    }


def filter_models(
    models: List[Dict],
    search_term: str = "",
    selected_providers: List[str] = None,
) -> List[Dict]:
    """Filter models based on search criteria"""
    if selected_providers is None:
        selected_providers = []

    filtered = []

    for model_info in models:
        # Search filter
        if search_term.strip():
            search_lower = search_term.lower()
            if not (
                search_lower in model_info["name"].lower()
                or search_lower in model_info["id"].lower()
                or search_lower in model_info["provider"].lower()
            ):
                continue

        # Provider filter
        if selected_providers and model_info["provider"] not in selected_providers:
            continue

        filtered.append(model_info)

    # Sort by provider first, then by model name
    filtered.sort(key=lambda x: (x["provider"].lower(), x["name"].lower()))

    return filtered


def format_model_choice(model_info: Dict) -> Tuple[str, str]:
    """Format model info for dropdown display"""
    name = model_info["name"]
    model_id = model_info["id"]
    context_k = model_info["context_length"] // 1000
    prompt_price = model_info["prompt_price"]

    # Format context length
    if context_k >= 200:
        context_str = f"{context_k}K+"
    else:
        context_str = f"{context_k}K"

    # Format price
    if prompt_price == 0:
        price_str = "Free"
    elif prompt_price < 1:
        price_str = f"${prompt_price:.3f}/1M"
    else:
        price_str = f"${prompt_price:.2f}/1M"

    display_text = f"{name} | {context_str} | {price_str} | {model_id}"

    return (display_text, model_id)


def pdf_page_to_base64(pdf_file, page_num):
    """Convert a specific PDF page to base64 encoded image"""
    try:
        # Handle different input types from Gradio
        if hasattr(pdf_file, "read"):
            pdf_data = pdf_file.read()
        else:
            pdf_data = pdf_file

        # Open PDF
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

        # Check if page exists
        if page_num < 1 or page_num > len(pdf_document):
            return (
                None,
                f"Page {page_num} not found. PDF has {len(pdf_document)} pages.",
            )

        # Get the page (0-indexed)
        page = pdf_document.load_page(page_num - 1)

        # Convert page to image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")

        # Convert to PIL Image for display
        pil_image = Image.open(io.BytesIO(img_data))

        # Convert to base64 for API
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        pdf_document.close()
        return img_base64, pil_image, None

    except Exception as e:
        return None, None, f"Error processing PDF: {str(e)}"


def query_openrouter_model(model_id: str, question: str, img_base64: str) -> str:
    """Send query to OpenRouter model"""
    logger.info(f"Querying OpenRouter model: {model_id}")
    start_time = time.time()

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:7860",  # For OpenRouter analytics
            "X-Title": "VLM Comparison Tool",
        }

        # Format according to OpenRouter docs - text first, then image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                ],
            }
        ]

        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 0.7,
        }

        logger.info(
            f"Sending request to OpenRouter - Payload size: {len(json.dumps(payload))} chars"
        )

        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )

        request_time = time.time() - start_time
        logger.info(f"OpenRouter request completed in {request_time:.2f} seconds")

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                logger.info(
                    f"Successful response from {model_id} - Length: {len(content)} chars"
                )
                return content
            else:
                logger.error(f"No response content found for {model_id}")
                return "No response content found"
        else:
            error_msg = f"HTTP {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg += (
                        f": {error_data['error'].get('message', 'Unknown error')}"
                    )
            except Exception:
                error_msg += f": {response.text[:200]}"
            logger.error(f"OpenRouter API error for {model_id}: {error_msg}")
            return f"Error: {error_msg}"

    except requests.exceptions.Timeout:
        logger.error(f"Request timeout for {model_id}")
        return "Error: Request timed out (60s)"
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception for {model_id}: {e}")
        return f"Error: {str(e)}"
    except json.JSONDecodeError:
        logger.error(f"JSON decode error for {model_id}")
        return "Error: Invalid JSON response"


def compare_models(
    pdf_file, page_num, question, model_a_id, model_b_id, use_retrieval=True
):
    """Main function to compare both models - supports retrieval with page display fallback"""
    global current_retriever

    logger.info(f"Starting model comparison - Models: {model_a_id} vs {model_b_id}")
    logger.info(f"Question: '{question}'")
    logger.info(f"Use retrieval: {use_retrieval}, Page: {page_num}")
    log_system_stats()

    if not pdf_file:
        error_msg = "Please upload a PDF file"
        logger.warning("No PDF file provided")
        return (
            error_msg,
            error_msg,
            error_msg,
            True,
            "*No PDF uploaded*",
        )  # Show status on error

    if not question.strip():
        error_msg = "Please enter a question"
        logger.warning("No question provided")
        return (
            error_msg,
            error_msg,
            error_msg,
            True,
            "*No question provided*",
        )  # Show status on error

    if not model_a_id or not model_b_id:
        error_msg = "Please select both models"
        logger.warning("Missing model selection")
        return (
            error_msg,
            error_msg,
            error_msg,
            True,
            "*Missing model selection*",
        )  # Show status on error

    try:
        start_time = time.time()
        # Determine if we should use retrieval or page display fallback
        use_retrieval_mode = use_retrieval and current_retriever is not None
        logger.info(f"Retrieval mode: {use_retrieval_mode}")

        retrieval_display_text = ""

        if use_retrieval_mode:
            # Use multimodal retrieval to find relevant pages
            logger.info("Using multimodal retrieval to find relevant pages")
            relevant_pages, retrieval_display_text = retrieve_relevant_pages(
                question, k=3
            )

            if not relevant_pages:
                # Fall back to page display
                logger.info(f"No relevant pages found, falling back to page {page_num}")
                img_base64, pil_image, error = pdf_page_to_base64(pdf_file, page_num)
                if error:
                    logger.error(f"Error converting page to base64: {error}")
                    return error, error, error, True, retrieval_display_text
                status_msg = f"No relevant pages found via retrieval. Using page {page_num} instead."
                retrieval_display_text = "*No relevant pages found for this query. Using current page instead.*"
            else:
                # Use the most relevant page (first result)
                img_bytes, score = relevant_pages[0]
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                status_msg = f"Using most relevant page (similarity: {score:.3f})"
                logger.info(f"Using retrieved page with similarity score: {score:.4f}")

        else:
            # Use current page display
            logger.info(f"Using page display mode - page {page_num}")
            img_base64, pil_image, error = pdf_page_to_base64(pdf_file, page_num)
            if error:
                logger.error(f"Error converting page to base64: {error}")
                return error, error, error, True, "*Error processing page*"
            status_msg = f"Using currently displayed page {page_num}"
            retrieval_display_text = (
                f"*Using page display mode - currently showing page {page_num}*"
            )

        # Query both models
        logger.info("Querying models...")
        model_a_start = time.time()
        response_a = query_openrouter_model(model_a_id, question, img_base64)
        model_a_time = time.time() - model_a_start

        model_b_start = time.time()
        response_b = query_openrouter_model(model_b_id, question, img_base64)
        model_b_time = time.time() - model_b_start

        total_time = time.time() - start_time

        logger.info(f"Model A ({model_a_id}) response time: {model_a_time:.2f}s")
        logger.info(f"Model B ({model_b_id}) response time: {model_b_time:.2f}s")
        logger.info(f"Total comparison time: {total_time:.2f}s")
        logger.info(f"Model A response length: {len(response_a)} chars")
        logger.info(f"Model B response length: {len(response_b)} chars")
        log_system_stats()

        # Check if responses contain errors
        if "Error:" in response_a or "Error:" in response_b:
            status_msg = "One or more models returned an error"
            return (
                response_a,
                response_b,
                status_msg,
                True,
                retrieval_display_text,
            )  # Show status on error

        # Success - show status with page info
        return (
            response_a,
            response_b,
            status_msg,
            True,
            retrieval_display_text,
        )  # Show status for page info

    except Exception as e:
        error_msg = f"Error during comparison: {str(e)}"
        return (
            error_msg,
            error_msg,
            error_msg,
            True,
            "*Error during comparison*",
        )  # Show status on error


def preview_pdf_page(pdf_file, page_num):
    """Preview a specific PDF page for user browsing"""
    if not pdf_file:
        return None

    try:
        # Handle different input types from Gradio
        if hasattr(pdf_file, "read"):
            pdf_data = pdf_file.read()
        else:
            pdf_data = pdf_file

        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

        # Check if page exists
        if page_num < 1 or page_num > len(pdf_document):
            pdf_document.close()
            return None

        # Get the page (0-indexed)
        page = pdf_document.load_page(page_num - 1)

        # Convert page to image for preview
        mat = fitz.Matrix(1.5, 1.5)  # 1.5x zoom for good preview quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")

        # Convert to PIL Image for display
        pil_image = Image.open(io.BytesIO(img_data))

        pdf_document.close()
        return pil_image

    except Exception:
        return None


def get_pdf_info_and_preview(pdf_file):
    """Get basic info about uploaded PDF and show first page preview"""
    if not pdf_file:
        return "No PDF uploaded", None

    try:
        # Handle different input types from Gradio
        if hasattr(pdf_file, "read"):
            pdf_data = pdf_file.read()
        else:
            pdf_data = pdf_file

        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        num_pages = len(pdf_document)

        # Get first page for preview
        if num_pages > 0:
            page = pdf_document.load_page(0)  # First page
            mat = fitz.Matrix(1.5, 1.5)  # 1.5x zoom for good preview quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
        else:
            pil_image = None

        pdf_document.close()
        info_text = f"PDF loaded successfully. Total pages: {num_pages}"
        return info_text, pil_image
    except Exception as e:
        return f"Error reading PDF: {str(e)}", None


def initialize_retrieval_model(model_name, progress=gr.Progress()):
    """Initialize the multimodal retrieval model"""
    global current_retriever

    logger.info(f"Starting retrieval model initialization: {model_name}")
    log_detailed_gpu_stats("model_initialization_start")

    if not RETRIEVAL_AVAILABLE:
        logger.error("Retrieval system not available")
        return (
            "Error: Retrieval system not available",
            True,
        )  # error

    if not model_name or model_name == "None":
        current_retriever = None
        logger.info("No retrieval model selected - using page display fallback")
        return (
            "No retrieval model selected - page display will be used instead",
            False,
        )  # show page selection

    try:
        start_time = time.time()
        progress(0.1, desc="Creating retriever...")
        logger.info(f"Creating retriever for model: {model_name}")

        # Check GPU safety before attempting to load model
        gpu_safe, gpu_status = check_gpu_safety()
        if not gpu_safe:
            logger.warning(f"GPU not safe for use: {gpu_status}")
            logger.info("🔄 Switching to CPU mode for safety")
            force_cpu_mode()
            progress(0.2, desc="GPU unsafe, switching to CPU mode...")

        current_retriever = create_retriever(model_name)

        if not current_retriever:
            logger.error(f"Failed to create retriever for model: {model_name}")
            return (
                f"Error: Unsupported model {model_name}",
                True,
            )  # error

        progress(0.3, desc="Initializing model...")
        logger.info("Initializing retrieval model...")
        log_detailed_gpu_stats("model_initialization_before_load")

        def progress_callback(message, status):
            logger.info(f"Model initialization - {status}: {message}")
            if status == "downloading":
                progress(0.5, desc=message)
                log_detailed_gpu_stats("model_downloading")
            elif status == "ready":
                progress(0.8, desc=message)
                log_detailed_gpu_stats("model_ready")
            elif status == "error":
                progress(1.0, desc=f"Error: {message}")
            elif status == "oom_fallback":
                progress(0.6, desc=f"GPU OOM detected, retrying on CPU: {message}")

        success = current_retriever.initialize_model(progress_callback)

        initialization_time = time.time() - start_time
        logger.info(
            f"Model initialization completed in {initialization_time:.2f} seconds"
        )
        log_detailed_gpu_stats("model_initialization_complete")

        if success:
            progress(1.0, desc="Model ready for indexing!")
            logger.info("Retrieval model initialized successfully")

            # Check if we ended up on CPU due to GPU issues
            status = current_retriever.get_status()
            device_info = status.get("device_info", {})
            if device_info.get("using_cpu", False):
                return (
                    "Model initialized successfully on CPU (GPU fallback). Upload a PDF to start indexing.",
                    False,  # no error
                )
            else:
                return (
                    "Model initialized successfully. Upload a PDF to start indexing.",
                    False,  # no error
                )
        else:
            current_retriever = None
            logger.error("Model initialization failed")
            return "Error initializing model", True  # error

    except Exception as e:
        current_retriever = None
        logger.error(f"Exception during model initialization: {e}", exc_info=True)
        log_detailed_gpu_stats("model_initialization_error")

        # Check if this was an OOM error and offer to retry on CPU
        error_str = str(e).lower()
        if any(
            oom_indicator in error_str
            for oom_indicator in ["out of memory", "cuda out of memory", "oom"]
        ):
            logger.info(
                "🔄 OOM detected during initialization, attempting CPU fallback"
            )
            try:
                force_cpu_mode()
                progress(0.5, desc="GPU OOM detected, retrying on CPU...")
                current_retriever = create_retriever(model_name)
                if current_retriever:
                    success = current_retriever.initialize_model(progress_callback)
                    if success:
                        logger.info("✅ Successfully recovered using CPU fallback")
                        return (
                            "Model initialized successfully on CPU (GPU OOM recovery). Upload a PDF to start indexing.",
                            False,
                        )
            except Exception as cpu_error:
                logger.error(f"CPU fallback also failed: {cpu_error}")

        return f"Error: {str(e)}", True  # error


def process_pdf_for_retrieval(pdf_file, progress=gr.Progress()):
    """Process PDF with multimodal retrieval"""
    global current_retriever

    logger.info("Starting PDF processing for retrieval")
    log_detailed_gpu_stats("pdf_indexing_start")

    if not current_retriever:
        logger.info("No retriever available, falling back to regular preview")
        return get_pdf_info_and_preview(pdf_file)  # Fall back to regular preview

    if not pdf_file:
        logger.warning("No PDF file provided")
        return "No PDF uploaded", None

    try:
        start_time = time.time()
        progress(0.1, desc="Starting PDF indexing...")
        logger.info("Beginning PDF indexing process")

        # Get PDF data
        if hasattr(pdf_file, "read"):
            pdf_data = pdf_file.read()
        else:
            pdf_data = pdf_file

        pdf_size_mb = len(pdf_data) / (1024 * 1024)
        logger.info(f"PDF size: {pdf_size_mb:.2f} MB")

        def progress_callback(message, status):
            logger.info(f"PDF processing - {status}: {message}")
            if status == "indexing":
                progress(0.3, desc=message)
            elif status == "complete":
                progress(1.0, desc=message)
            elif status == "error":
                progress(1.0, desc=f"Error: {message}")

        # Index the PDF
        logger.info("Starting PDF indexing...")
        logger.info(f"PDF size: {pdf_size_mb:.2f} MB - Starting indexing process")
        indexing_start_time = time.time()

        success = current_retriever.index_pdf(pdf_data, progress_callback)

        indexing_time = time.time() - start_time
        pure_indexing_time = time.time() - indexing_start_time
        logger.info(
            f"PDF indexing completed in {indexing_time:.2f} seconds (pure indexing: {pure_indexing_time:.2f}s)"
        )
        log_detailed_gpu_stats("pdf_indexing_complete")

        if success:
            # Get basic PDF info
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            num_pages = len(pdf_document)
            logger.info(f"PDF successfully indexed - {num_pages} pages processed")

            # Get status for timing info
            status = current_retriever.get_status()
            timing = status.get("timing", {})
            gpu_usage = status.get("gpu_usage", {})
            device_info = status.get("device_info", {})

            logger.info(f"Retrieval system status: {status}")

            device_mode = device_info.get("device_mode", "GPU")

            timing_info = f"Indexing completed on {device_mode} in {timing.get('indexing_time', indexing_time):.2f}s"
            if gpu_usage.get("total_mb", 0) > 0:
                timing_info += f" (GPU: {gpu_usage['used_mb']:.0f}MB/{gpu_usage['total_mb']:.0f}MB)"
                logger.info(
                    f"GPU Usage - Used: {gpu_usage['used_mb']:.0f}MB, Total: {gpu_usage['total_mb']:.0f}MB"
                )

            info_text = f"PDF indexed successfully. Pages: {num_pages}. {timing_info}"  # Show first page as preview
            if num_pages > 0:
                page = pdf_document.load_page(0)
                mat = fitz.Matrix(1.5, 1.5)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))
            else:
                pil_image = None

            pdf_document.close()
            return info_text, pil_image
        else:
            return "Error indexing PDF", None

    except Exception as e:
        return f"Error processing PDF: {str(e)}", None


def process_pdf_with_status(pdf_file):
    """Process PDF and return HTML formatted status with timing info"""
    if not pdf_file:
        return "<i>No PDF uploaded</i>", None, "", False

    # Get filename from uploaded file
    filename = (
        getattr(pdf_file, "name", "uploaded.pdf")
        if hasattr(pdf_file, "name")
        else "uploaded.pdf"
    )

    # Process the PDF (this will also handle the indexing if retrieval model is available)
    start_time = time.time()
    info_text, preview_image = process_pdf_for_retrieval(pdf_file)
    total_processing_time = time.time() - start_time

    # Extract indexing time from the info_text if available
    indexing_time_html = ""
    show_timing = False

    if "Indexing completed in" in info_text:
        # Extract the timing information to display prominently
        import re

        timing_match = re.search(r"Indexing completed in ([\d.]+)s", info_text)
        gpu_match = re.search(r"\(GPU: ([\d.]+)MB/([\d.]+)MB\)", info_text)

        if timing_match:
            indexing_time = float(timing_match.group(1))
            show_timing = True

            # Create prominent timing display
            if gpu_match:
                gpu_used = gpu_match.group(1)
                gpu_total = gpu_match.group(2)
                gpu_percent = (float(gpu_used) / float(gpu_total)) * 100

                indexing_time_html = f"""
                <div style='background: linear-gradient(90deg, #4CAF50, #45a049); padding: 15px; border-radius: 10px; margin: 10px 0; color: white; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2);'>
                    <h3 style='margin: 0; font-size: 18px;'>⚡ Indexing Performance</h3>
                    <div style='font-size: 24px; font-weight: bold; margin: 8px 0;'>{indexing_time:.2f} seconds</div>
                    <div style='font-size: 14px; opacity: 0.9;'>
                        GPU Memory: {gpu_used}MB / {gpu_total}MB ({gpu_percent:.1f}%)
                    </div>
                </div>
                """
            else:
                indexing_time_html = f"""
                <div style='background: linear-gradient(90deg, #4CAF50, #45a049); padding: 15px; border-radius: 10px; margin: 10px 0; color: white; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2);'>
                    <h3 style='margin: 0; font-size: 18px;'>⚡ Indexing Performance</h3>
                    <div style='font-size: 24px; font-weight: bold; margin: 8px 0;'>{indexing_time:.2f} seconds</div>
                </div>
                """
    elif current_retriever is None:
        # No indexing performed (no retrieval model)
        indexing_time_html = f"""
        <div style='background: linear-gradient(90deg, #2196F3, #1976D2); padding: 15px; border-radius: 10px; margin: 10px 0; color: white; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2);'>
            <h3 style='margin: 0; font-size: 18px;'>📄 PDF Processing</h3>
            <div style='font-size: 24px; font-weight: bold; margin: 8px 0;'>{total_processing_time:.2f} seconds</div>
            <div style='font-size: 14px; opacity: 0.9;'>No indexing - page display mode</div>
        </div>
        """
        show_timing = True
    elif "Error" in info_text:
        # Error during indexing
        indexing_time_html = """
        <div style='background: linear-gradient(90deg, #f44336, #d32f2f); padding: 15px; border-radius: 10px; margin: 10px 0; color: white; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2);'>
            <h3 style='margin: 0; font-size: 18px;'>❌ Indexing Error</h3>
            <div style='font-size: 14px; opacity: 0.9;'>Check logs for details</div>
        </div>
        """
        show_timing = True

    # Format file status as HTML
    if "Error" in info_text:
        html_status = f"<span style='color: red;'>📄 {filename} - {info_text}</span>"
    else:
        html_status = f"<span style='color: green;'>📄 {filename} - {info_text}</span>"

    return html_status, preview_image, indexing_time_html, show_timing


def retrieve_relevant_pages(query, k=3):
    """Retrieve relevant pages using multimodal search"""
    global current_retriever

    logger.info(f"Starting page retrieval - Query: '{query}', k={k}")
    log_detailed_gpu_stats("retrieval_start")

    if not current_retriever or not query.strip():
        logger.warning("No retriever available or empty query")
        return [], ""

    try:
        start_time = time.time()
        log_detailed_gpu_stats("retrieval_before_search")
        results = current_retriever.search(query, k=k)
        retrieval_time = time.time() - start_time

        logger.info(f"Page retrieval completed in {retrieval_time:.2f} seconds")
        logger.info(f"Retrieved {len(results)} relevant pages")

        # Create formatted display text
        if not results:
            formatted_results = "*No relevant pages found for this query.*"
        else:
            formatted_results = f"**Found {len(results)} relevant page(s) in {retrieval_time:.2f}s:**\n\n"
            for i, (img_bytes, score) in enumerate(results):
                # Try to determine page number if possible
                # Note: This is a simplified approach - in practice you'd want to store page metadata
                page_info = f"Page {i+1}"  # Placeholder - actual page numbers would need to be tracked during indexing
                formatted_results += (
                    f"**{i+1}.** {page_info} - Similarity: {score:.3f}\n"
                )

        # Log detailed results
        for i, (img_bytes, score) in enumerate(results):
            logger.info(
                f"Result {i+1}: Similarity score {score:.4f}, Image size: {len(img_bytes)} bytes"
            )

        log_detailed_gpu_stats("retrieval_complete")
        return results, formatted_results
    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        log_detailed_gpu_stats("retrieval_error")
        return [], f"*Error during retrieval: {str(e)}*"


# Initialize models list
logger.info("Fetching available VLM models from OpenRouter...")
raw_models = get_available_models()

# Process models for filtering
all_model_info = [extract_model_info(model) for model in raw_models]

# Get unique providers for filter checkboxes
all_providers = sorted(list(set(model["provider"] for model in all_model_info)))

# Initial model choices (all models, sorted by price)
initial_filtered = filter_models(all_model_info)
initial_choices = [format_model_choice(model) for model in initial_filtered]

if not initial_choices:
    initial_choices = [("No VLM models available - check API key", "")]
    logger.warning("No VLM models fetched from OpenRouter API")
else:
    logger.info(
        f"Found {len(all_model_info)} vision-capable models from {len(all_providers)} providers"
    )


# PDF navigation functions
def get_pdf_page_count(pdf_file):
    """Get the total number of pages in a PDF"""
    if not pdf_file:
        return 1

    try:
        # Handle different input types from Gradio
        if hasattr(pdf_file, "read"):
            pdf_data = pdf_file.read()
        else:
            pdf_data = pdf_file

        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        page_count = len(pdf_document)
        pdf_document.close()
        return max(1, page_count)
    except Exception:
        return 1


def navigate_to_previous_page(current_page, pdf_file):
    """Navigate to the previous page"""
    return max(1, current_page - 1)


def navigate_to_next_page(current_page, pdf_file):
    """Navigate to the next page"""
    max_pages = get_pdf_page_count(pdf_file)
    return min(max_pages, current_page + 1)


# Custom CSS for better layout
custom_css = """
.gr-group .prose {
    padding: 12px !important;
}
"""

# Create Gradio interface
with gr.Blocks(
    title="VLM Comparison Tool", theme=gr.themes.Citrus(), css=custom_css
) as demo:
    gr.Markdown("# 🤖 Vision Language Model Comparison Tool")
    gr.Markdown(
        "Select any VLM available on OpenRouter, upload a PDF, and compare responses."
    )

    with gr.Row():
        with gr.Column(scale=1):
            # Multimodal Retrieval Model Section
            gr.Markdown("## 1. Select multimodal retrieval model")
            retrieval_model_dropdown = gr.Dropdown(
                choices=["None"]
                + (MultimodalRetriever.SUPPORTED_MODELS if RETRIEVAL_AVAILABLE else []),
                label="Choose multimodal retrieval model",
                value="None",
                interactive=True,
                info="Select a model for intelligent page retrieval, or 'None' to use page display",
            )

            retrieval_status = gr.Textbox(
                label="Retrieval Status",
                value="No retrieval model selected",
                interactive=False,
                visible=False,
            )

            # Response Generation Models Section
            gr.Markdown("## 2. Select response generation model")

            with gr.Row():
                with gr.Column(scale=1):
                    provider_checkboxes = gr.CheckboxGroup(
                        choices=all_providers,
                        label="Providers (leave empty to see all)",
                        value=[],
                        interactive=True,
                    )

                    # Model count display
                    model_count_display = gr.Textbox(
                        value=f"Showing {len(initial_choices)} of {len(all_model_info)} models",
                        interactive=False,
                        show_label=False,
                        container=False,
                    )

                    refresh_models_btn = gr.Button(
                        "🔄 Refresh Models", variant="secondary"
                    )

                with gr.Column(scale=1):
                    model_a_dropdown = gr.Dropdown(
                        choices=initial_choices,
                        label="Model A",
                        value=initial_choices[0][1]
                        if initial_choices and initial_choices[0][1]
                        else None,
                        interactive=True,
                        filterable=True,
                    )
                    model_b_dropdown = gr.Dropdown(
                        choices=initial_choices,
                        label="Model B",
                        value=initial_choices[1][1]
                        if len(initial_choices) > 1 and initial_choices[1][1]
                        else None,
                        interactive=True,
                        filterable=True,
                    )

            # PDF Upload and Processing Section
            gr.Markdown("## 3. PDF upload & processing")
            pdf_input = gr.File(
                label="📄 Upload PDF",
                file_types=[".pdf"],
                type="binary",
                container=False,
            )

            # Filename display (will be updated with actual filename when uploaded)
            pdf_status = gr.HTML(value="<i>No PDF uploaded</i>")

            # Indexing time display (prominent display for performance monitoring)
            indexing_time_display = gr.HTML(
                value="", visible=False, label="Indexing Performance"
            )

            gr.Markdown("## 4. Send query")

            question_input = gr.Textbox(
                label="Enter Query",
                placeholder="Ask a question about this PDF...",
                lines=3,
            )

            compare_btn = gr.Button("⚡ Compare Models", variant="primary", size="lg")

        with gr.Column(scale=1):
            # PDF Preview Section - Now gets full right column
            gr.Markdown("## 📖 PDF Preview")
            pdf_display = gr.Image(
                label="PDF Preview - Retrieval mode will find relevant pages automatically",
                type="pil",
                height=600,  # Increased height to use more space
            )

            # PDF Navigation Controls
            gr.Markdown("### 📄 PDF Navigation")
            with gr.Row():
                prev_page_btn = gr.Button("← Previous", size="sm")
                page_display = gr.Number(
                    label="Current Page",
                    show_label=False,
                    value=1,
                    minimum=1,
                    step=1,
                    interactive=True,
                    scale=2,
                )
                next_page_btn = gr.Button("Next →", size="sm")

            # Retrieval Results Display
            gr.Markdown("### 🔍 Retrieved Pages")
            retrieval_results_display = gr.Markdown(
                value="*No retrieval performed yet. Pages returned by the retrieval model will appear here.*",
                label="Retrieved Pages Information",
                elem_classes=["retrieval-results"],
            )

    # Function to clear retrieval results when switching modes
    def clear_retrieval_results():
        return "*No retrieval model selected - page display mode active*"

    # Function to reset displays when no PDF
    def reset_displays_on_empty():
        return "<i>No PDF uploaded</i>", None, "", False

    # Results Section
    gr.Markdown("## 📊 Comparison Results")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 Model A Response")
            with gr.Group():
                response_a = gr.Markdown(
                    value="Model A response will appear here...", height=400
                )

        with gr.Column(scale=1):
            gr.Markdown("### 🤖 Model B Response")
            with gr.Group():
                response_b = gr.Markdown(
                    value="Model B response will appear here...", height=400
                )

    # Helper function to refresh models
    def refresh_models():
        global all_model_info, all_providers
        new_raw_models = get_available_models()
        all_model_info = [extract_model_info(model) for model in new_raw_models]
        all_providers = sorted(list(set(model["provider"] for model in all_model_info)))

        new_filtered = filter_models(all_model_info)
        new_choices = [format_model_choice(model) for model in new_filtered]

        if not new_choices:
            new_choices = [("No models available - check API key", "")]

        return (
            gr.Dropdown(
                choices=new_choices,
                value=new_choices[0][1] if new_choices and new_choices[0][1] else None,
            ),
            gr.Dropdown(
                choices=new_choices,
                value=new_choices[1][1]
                if len(new_choices) > 1 and new_choices[1][1]
                else None,
            ),
            gr.CheckboxGroup(choices=all_providers, value=[]),
            f"Refreshed: Found {len(all_model_info)} models from {len(all_providers)} providers",
        )

    # Helper function to apply filters
    def apply_filters(selected_providers):
        filtered_models = filter_models(
            all_model_info,
            search_term="",  # No search term anymore
            selected_providers=selected_providers if selected_providers else [],
        )

        filtered_choices = [format_model_choice(model) for model in filtered_models]

        if not filtered_choices:
            filtered_choices = [("No models match your criteria", "")]

        count_msg = f"Showing {len(filtered_choices)} of {len(all_model_info)} models"

        return (
            gr.Dropdown(choices=filtered_choices, value=None),  # Reset selections
            gr.Dropdown(choices=filtered_choices, value=None),
            count_msg,
        )

    # Helper function to get model name from ID
    def get_model_name(model_id):
        if not model_id:
            return "Not Selected"
        for model_info in all_model_info:
            if model_info["id"] == model_id:
                return model_info["name"]
        return model_id.split("/")[-1]  # fallback to last part of ID

    # Wrapper function to handle comparison
    def compare_with_status_visibility(
        pdf_file, page_num, question, model_a_id, model_b_id
    ):
        # Determine if we should use retrieval based on current_retriever state
        use_retrieval = current_retriever is not None

        response_a, response_b, status_msg, show_status, retrieval_results = (
            compare_models(
                pdf_file, page_num, question, model_a_id, model_b_id, use_retrieval
            )
        )

        return (response_a, response_b, retrieval_results)

    # Event handlers
    pdf_input.change(
        fn=process_pdf_with_status,
        inputs=[pdf_input],
        outputs=[
            pdf_status,
            pdf_display,
            indexing_time_display,
            indexing_time_display,
        ],  # Last one controls visibility
    )

    # PDF page preview when page display changes
    page_display.change(
        fn=preview_pdf_page, inputs=[pdf_input, page_display], outputs=[pdf_display]
    )

    # Navigation button handlers - update both display and input
    def navigate_prev_and_sync(current_page, pdf_file):
        new_page = navigate_to_previous_page(current_page, pdf_file)
        return new_page

    def navigate_next_and_sync(current_page, pdf_file):
        new_page = navigate_to_next_page(current_page, pdf_file)
        return new_page

    prev_page_btn.click(
        fn=navigate_prev_and_sync,
        inputs=[page_display, pdf_input],
        outputs=[page_display],
    )

    next_page_btn.click(
        fn=navigate_next_and_sync,
        inputs=[page_display, pdf_input],
        outputs=[page_display],
    )

    refresh_models_btn.click(
        fn=refresh_models,
        outputs=[
            model_a_dropdown,
            model_b_dropdown,
            provider_checkboxes,
            model_count_display,
        ],
    )

    # Retrieval model initialization
    retrieval_model_dropdown.change(
        fn=initialize_retrieval_model,
        inputs=[retrieval_model_dropdown],
        outputs=[
            retrieval_status,
            retrieval_status,  # Controls visibility of retrieval_status
        ],
    )

    # Clear retrieval results when changing retrieval model
    retrieval_model_dropdown.change(
        fn=clear_retrieval_results,
        outputs=[retrieval_results_display],
    )

    # Real-time filtering events (simplified - only provider filter now)
    provider_checkboxes.change(
        fn=apply_filters,
        inputs=[provider_checkboxes],
        outputs=[model_a_dropdown, model_b_dropdown, model_count_display],
    )

    compare_btn.click(
        fn=compare_with_status_visibility,
        inputs=[
            pdf_input,
            page_display,
            question_input,
            model_a_dropdown,
            model_b_dropdown,
        ],
        outputs=[response_a, response_b, retrieval_results_display],
    )

    # Allow Enter key to trigger comparison
    question_input.submit(
        fn=compare_with_status_visibility,
        inputs=[
            pdf_input,
            page_display,
            question_input,
            model_a_dropdown,
            model_b_dropdown,
        ],
        outputs=[response_a, response_b, retrieval_results_display],
    )

if __name__ == "__main__":
    logger.info("=== VLM COMPARISON TOOL STARTUP ===")
    logger.info(f"Log file: {log_filename}")
    log_detailed_gpu_stats("application_startup")
    logger.info("Starting Gradio application...")

    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public sharing
    )
