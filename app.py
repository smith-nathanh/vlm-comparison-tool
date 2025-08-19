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

    GPU_MONITORING_AVAILABLE = True
    logger.info("GPU monitoring available")
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    logger.info("GPU monitoring not available")


def check_high_gpu_usage():
    """Check GPU usage and warn if it nears 90%"""
    if not GPU_MONITORING_AVAILABLE:
        return

    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            if gpu.memoryUtil >= 0.9:
                logger.warning(
                    f"GPU {i} ({gpu.name}) memory usage high: {gpu.memoryUtil*100:.1f}%"
                )
    except Exception as e:
        logger.debug(f"Failed to check GPU usage: {e}")


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

# Indexing state tracking
indexing_state = {
    "status": "not_started",  # not_started, in_progress, success, failed
    "last_error": None,
    "last_attempt_time": None,
    "model_failed": False,  # Track if model initialization failed
    "model_initializing": False,  # Track if model is currently initializing
}

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
    check_high_gpu_usage()

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
                img_bytes, score, page_num = relevant_pages[0]
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                status_msg = f"Using most relevant page {page_num} (similarity: {score:.3f})"
                logger.info(f"Using retrieved page {page_num} with similarity score: {score:.4f}")

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

        # Create timing displays
        timing_a_html = f"""
        <div style='background: #e3f2fd; padding: 8px; border-radius: 4px; margin-top: 8px; text-align: center;'>
            ⏱️ Response time: <strong>{model_a_time:.2f}s</strong> | Length: {len(response_a)} chars
        </div>
        """
        
        timing_b_html = f"""
        <div style='background: #e8f5e8; padding: 8px; border-radius: 4px; margin-top: 8px; text-align: center;'>
            ⏱️ Response time: <strong>{model_b_time:.2f}s</strong> | Length: {len(response_b)} chars
        </div>
        """

        # Check if responses contain errors
        if "Error:" in response_a or "Error:" in response_b:
            status_msg = "One or more models returned an error"
            return (
                response_a,
                response_b,
                status_msg,
                True,
                retrieval_display_text,
                timing_a_html,
                timing_b_html,
            )  # Show status on error

        # Success - show status with page info
        return (
            response_a,
            response_b,
            status_msg,
            True,
            retrieval_display_text,
            timing_a_html,
            timing_b_html,
        )  # Show status for page info

    except Exception as e:
        error_msg = f"Error during comparison: {str(e)}"
        error_timing = f"""
        <div style='background: #ffebee; padding: 8px; border-radius: 4px; margin-top: 8px; text-align: center; color: #c62828;'>
            ❌ Error occurred during processing
        </div>
        """
        return (
            error_msg,
            error_msg,
            error_msg,
            True,
            "*Error during comparison*",
            error_timing,
            error_timing,
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
    global current_retriever, indexing_state

    logger.info(f"Starting retrieval model initialization: {model_name}")
    check_high_gpu_usage()

    if not RETRIEVAL_AVAILABLE:
        logger.error("Retrieval system not available")
        return (
            "Error: Retrieval system not available",
            True,
        )  # error

    if not model_name or model_name == "None":
        current_retriever = None
        # Reset indexing state when no model is selected
        indexing_state["status"] = "not_started"
        indexing_state["last_error"] = None
        indexing_state["last_attempt_time"] = None
        indexing_state["model_failed"] = False
        indexing_state["model_initializing"] = False
        logger.info("No retrieval model selected - using page display fallback")
        return (
            "No retrieval model selected - page display will be used instead",
            False,
        )  # show page selection

    try:
        # Mark model as initializing
        indexing_state["model_initializing"] = True

        start_time = time.time()
        progress(0.1, desc="Creating retriever...")
        logger.info(f"Creating retriever for model: {model_name}")

        current_retriever = create_retriever(model_name)

        if not current_retriever:
            logger.error(f"Failed to create retriever for model: {model_name}")
            return (
                f"Error: Unsupported model {model_name}",
                True,
            )  # error

        progress(0.3, desc="Initializing model...")
        logger.info("Initializing retrieval model...")

        def progress_callback(message, status):
            logger.info(f"Model initialization - {status}: {message}")
            if status == "downloading":
                progress(0.5, desc=message)
            elif status == "ready":
                progress(0.8, desc=message)
            elif status == "error":
                progress(1.0, desc=f"Error: {message}")
            elif status == "oom_fallback":
                progress(0.6, desc=f"GPU OOM detected, retrying on CPU: {message}")

        success = current_retriever.initialize_model(progress_callback)

        initialization_time = time.time() - start_time
        logger.info(
            f"Model initialization completed in {initialization_time:.2f} seconds"
        )
        check_high_gpu_usage()

        if success:
            progress(1.0, desc="Model ready for indexing!")
            logger.info("Retrieval model initialized successfully")

            # Reset indexing state when new model is successfully initialized
            indexing_state["status"] = "not_started"
            indexing_state["last_error"] = None
            indexing_state["last_attempt_time"] = None
            indexing_state["model_failed"] = False
            indexing_state["model_initializing"] = False

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
            # Reset indexing state on model initialization failure
            indexing_state["status"] = "not_started"
            indexing_state["last_error"] = "Model initialization failed"
            indexing_state["last_attempt_time"] = time.time()
            indexing_state["model_failed"] = True
            indexing_state["model_initializing"] = False
            logger.error("Model initialization failed")
            return "Error initializing model - likely GPU out of memory", True  # error

    except Exception as e:
        current_retriever = None
        # Reset indexing state on exception
        indexing_state["status"] = "not_started"
        indexing_state["last_error"] = f"Model initialization exception: {str(e)}"
        indexing_state["last_attempt_time"] = time.time()
        indexing_state["model_failed"] = True
        indexing_state["model_initializing"] = False
        logger.error(f"Exception during model initialization: {e}", exc_info=True)

        # Check if it's a GPU OOM error
        if "out of memory" in str(e).lower():
            return (
                "Error: GPU out of memory - try using a smaller model (ColPali instead of ColQwen2)",
                True,
            )
        return f"Error: {str(e)}", True  # error


def process_pdf_for_retrieval(pdf_file, progress=gr.Progress()):
    """Process PDF with multimodal retrieval"""
    global current_retriever

    logger.info("Starting PDF processing for retrieval")
    check_high_gpu_usage()

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

        # Track if an error occurred during indexing
        indexing_error = None

        def progress_callback(message, status):
            nonlocal indexing_error
            logger.info(f"PDF processing - {status}: {message}")
            if status == "indexing":
                progress(0.3, desc=message)
            elif status == "complete":
                progress(1.0, desc=message)
            elif status == "error":
                progress(1.0, desc=f"Error: {message}")
                indexing_error = message

        # Index the PDF
        logger.info("Starting PDF indexing...")
        logger.info(f"PDF size: {pdf_size_mb:.2f} MB - Starting indexing process")
        indexing_start_time = time.time()

        success = current_retriever.index_pdf(pdf_data, progress_callback)

        indexing_time = time.time() - start_time
        pure_indexing_time = time.time() - indexing_start_time

        # Check if there was an error during indexing
        if indexing_error:
            error_msg = f"PDF indexing failed: {indexing_error}"
            logger.error(error_msg)
            check_high_gpu_usage()
            return f"Error: {error_msg}", None

        # If indexing failed but no specific error was captured
        if not success:
            error_msg = "PDF indexing failed - check GPU memory availability"
            logger.error(error_msg)
            check_high_gpu_usage()
            return f"Error: {error_msg}", None

        logger.info(
            f"PDF indexing completed in {indexing_time:.2f} seconds (pure indexing: {pure_indexing_time:.2f}s)"
        )
        check_high_gpu_usage()

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
            # If we don't have a specific error message, provide a generic one
            if not indexing_error:
                error_msg = f"PDF indexing failed after {pure_indexing_time:.2f}s - check GPU memory availability"
            else:
                error_msg = f"PDF indexing failed: {indexing_error}"
            logger.error(error_msg)
            check_high_gpu_usage()
            return f"Error: {error_msg}", None

    except Exception as e:
        error_msg = f"Exception during PDF processing: {str(e)}"
        logger.error(error_msg, exc_info=True)
        check_high_gpu_usage()
        return f"Error: {error_msg}", None


def process_pdf_with_status(pdf_file):
    """Process PDF upload and return basic info without indexing"""
    if not pdf_file:
        return "<i>No PDF uploaded</i>", None, "", False

    # Get filename from uploaded file
    filename = (
        getattr(pdf_file, "name", "uploaded.pdf")
        if hasattr(pdf_file, "name")
        else "uploaded.pdf"
    )

    # Just get basic PDF info without indexing
    info_text, preview_image = get_pdf_info_and_preview(pdf_file)

    # Format the status display
    status_html = f"<b>📄 {filename}</b><br/>{info_text}"

    return status_html, preview_image, "", False


def index_pdf_manually(pdf_file, progress=gr.Progress()):
    """Manually trigger PDF indexing for retrieval"""
    global current_retriever, indexing_state

    import time

    if not pdf_file:
        error_msg = "No PDF uploaded"
        logger.warning(error_msg)
        indexing_state["status"] = "failed"
        indexing_state["last_error"] = error_msg
        indexing_state["last_attempt_time"] = time.time()
        return error_msg, gr.update(value="", visible=False), gr.update(visible=False)

    if not current_retriever:
        error_msg = "Please select a retrieval model first"
        logger.warning(error_msg)
        indexing_state["status"] = "failed"
        indexing_state["last_error"] = error_msg
        indexing_state["last_attempt_time"] = time.time()
        return error_msg, gr.update(value="", visible=False), gr.update(visible=False)

    # Check if model is still initializing
    if indexing_state.get("model_initializing", False):
        error_msg = "Model is still initializing - please wait"
        error_html = """
        <div style='background: #ff9800; padding: 10px; border-radius: 5px; color: white; text-align: center; margin: 5px 0;'>
            ⏳ Model is still initializing. Please wait for it to complete before indexing.
        </div>
        """
        return (
            error_msg,
            gr.update(value=error_html, visible=True),
            gr.update(visible=True),
        )

    # Check if the model failed to initialize
    if indexing_state.get("model_failed", False):
        error_msg = "Model initialization failed - please try a different model or check GPU memory"
        error_html = """
        <div style='background: #f44336; padding: 15px; border-radius: 5px; color: white; margin: 10px 0;'>
            <h3 style='margin: 0 0 10px 0;'>❌ Model Initialization Failed</h3>
            <p style='margin: 5px 0;'>The selected model failed to initialize, likely due to GPU memory constraints.</p>
            <p style='margin: 5px 0;'><strong>Solutions:</strong></p>
            <ul style='margin: 5px 0 0 20px;'>
                <li>Select a smaller model</li>
                <li>Restart the application to free GPU memory</li>
                <li>Ensure you have sufficient GPU memory (24GB+ recommended)</li>
            </ul>
        </div>
        """
        return (
            error_msg,
            gr.update(value=error_html, visible=True),
            gr.update(visible=True),
        )

    logger.info("Manual PDF indexing triggered")
    indexing_state["status"] = "in_progress"
    indexing_state["last_error"] = None
    indexing_state["last_attempt_time"] = time.time()

    try:
        # Check GPU usage before starting
        check_high_gpu_usage()

        # Process the PDF for indexing
        info_text, preview_image = process_pdf_for_retrieval(pdf_file, progress)

        # Check if indexing was successful
        if "Error" in info_text:
            # Customize the error message based on the type of error
            if "GPU out of memory" in info_text or "out of memory" in info_text.lower():
                error_html = """
                <div style='background: #f44336; padding: 15px; border-radius: 5px; color: white; margin: 10px 0;'>
                    <h3 style='margin: 0 0 10px 0;'>❌ GPU Out of Memory Error</h3>
                    <p style='margin: 5px 0;'>The retrieval model requires more GPU memory than available.</p>
                    <p style='margin: 5px 0;'><strong>Solutions:</strong></p>
                    <ul style='margin: 5px 0 0 20px;'>
                        <li>Use a smaller retrieval model</li>
                        <li>Ensure you have sufficient GPU memory (24GB+ recommended)</li>
                    </ul>
                </div>
                """
            else:
                error_html = f"""
                <div style='background: #f44336; padding: 10px; border-radius: 5px; color: white; text-align: center; margin: 5px 0;'>
                    ❌ Indexing Failed: {info_text}
                </div>
                """
            logger.error(f"PDF indexing failed: {info_text}")
            indexing_state["status"] = "failed"
            indexing_state["last_error"] = info_text
            return (
                info_text,
                gr.update(value=error_html, visible=True),
                gr.update(visible=True),
            )

        # Check for successful indexing completion
        if "PDF indexed successfully" in info_text:
            import re

            # Extract timing info for display
            timing_match = re.search(r"in ([\d.]+)s", info_text)
            if timing_match:
                indexing_time = float(timing_match.group(1))

                # Check if timing is suspiciously low (indicates failure)
                if indexing_time < 0.1:
                    error_msg = "Indexing completed too quickly - likely failed silently (possibly GPU OOM)"
                    error_html = f"""
                    <div style='background: #ff9800; padding: 10px; border-radius: 5px; color: white; text-align: center; margin: 5px 0;'>
                        ⚠️ Indexing may have failed: {error_msg}
                    </div>
                    """
                    logger.error(error_msg)
                    indexing_state["status"] = "failed"
                    indexing_state["last_error"] = error_msg
                    return (
                        error_msg,
                        gr.update(value=error_html, visible=True),
                        gr.update(visible=True),
                    )

                success_html = f"""
                <div style='background: #4CAF50; padding: 10px; border-radius: 5px; color: white; text-align: center; margin: 5px 0;'>
                    ✅ PDF Indexed Successfully in {indexing_time:.2f} seconds
                </div>
                """
                logger.info(
                    f"PDF indexing completed successfully in {indexing_time:.2f}s"
                )
                indexing_state["status"] = "success"
                indexing_state["last_error"] = None
                return (
                    "PDF indexing completed successfully",
                    gr.update(value=success_html, visible=True),
                    gr.update(visible=True),
                )

        # Fallback - unclear status
        warning_msg = "PDF indexing status unclear - check logs for details"
        warning_html = f"""
        <div style='background: #ff9800; padding: 10px; border-radius: 5px; color: white; text-align: center; margin: 5px 0;'>
            ⚠️ {warning_msg}
        </div>
        """
        logger.warning(f"{warning_msg}. Info text: {info_text}")
        indexing_state["status"] = "failed"
        indexing_state["last_error"] = warning_msg
        return (
            warning_msg,
            gr.update(value=warning_html, visible=True),
            gr.update(visible=True),
        )

    except Exception as e:
        error_msg = f"Exception during PDF indexing: {str(e)}"
        error_html = f"""
        <div style='background: #f44336; padding: 10px; border-radius: 5px; color: white; text-align: center; margin: 5px 0;'>
            ❌ Indexing Failed: {str(e)}
        </div>
        """
        logger.error(error_msg, exc_info=True)
        indexing_state["status"] = "failed"
        indexing_state["last_error"] = error_msg
        return (
            error_msg,
            gr.update(value=error_html, visible=True),
            gr.update(visible=True),
        )


def get_indexing_status_display():
    """Get HTML display of current indexing status"""
    global indexing_state
    import time

    status = indexing_state["status"]
    last_error = indexing_state["last_error"]
    last_attempt = indexing_state["last_attempt_time"]

    if status == "not_started":
        return """
        <div style='background: #9E9E9E; padding: 8px; border-radius: 4px; color: white; text-align: center; margin: 5px 0; font-size: 14px;'>
            📄 No PDF indexed yet
        </div>
        """
    elif status == "in_progress":
        return """
        <div style='background: #2196F3; padding: 8px; border-radius: 4px; color: white; text-align: center; margin: 5px 0; font-size: 14px;'>
            ⏳ Indexing in progress...
        </div>
        """
    elif status == "success":
        time_ago = ""
        if last_attempt:
            minutes_ago = int((time.time() - last_attempt) / 60)
            if minutes_ago == 0:
                time_ago = " (just now)"
            elif minutes_ago == 1:
                time_ago = " (1 minute ago)"
            else:
                time_ago = f" ({minutes_ago} minutes ago)"

        return f"""
        <div style='background: #4CAF50; padding: 8px; border-radius: 4px; color: white; text-align: center; margin: 5px 0; font-size: 14px;'>
            ✅ PDF indexed successfully{time_ago}
        </div>
        """
    elif status == "failed":
        time_ago = ""
        if last_attempt:
            minutes_ago = int((time.time() - last_attempt) / 60)
            if minutes_ago == 0:
                time_ago = " (just now)"
            elif minutes_ago == 1:
                time_ago = " (1 minute ago)"
            else:
                time_ago = f" ({minutes_ago} minutes ago)"

        error_preview = (
            last_error[:50] + "..."
            if last_error and len(last_error) > 50
            else last_error
        )
        return f"""
        <div style='background: #f44336; padding: 8px; border-radius: 4px; color: white; text-align: center; margin: 5px 0; font-size: 14px;'>
            ❌ Indexing failed{time_ago}<br/>
            <small>{error_preview or 'Unknown error'}</small>
        </div>
        """
    else:
        return """
        <div style='background: #FF9800; padding: 8px; border-radius: 4px; color: white; text-align: center; margin: 5px 0; font-size: 14px;'>
            ⚠️ Unknown indexing status
        </div>
        """


def retrieve_relevant_pages(query, k=3):
    """Retrieve relevant pages using multimodal search"""
    global current_retriever, indexing_state

    logger.info(f"Starting page retrieval - Query: '{query}', k={k}")
    check_high_gpu_usage()

    if not current_retriever or not query.strip():
        logger.warning("No retriever available or empty query")
        return [], ""

    # Check if indexing is in a failed state
    if indexing_state["status"] == "failed":
        error_msg = (
            "⚠️ **Indexing Failed** - Cannot retrieve pages because PDF indexing failed."
        )
        if indexing_state["last_error"]:
            error_msg += f"\n\n**Last Error:** {indexing_state['last_error']}"
        error_msg += "\n\n*Please re-index the PDF before attempting to query.*"
        logger.warning(
            f"Retrieval blocked due to failed indexing state: {indexing_state['last_error']}"
        )
        return [], error_msg

    # Check if indexing hasn't been attempted yet
    if indexing_state["status"] == "not_started":
        warning_msg = "⚠️ **No PDF Indexed** - Please upload and index a PDF before attempting to retrieve pages."
        logger.warning("Retrieval attempted without any PDF being indexed")
        return [], warning_msg

    # Check if indexing is currently in progress
    if indexing_state["status"] == "in_progress":
        warning_msg = "⚠️ **Indexing In Progress** - Please wait for PDF indexing to complete before querying."
        logger.warning("Retrieval attempted while indexing is still in progress")
        return [], warning_msg

    try:
        start_time = time.time()
        results = current_retriever.search(query, k=k)
        retrieval_time = time.time() - start_time

        logger.info(f"Page retrieval completed in {retrieval_time:.2f} seconds")
        logger.info(f"Retrieved {len(results)} relevant pages")

        # Create formatted display text
        if not results:
            formatted_results = "*No relevant pages found for this query.*"
        else:
            formatted_results = f"**Found {len(results)} relevant page(s) in {retrieval_time:.2f}s:**\n\n"
            for i, (img_bytes, score, page_num) in enumerate(results):
                formatted_results += (
                    f"**{i+1}.** Page {page_num} - Similarity: {score:.3f}\n"
                )

        # Log detailed results
        for i, (img_bytes, score, page_num) in enumerate(results):
            logger.info(
                f"Result {i+1}: Page {page_num}, Similarity score {score:.4f}, Image size: {len(img_bytes)} bytes"
            )

        check_high_gpu_usage()
        return results, formatted_results
    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
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
        "Upload a PDF, choose an indexing model, select response generation models, and submit your query."
    )

    with gr.Row():
        with gr.Column(scale=1):
            # PDF Upload and Processing Section - MOVED TO #1
            gr.Markdown("## 1. PDF upload & processing")
            pdf_input = gr.File(
                label="📄 Upload PDF",
                file_types=[".pdf"],
                type="binary",
                container=False,
            )

            # Filename display (will be updated with actual filename when uploaded)
            pdf_status = gr.HTML(value="<i>No PDF uploaded</i>")

            # Multimodal Retrieval Model Section - MOVED TO #2
            gr.Markdown("## 2. Select multimodal retrieval model")
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

            # Index PDF Button - NEW #3
            gr.Markdown("## 3. Index PDF for retrieval")
            index_pdf_btn = gr.Button(
                "🔍 Index PDF", variant="secondary", size="lg", interactive=False
            )
            gr.Markdown(
                "💡 **Tip:** If you do not want to index the PDF, you must select the specific page in the PDF to send to the models",
                elem_classes=["tooltip-text"],
            )

            # Indexing time display (prominent display for performance monitoring)
            indexing_time_display = gr.HTML(
                value="", visible=False, label="Indexing Performance"
            )

            # Response Generation Models Section - MOVED TO #4
            gr.Markdown("## 4. Select response generation models")

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

            gr.Markdown("## 5. Submit query")

            question_input = gr.Textbox(
                label="Enter Query",
                placeholder="Ask a question about this PDF...",
                lines=3,
                show_label=False,
            )

            compare_btn = gr.Button("⚡ Submit Query", variant="primary", size="lg")

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

            # Comparison Results Section - MOVED HERE
            gr.Markdown("## 📊 Comparison Results")
            gr.Markdown("### 🤖 Model A Response")
            with gr.Group():
                response_a = gr.Markdown(
                    value="Model A response will appear here...", height=400
                )
                timing_a = gr.HTML(value="", visible=False, label="Model A Timing")

            gr.Markdown("### 🤖 Model B Response")
            with gr.Group():
                response_b = gr.Markdown(
                    value="Model B response will appear here...", height=400
                )
                timing_b = gr.HTML(value="", visible=False, label="Model B Timing")

    # Function to clear retrieval results when switching modes
    def clear_retrieval_results():
        return "*No retrieval model selected - page display mode active*"

    # Function to reset displays when no PDF
    def reset_displays_on_empty():
        return "<i>No PDF uploaded</i>", None, "", False

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

        response_a, response_b, status_msg, show_status, retrieval_results, timing_a, timing_b = (
            compare_models(
                pdf_file, page_num, question, model_a_id, model_b_id, use_retrieval
            )
        )

        return (response_a, response_b, retrieval_results, timing_a, timing_b)

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

    # Enable/disable Index PDF button when retrieval model or PDF changes
    def update_index_button_state(pdf_file, retrieval_model):
        # Check if model is initializing or failed
        if indexing_state.get("model_initializing", False):
            return gr.Button(value="⏳ Model Initializing...", interactive=False)
        elif indexing_state.get("model_failed", False):
            return gr.Button(value="❌ Model Failed", interactive=False)
        elif pdf_file and retrieval_model and retrieval_model != "None":
            return gr.Button(value="🔍 Index PDF", interactive=True)
        else:
            return gr.Button(value="🔍 Index PDF", interactive=False)

    pdf_input.change(
        fn=update_index_button_state,
        inputs=[pdf_input, retrieval_model_dropdown],
        outputs=[index_pdf_btn],
    )

    retrieval_model_dropdown.change(
        fn=update_index_button_state,
        inputs=[pdf_input, retrieval_model_dropdown],
        outputs=[index_pdf_btn],
    )

    # Index PDF button handler
    index_pdf_btn.click(
        fn=index_pdf_manually,
        inputs=[pdf_input],
        outputs=[
            retrieval_status,
            indexing_time_display,
            indexing_time_display,  # Controls visibility
        ],
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
    ).then(
        fn=update_index_button_state,
        inputs=[pdf_input, retrieval_model_dropdown],
        outputs=[index_pdf_btn],
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
        outputs=[response_a, response_b, retrieval_results_display, timing_a, timing_b],
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
        outputs=[response_a, response_b, retrieval_results_display, timing_a, timing_b],
    )

if __name__ == "__main__":
    logger.info("=== VLM COMPARISON TOOL STARTUP ===")
    logger.info(f"Log file: {log_filename}")
    check_high_gpu_usage()
    logger.info("Starting Gradio application...")

    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public sharing
    )
