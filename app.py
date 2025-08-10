import base64
import io
import json
import os
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
import gradio as gr
import requests
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Import our custom retrieval system
try:
    from retrieval import MultimodalRetriever, create_retriever

    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False
    print(
        "Warning: Retrieval system not available - multimodal retrieval will be disabled"
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
            print(f"Error fetching models: {response.status_code}")
            return []

    except Exception as e:
        print(f"Error fetching models: {str(e)}")
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

        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content
            else:
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
            return f"Error: {error_msg}"

    except requests.exceptions.Timeout:
        return "Error: Request timed out (60s)"
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON response"


def compare_models(
    pdf_file, page_num, question, model_a_id, model_b_id, use_retrieval=True
):
    """Main function to compare both models - supports both retrieval and manual page selection"""
    global current_retriever

    if not pdf_file:
        error_msg = "Please upload a PDF file"
        return error_msg, error_msg, error_msg, True  # Show status on error

    if not question.strip():
        error_msg = "Please enter a question"
        return error_msg, error_msg, error_msg, True  # Show status on error

    if not model_a_id or not model_b_id:
        error_msg = "Please select both models"
        return error_msg, error_msg, error_msg, True  # Show status on error

    try:
        # Determine if we should use retrieval or manual page selection
        use_retrieval_mode = use_retrieval and current_retriever is not None

        if use_retrieval_mode:
            # Use multimodal retrieval to find relevant pages
            relevant_pages = retrieve_relevant_pages(question, k=3)

            if not relevant_pages:
                # Fall back to manual page selection
                img_base64, pil_image, error = pdf_page_to_base64(pdf_file, page_num)
                if error:
                    return error, error, error, True
                status_msg = f"No relevant pages found via retrieval. Using page {page_num} instead."
            else:
                # Use the most relevant page (first result)
                img_bytes, score = relevant_pages[0]
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                status_msg = f"Using most relevant page (similarity: {score:.3f})"

        else:
            # Use manual page selection
            img_base64, pil_image, error = pdf_page_to_base64(pdf_file, page_num)
            if error:
                return error, error, error, True
            status_msg = f"Using manually selected page {page_num}"

        # Query both models
        response_a = query_openrouter_model(model_a_id, question, img_base64)
        response_b = query_openrouter_model(model_b_id, question, img_base64)

        # Check if responses contain errors
        if "Error:" in response_a or "Error:" in response_b:
            status_msg = "One or more models returned an error"
            return response_a, response_b, status_msg, True  # Show status on error

        # Success - show status with page info
        return response_a, response_b, status_msg, True  # Show status for page info

    except Exception as e:
        error_msg = f"Error during comparison: {str(e)}"
        return error_msg, error_msg, error_msg, True  # Show status on error


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

    if not RETRIEVAL_AVAILABLE:
        return "Error: Retrieval system not available", True

    if not model_name or model_name == "None":
        current_retriever = None
        return "No retrieval model selected - manual page selection required", False

    try:
        progress(0.1, desc="Creating retriever...")
        current_retriever = create_retriever(model_name)

        if not current_retriever:
            return f"Error: Unsupported model {model_name}", True

        progress(0.3, desc="Initializing model...")

        def progress_callback(message, status):
            if status == "downloading":
                progress(0.5, desc=message)
            elif status == "ready":
                progress(0.8, desc=message)
            elif status == "error":
                progress(1.0, desc=f"Error: {message}")

        success = current_retriever.initialize_model(progress_callback)

        if success:
            progress(1.0, desc="Model ready for indexing!")
            return (
                "Model initialized successfully. Upload a PDF to start indexing.",
                False,
            )
        else:
            current_retriever = None
            return "Error initializing model", True

    except Exception as e:
        current_retriever = None
        return f"Error: {str(e)}", True


def process_pdf_for_retrieval(pdf_file, progress=gr.Progress()):
    """Process PDF with multimodal retrieval"""
    global current_retriever

    if not current_retriever:
        return get_pdf_info_and_preview(pdf_file)  # Fall back to regular preview

    if not pdf_file:
        return "No PDF uploaded", None

    try:
        progress(0.1, desc="Starting PDF indexing...")

        # Get PDF data
        if hasattr(pdf_file, "read"):
            pdf_data = pdf_file.read()
        else:
            pdf_data = pdf_file

        def progress_callback(message, status):
            if status == "indexing":
                progress(0.3, desc=message)
            elif status == "complete":
                progress(1.0, desc=message)
            elif status == "error":
                progress(1.0, desc=f"Error: {message}")

        # Index the PDF
        success = current_retriever.index_pdf(pdf_data, progress_callback)

        if success:
            # Get basic PDF info
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            num_pages = len(pdf_document)

            # Get status for timing info
            status = current_retriever.get_status()
            timing = status.get("timing", {})
            gpu_usage = status.get("gpu_usage", {})

            timing_info = f"Indexing completed in {timing.get('indexing_time', 0):.2f}s"
            if gpu_usage.get("total_mb", 0) > 0:
                timing_info += f" (GPU: {gpu_usage['used_mb']:.0f}MB/{gpu_usage['total_mb']:.0f}MB)"

            info_text = f"PDF indexed successfully. Pages: {num_pages}. {timing_info}"

            # Show first page as preview
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
    """Process PDF and return HTML formatted status"""
    if not pdf_file:
        return "<i>No PDF uploaded</i>", None

    # Get filename from uploaded file
    filename = (
        getattr(pdf_file, "name", "uploaded.pdf")
        if hasattr(pdf_file, "name")
        else "uploaded.pdf"
    )

    # Process the PDF (this will also handle the indexing if retrieval model is available)
    info_text, preview_image = process_pdf_for_retrieval(pdf_file)

    # Format as HTML with filename and status
    if "Error" in info_text:
        html_status = f"<span style='color: red;'>📄 {filename} - {info_text}</span>"
    else:
        html_status = f"<span style='color: green;'>📄 {filename} - {info_text}</span>"

    return html_status, preview_image


def retrieve_relevant_pages(query, k=3):
    """Retrieve relevant pages using multimodal search"""
    global current_retriever

    if not current_retriever or not query.strip():
        return []

    try:
        return current_retriever.search(query, k=k)
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []


# Initialize models list
print("Fetching available VLM models from OpenRouter...")
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
    print("Warning: No VLM models fetched from OpenRouter API")
else:
    print(
        f"Found {len(all_model_info)} vision-capable models from {len(all_providers)} providers"
    )

# Custom CSS for better layout and compact upload
custom_css = """
.gr-group .prose {
    padding: 12px !important;
}
.compact-upload {
    max-width: 200px !important;
}
.compact-upload .upload-container {
    min-height: 40px !important;
    padding: 8px !important;
}
/* Hide the file drop area completely - target all possible elements */
.compact-upload .file-drop-area,
.compact-upload .gr-file-upload-area,
.compact-upload .upload-drop-zone,
.compact-upload .drop-zone,
.compact-upload .file-upload-drop,
.compact-upload .gr-file .file-drop {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
}
/* Hide drag and drop text and icons */
.compact-upload .drag-text,
.compact-upload .or-text,
.compact-upload .upload-icon,
.compact-upload .file-upload-text {
    display: none !important;
}
/* Show only the upload button */
.compact-upload .upload-button,
.compact-upload .gr-button,
.compact-upload button {
    width: 100% !important;
    margin: 0 !important;
    padding: 8px 16px !important;
}
/* Override any background styling */
.compact-upload {
    background: transparent !important;
    border: none !important;
}
"""

# Create Gradio interface
with gr.Blocks(
    title="VLM Comparison Tool", theme=gr.themes.Soft(), css=custom_css
) as demo:
    gr.Markdown("# 🤖 Vision Language Model Comparison Tool")
    gr.Markdown(
        "Select any VLM available on OpenRouter, upload a PDF, and compare responses."
    )

    with gr.Row():
        with gr.Column(scale=1):
            # Multimodal Retrieval Model Section
            gr.Markdown("## 🔍 Multimodal Retrieval Model")
            retrieval_model_dropdown = gr.Dropdown(
                choices=["None"]
                + (MultimodalRetriever.SUPPORTED_MODELS if RETRIEVAL_AVAILABLE else []),
                label="Choose multimodal retrieval model",
                value="None",
                interactive=True,
                info="Select a model for intelligent page retrieval, or 'None' for manual page selection",
            )

            retrieval_status = gr.Textbox(
                label="Retrieval Status",
                value="No retrieval model selected",
                interactive=False,
                visible=False,
            )

            # Response Generation Models Section
            gr.Markdown("## 🤖 Response Generation Models")

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
            gr.Markdown("## 📄 PDF Upload & Processing")
            pdf_input = gr.File(
                label="📄 Upload PDF",
                file_types=[".pdf"],
                type="binary",
                elem_classes=["compact-upload"],
            )

            # Filename display (will be updated with actual filename when uploaded)
            pdf_status = gr.HTML(value="<i>No PDF uploaded</i>")

            gr.Markdown("###  Manual Page Selection (Optional)")
            gr.Markdown("*Only required when no retrieval model is selected*")
            page_input = gr.Number(
                label="Page Number (fallback)",
                value=1,
                minimum=1,
                step=1,
                info="Used as fallback when retrieval finds no relevant pages",
            )

            question_input = gr.Textbox(
                label="❓ Question",
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

        response_a, response_b, status_msg, show_status = compare_models(
            pdf_file, page_num, question, model_a_id, model_b_id, use_retrieval
        )

        return (response_a, response_b)

    # Event handlers
    pdf_input.change(
        fn=process_pdf_with_status,
        inputs=[pdf_input],
        outputs=[pdf_status, pdf_display],
    )

    # PDF page preview when page number changes
    page_input.change(
        fn=preview_pdf_page, inputs=[pdf_input, page_input], outputs=[pdf_display]
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
            retrieval_status,
        ],  # Second output controls visibility
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
            page_input,
            question_input,
            model_a_dropdown,
            model_b_dropdown,
        ],
        outputs=[response_a, response_b],
    )

    # Allow Enter key to trigger comparison
    question_input.submit(
        fn=compare_with_status_visibility,
        inputs=[
            pdf_input,
            page_input,
            question_input,
            model_a_dropdown,
            model_b_dropdown,
        ],
        outputs=[response_a, response_b],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public sharing
    )
