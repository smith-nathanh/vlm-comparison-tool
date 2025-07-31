import gradio as gr
import fitz  # PyMuPDF
import base64
import io
import requests
from PIL import Image
import json
import os
from typing import List, Dict, Tuple, Optional

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
            "X-Title": "VLM Comparison Tool"
        }
        
        response = requests.get(f"{OPENROUTER_BASE_URL}/models", headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            
            # Filter for vision-capable models using OpenRouter's modalities field
            vlm_models = []
            for model in models:
                # Check if model supports vision via modalities
                modalities = model.get("modalities", [])
                if "image" in modalities:
                    vlm_models.append(model)
                    continue
                
                # Fallback: check architecture field for vision capabilities
                architecture = model.get("architecture", {})
                if architecture.get("modality") == "multimodal":
                    vlm_models.append(model)
                    continue
                
                # Fallback: known vision model patterns
                model_id = model.get("id", "").lower()
                if any(pattern in model_id for pattern in [
                    "gpt-4", "claude", "gemini", "llava", "pixtral", "qwen", "internvl", 
                    "vision", "vlm", "multimodal", "cogvlm", "yi-vl", "deepseek-vl"
                ]):
                    vlm_models.append(model)
            
            return vlm_models[:50]  # Limit to first 50 for performance
        else:
            print(f"Error fetching models: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        return []

def pdf_page_to_base64(pdf_file, page_num):
    """Convert a specific PDF page to base64 encoded image"""
    try:
        # Open PDF
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        
        # Check if page exists
        if page_num < 1 or page_num > len(pdf_document):
            return None, f"Page {page_num} not found. PDF has {len(pdf_document)} pages."
        
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
        pil_image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
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
            "X-Title": "VLM Comparison Tool"
        }
        
        # Format according to OpenRouter docs - text first, then image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ]
            }
        ]
        
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60
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
                    error_msg += f": {error_data['error'].get('message', 'Unknown error')}"
            except:
                error_msg += f": {response.text[:200]}"
            return f"Error: {error_msg}"
            
    except requests.exceptions.Timeout:
        return "Error: Request timed out (60s)"
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON response"

def compare_models(pdf_file, page_num, question, model_a_id, model_b_id):
    """Main function to compare both models"""
    if not pdf_file:
        return "Please upload a PDF file", "Please upload a PDF file", None, ""
    
    if not question.strip():
        return "Please enter a question", "Please enter a question", None, ""
    
    if not model_a_id or not model_b_id:
        return "Please select both models", "Please select both models", None, ""
    
    # Convert PDF page to base64
    img_base64, pil_image, error = pdf_page_to_base64(pdf_file, page_num)
    
    if error:
        return error, error, None, ""
    
    # Query both models
    status_msg = f"Querying {model_a_id} and {model_b_id}..."
    
    try:
        response_a = query_openrouter_model(model_a_id, question, img_base64)
        response_b = query_openrouter_model(model_b_id, question, img_base64)
        
        status_msg = f"✅ Comparison complete using {model_a_id} vs {model_b_id}"
        
        return response_a, response_b, pil_image, status_msg
    except Exception as e:
        error_msg = f"Error during comparison: {str(e)}"
        return error_msg, error_msg, pil_image, error_msg

def get_pdf_info(pdf_file):
    """Get basic info about uploaded PDF"""
    if not pdf_file:
        return "No PDF uploaded"
    
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        num_pages = len(pdf_document)
        pdf_document.close()
        return f"PDF loaded successfully. Total pages: {num_pages}"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Initialize models list
print("Fetching available VLM models from OpenRouter...")
available_models = get_available_models()

# Create better model display names with pricing info
model_choices = []
for model in available_models:
    name = model.get('name', model.get('id', 'Unknown'))
    model_id = model.get('id', '')
    
    # Add pricing info if available
    pricing = model.get('pricing', {})
    prompt_price = pricing.get('prompt', '0')
    completion_price = pricing.get('completion', '0')
    
    display_name = f"{name}"
    if prompt_price != '0' or completion_price != '0':
        display_name += f" (${prompt_price}/{completion_price} per 1M tokens)"
    
    model_choices.append((f"{display_name} | {model_id}", model_id))

if not model_choices:
    model_choices = [("No VLM models available - check API key", "")]
    print("Warning: No VLM models fetched from OpenRouter API")
else:
    print(f"Found {len(model_choices)} vision-capable models")

# Create Gradio interface
with gr.Blocks(title="VLM Comparison Tool", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Vision Language Model Comparison Tool")
    gr.Markdown("Upload a PDF, select models, ask a question, and compare responses from different VLMs via OpenRouter.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # File upload section
            pdf_input = gr.File(
                label="📄 Upload PDF",
                file_types=[".pdf"],
                type="binary"
            )
            pdf_info = gr.Textbox(
                label="PDF Info",
                interactive=False,
                value="No PDF uploaded"
            )
            page_input = gr.Number(
                label="📃 Page Number",
                value=1,
                minimum=1,
                step=1
            )
            
            # Model selection section
            gr.Markdown("### 🔧 Model Selection")
            with gr.Row():
                model_a_dropdown = gr.Dropdown(
                    choices=model_choices,
                    label="Model A",
                    value=model_choices[0][1] if model_choices and model_choices[0][1] else None,
                    interactive=True
                )
                model_b_dropdown = gr.Dropdown(
                    choices=model_choices,
                    label="Model B", 
                    value=model_choices[1][1] if len(model_choices) > 1 and model_choices[1][1] else None,
                    interactive=True
                )
            
            refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")
            
            # Question section
            question_input = gr.Textbox(
                label="❓ Question",
                placeholder="Ask a question about the PDF page...",
                lines=3
            )
            
            compare_btn = gr.Button("⚡ Compare Models", variant="primary", size="lg")
            
            status_display = gr.Textbox(
                label="Status",
                interactive=False,
                value="Ready to compare models"
            )
    
    with gr.Row():
        with gr.Column(scale=1):
            model_a_title = gr.Markdown("### 🤖 Model A Response")
            response_a = gr.Textbox(
                label="",
                lines=12,
                interactive=False,
                placeholder="Model A response will appear here..."
            )
        
        with gr.Column(scale=1):
            model_b_title = gr.Markdown("### 🤖 Model B Response")
            response_b = gr.Textbox(
                label="",
                lines=12,
                interactive=False,
                placeholder="Model B response will appear here..."
            )
    
    with gr.Row():
        pdf_display = gr.Image(
            label="📖 PDF Page Preview",
            type="pil"
        )
    
    # Helper function to refresh models
    def refresh_models():
        new_models = get_available_models()
        new_choices = [(f"{model.get('name', model.get('id', 'Unknown'))} ({model.get('id', '')})", model.get('id', '')) 
                      for model in new_models]
        if not new_choices:
            new_choices = [("No models available - check API key", "")]
        
        return (
            gr.Dropdown(choices=new_choices, value=new_choices[0][1] if new_choices and new_choices[0][1] else None),
            gr.Dropdown(choices=new_choices, value=new_choices[1][1] if len(new_choices) > 1 and new_choices[1][1] else None),
            f"Refreshed: Found {len(new_choices)} models"
        )
    
    # Event handlers
    pdf_input.change(
        fn=get_pdf_info,
        inputs=[pdf_input],
        outputs=[pdf_info]
    )
    
    refresh_models_btn.click(
        fn=refresh_models,
        outputs=[model_a_dropdown, model_b_dropdown, status_display]
    )
    
    compare_btn.click(
        fn=compare_models,
        inputs=[pdf_input, page_input, question_input, model_a_dropdown, model_b_dropdown],
        outputs=[response_a, response_b, pdf_display, status_display]
    )
    
    # Allow Enter key to trigger comparison
    question_input.submit(
        fn=compare_models,
        inputs=[pdf_input, page_input, question_input, model_a_dropdown, model_b_dropdown],
        outputs=[response_a, response_b, pdf_display, status_display]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False  # Set to True for public sharing
    )