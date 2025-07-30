import gradio as gr
import fitz  # PyMuPDF
import base64
import io
import requests
from PIL import Image
import json

# Configuration - update these with your actual endpoints
MODEL_A_ENDPOINT = "http://your-model-a-endpoint/predict"
MODEL_B_ENDPOINT = "http://your-model-b-endpoint/predict"
MODEL_A_NAME = "Model A"
MODEL_B_NAME = "Model B"

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

def query_model(endpoint, question, img_base64):
    """Send query to model endpoint"""
    try:
        payload = {
            "question": question,
            "image": img_base64
        }
        
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            # Adjust this based on your API response format
            return result.get("answer", "No answer field in response")
        else:
            return f"Error: HTTP {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON response"

def compare_models(pdf_file, page_num, question):
    """Main function to compare both models"""
    if not pdf_file:
        return "Please upload a PDF file", "Please upload a PDF file", None
    
    if not question.strip():
        return "Please enter a question", "Please enter a question", None
    
    # Convert PDF page to base64
    img_base64, pil_image, error = pdf_page_to_base64(pdf_file, page_num)
    
    if error:
        return error, error, None
    
    # Query both models
    response_a = query_model(MODEL_A_ENDPOINT, question, img_base64)
    response_b = query_model(MODEL_B_ENDPOINT, question, img_base64)
    
    return response_a, response_b, pil_image

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

# Create Gradio interface
with gr.Blocks(title="VLM Comparison Tool") as demo:
    gr.Markdown("# Vision Language Model Comparison")
    gr.Markdown("Upload a PDF, select a page, ask a question, and compare responses from two models.")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="binary"
            )
            pdf_info = gr.Textbox(
                label="PDF Info",
                interactive=False,
                value="No PDF uploaded"
            )
            page_input = gr.Number(
                label="Page Number",
                value=1,
                minimum=1,
                step=1
            )
            question_input = gr.Textbox(
                label="Question",
                placeholder="Ask a question about the PDF page...",
                lines=3
            )
            compare_btn = gr.Button("Compare Models", variant="primary")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(f"### {MODEL_A_NAME} Response")
            response_a = gr.Textbox(
                label="",
                lines=10,
                interactive=False
            )
        
        with gr.Column(scale=1):
            gr.Markdown(f"### {MODEL_B_NAME} Response")
            response_b = gr.Textbox(
                label="",
                lines=10,
                interactive=False
            )
    
    with gr.Row():
        pdf_display = gr.Image(
            label="PDF Page",
            type="pil"
        )
    
    # Event handlers
    pdf_input.change(
        fn=get_pdf_info,
        inputs=[pdf_input],
        outputs=[pdf_info]
    )
    
    compare_btn.click(
        fn=compare_models,
        inputs=[pdf_input, page_input, question_input],
        outputs=[response_a, response_b, pdf_display]
    )
    
    # Allow Enter key to trigger comparison
    question_input.submit(
        fn=compare_models,
        inputs=[pdf_input, page_input, question_input],
        outputs=[response_a, response_b, pdf_display]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False  # Set to True for public sharing
    )