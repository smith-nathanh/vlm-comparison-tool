# VLM Comparison Tool

A web application for comparing Vision Language Models (VLMs) on PDF documents using OpenRouter's API. Upload a PDF, ask questions, and compare responses from different VLMs side-by-side.

## Features

- **Model Selection**: Choose from any vision-capable models available on OpenRouter
- **Advanced Filtering**: Filter models by provider, context length, and pricing
- **PDF Processing**: Upload PDFs and browse pages with real-time preview
- **Multimodal PDF Indexing**: Index entire PDFs using state-of-the-art multimodal retrieval models
- **Intelligent Page Retrieval**: Automatically find the most relevant pages based on your questions
- **Side-by-Side Comparison**: Compare responses from two models simultaneously

## PDF Indexing & Multimodal Retrieval

The tool features an advanced multimodal retrieval system that can intelligently index and search through entire PDF documents:

### Supported Retrieval Models

- **ColQwen2-v1.0** (`vidore/colqwen2-v1.0`) - Latest multimodal retrieval model based on Qwen2-VL
- **ColPali-v1.3** (`vidore/colpali-v1.3`) - Advanced document understanding model for visual and textual content

### How PDF Indexing Works

1. **Upload & Index**: When you upload a PDF, the system can create a multimodal index of all pages
2. **Visual + Text Understanding**: The retrieval models analyze both visual elements (charts, diagrams, layouts) and textual content
3. **Intelligent Page Selection**: Instead of manually browsing pages, ask questions and the system finds the most relevant pages automatically
4. **Contextual Retrieval**: Get answers that span multiple pages when relevant information is distributed across the document


## Supported Models

### Vision Language Models (VLMs)

The tool automatically fetches all vision-capable models from OpenRouter, including:
- GPT-4 Vision models (OpenAI)
- Claude 3 models with vision (Anthropic)
- Gemini Pro Vision (Google)
- Qwen-VL models
- LLaVA variants
- And many more as they become available

### Multimodal Retrieval Models

For PDF indexing and intelligent page retrieval:
- **vidore/colqwen2-v1.0** - Latest Qwen2-VL based retrieval model
- **vidore/colpali-v1.3** - Advanced ColPali document understanding model

## Prerequisites

- Python 3.8+
- OpenRouter API key
- CUDA-compatible GPU (for fast PDF indexing)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vlm-comparison-tool
```

2. Install uv (recommended - fast Python package manager):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install dependencies:
```bash
uv sync
```

Alternatively, install with pip (including retrieval dependencies):
```bash
pip install gradio pymupdf pillow requests torch transformers byaldi
```

4. Set up your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Getting an API Key

1. Visit [https://openrouter.ai/](https://openrouter.ai/)
2. Sign up or log in to your account
3. Navigate to the API Keys section
4. Create a new API key
5. Set it as the `OPENROUTER_API_KEY` environment variable

## Usage

1. Start the application:
```bash
uv run app.py
```

If not using uv:
```bash
python app.py
```

2. Open your browser to `http://localhost:7860`

3. **Model Selection**:
   - **Retrieval Models**: Select a multimodal retrieval model (ColQwen2 or ColPali) for PDF indexing
   - **Response Models**: Use filters to narrow down VLM models by provider, context length, or price
   - Select Model A and Model B from the filtered list
   - Click "Refresh" to get the latest available models

4. **PDF Processing**:
   - Upload a PDF file
   - **Option 1 - Single Page Mode**: Browse to a specific page you want to analyze
   - **Option 2 - Retrieval Mode**: Index the entire PDF with the selected retrieval model
     - First-time indexing may take a few minutes as models are downloaded
     - Subsequent PDFs will index much faster
   - Enter your question about the PDF content

5. **Get Results**:
   - **Single Page**: Models analyze only the selected page
   - **Retrieval Mode**: System finds the most relevant pages and shows them alongside model responses
   - Click "Compare Models" to see responses side-by-side

## Advanced Usage

### PDF Indexing Tips

- **First Use**: Allow extra time for model downloads (5-10 minutes on first run)
- **GPU Memory**: Monitor GPU usage in logs - the system will warn about memory issues
- **Large PDFs**: Indexing time scales with document size; typical research papers index in 30-60 seconds
- **Model Selection**: ColQwen2-v1.0 is generally faster, ColPali-v1.3 may be more accurate for complex documents

### Retrieval vs Single Page Mode

- **Use Retrieval When**:
  - You have questions that might span multiple pages
  - You're exploring a large document without knowing where information is located
  - You want to find all relevant sections for a complex query

- **Use Single Page When**:
  - You know exactly which page contains the information
  - You want to analyze a specific diagram, chart, or section
  - You need faster response times (no indexing required)

## Configuration

The application runs on `http://localhost:7860` by default. To change this:

```python
demo.launch(
    server_name="0.0.0.0",  # Allow external access
    server_port=7860,       # Change port here
    share=False             # Set to True for public sharing
)
```

## API Limits

- **Response Length**: 4000 tokens maximum per response
- **Timeout**: 60 seconds per model query
- **Pricing**: Varies by model (displayed in the interface)

## Troubleshooting

### "No VLM models available - check API key"
- Verify your `OPENROUTER_API_KEY` environment variable is set correctly
- Check that your API key is valid and has sufficient credits
- Try clicking the "Refresh" button

### PDF Processing Errors
- Ensure your PDF is not corrupted or password-protected
- Check that the page number exists in the document
- Try a different PDF file

### Retrieval System Issues
- **"Retrieval model not loading"**: Check GPU memory availability, try restarting
- **"Out of GPU memory"**: Try a smaller PDF or use CPU mode (set in environment)
- **"Indexing failed"**: Check logs for detailed error messages, ensure PDF is valid
- **"No relevant pages found"**: Try rephrasing your question or use single-page mode

### Performance Issues
- **Slow indexing**: Ensure CUDA drivers are installed and GPU is available
- **High memory usage**: Monitor the logs/indexing_*.log files for memory statistics
- **Model download stuck**: Check internet connection, downloads can be 1-4GB per model

### Model Query Errors
- Some models may be temporarily unavailable
- Check your OpenRouter account for sufficient credits
- Try a different model combination

## Technical Details

### Dependencies
- **Core**: Gradio, PyMuPDF, Pillow, requests
- **Retrieval**: torch, transformers, byaldi (for ColPali/ColQwen models)
- **Optional**: GPUtil, psutil (for system monitoring)

### Logging & Monitoring
- Application logs: `logs/app_YYYYMMDD_HHMMSS.log`
- Indexing logs: `logs/indexing_YYYYMMDD_HHMMSS.log`
- GPU usage and timing statistics included in logs

### Model Storage
- Retrieval models are cached locally after first download
- Typical model sizes: 2-4GB each
- Models stored in HuggingFace cache directory

## Contributing
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.