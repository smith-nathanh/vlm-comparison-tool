# VLM Comparison Tool

A web application for comparing Vision Language Models (VLMs) on PDF documents using OpenRouter's API. Upload a PDF, select any page, ask questions, and compare responses from different VLMs side-by-side.

## Features

- **Model Selection**: Choose from any vision-capable models available on OpenRouter
- **Advanced Filtering**: Filter models by provider, context length, and pricing
- **PDF Processing**: Upload PDFs and browse pages with real-time preview
- **Side-by-Side Comparison**: Compare responses from two models simultaneously
- **Interactive Interface**: Clean Gradio-based web interface with tabs and real-time updates

## Supported Models

The tool automatically fetches all vision-capable models from OpenRouter, including:
- GPT-4 Vision models (OpenAI)
- Claude 3 models with vision (Anthropic)
- Gemini Pro Vision (Google)
- Qwen-VL models
- LLaVA variants
- And many more as they become available

## Prerequisites

- Python 3.8+
- OpenRouter API key

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

Alternatively, you can use pip:
```bash
pip install gradio pymupdf pillow requests
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

3. **Model Selection Tab**:
   - Use filters to narrow down models by provider, context length, or price
   - Select Model A and Model B from the filtered list
   - Click "Refresh" to get the latest available models

4. **Comparison Tab**:
   - Upload a PDF file
   - Browse to the page you want to analyze
   - Enter your question about the PDF content
   - Click "Compare Models" to see responses side-by-side

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

### Model Query Errors
- Some models may be temporarily unavailable
- Check your OpenRouter account for sufficient credits
- Try a different model combination

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.