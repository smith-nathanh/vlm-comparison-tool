#!/usr/bin/env python3
"""
Deployment script for FinColQwen RAG system
requirements:
    byaldi
    claudette
    pillow
"""

import argparse
import base64
import os
from io import BytesIO
from pathlib import Path

from byaldi import RAGMultiModalModel
from claudette import Chat, models
from PIL import Image


def setup_environment():
    """Setup environment variables"""
    # For local deployment, you'll need to set these environment variables
    # or modify this function to load from a config file
    hf_token = os.getenv("HF_TOKEN")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")


def initialize_rag_model():
    """Initialize the RAG model"""
    print("Loading RAG model...")
    return RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0-merged", verbose=1)


def create_index(rag_model, input_path, index_name, overwrite=True):
    """Create document index from PDF directory"""
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    print(f"Creating index '{index_name}' from {input_path}")
    rag_model.index(
        input_path=input_path,
        index_name=index_name,
        store_collection_with_index=True,
        overwrite=overwrite,
    )
    print("Index created successfully")


def search_documents(rag_model, query, k=5):
    """Search for relevant documents"""
    print(f"Searching for: {query}")
    results = rag_model.search(query, k=k)
    print(f"Found {len(results)} results")
    return results


def display_image(image_bytes, save_path=None):
    """Display or save the retrieved image"""
    image = Image.open(BytesIO(image_bytes))
    resized_image = image.resize((900, 600))

    if save_path:
        resized_image.save(save_path)
        print(f"Image saved to: {save_path}")
    else:
        # For script usage, save to temp file instead of displaying
        temp_path = "retrieved_document.png"
        resized_image.save(temp_path)
        print(f"Retrieved document saved as: {temp_path}")

    return resized_image


def query_claude(image_bytes, query_text):
    """Send image and query to Claude"""
    print(f"Available models: {models}")

    # Use Claude 3.5 Sonnet (adjust index if needed)
    chat = Chat(models[1])  # 'claude-3-5-sonnet-20250219'

    print("Querying Claude...")
    response = chat([image_bytes, query_text])
    return response


def main():
    parser = argparse.ArgumentParser(description="FinColQwen RAG Deployment Script")
    parser.add_argument(
        "--input-path", required=True, help="Path to directory containing PDFs"
    )
    parser.add_argument(
        "--index-name", required=True, help="Name for the document index"
    )
    parser.add_argument("--query", required=True, help="Query to search for")
    parser.add_argument(
        "--k", type=int, default=5, help="Number of results to retrieve"
    )
    parser.add_argument(
        "--save-image", help="Path to save the retrieved document image"
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip indexing if index already exists",
    )

    args = parser.parse_args()

    try:
        # Setup
        setup_environment()
        rag_model = initialize_rag_model()

        # Create index (unless skipping)
        if not args.skip_indexing:
            create_index(rag_model, args.input_path, args.index_name)

        # Search documents
        results = search_documents(rag_model, args.query, args.k)

        if not results:
            print("No results found for the query")
            return

        # Get the top result
        top_result = results[0]
        image_bytes = base64.b64decode(top_result.base64)

        # Display/save image
        display_image(image_bytes, args.save_image)

        # Query Claude
        response = query_claude(image_bytes, args.query)

        print("\n" + "=" * 50)
        print("CLAUDE RESPONSE:")
        print("=" * 50)
        print(response)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
