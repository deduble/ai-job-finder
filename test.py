#!/usr/bin/env python3
import asyncio
import json
import sys
import os
from typing import Dict, Any, Optional

# Import from our modules
from src.llm_providers.factory import create_llm_provider
from src.cv_processor import process_cv
from src.prompt_processor import process_prompt
from src.parameter_handler import apply_parameter_defaults

async def test_cv_processing():
    """Test CV processing with a local file"""
    # Check if file path was provided
    if len(sys.argv) < 2:
        print("Usage: python test.py path/to/cv.pdf [prompt]")
        sys.exit(1)
    
    # Get CV file path and optional prompt
    cv_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if API key is set
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)
    
    # Read CV file
    try:
        with open(cv_path, "rb") as f:
            cv_data = f.read()
            
        # Convert to base64 for testing
        import base64
        import mimetypes
        mime_type, _ = mimetypes.guess_type(cv_path)
        if not mime_type:
            mime_type = "application/octet-stream"
            
        cv_data_base64 = f"data:{mime_type};base64,{base64.b64encode(cv_data).decode('utf-8')}"
    except Exception as e:
        print(f"Error reading CV file: {str(e)}")
        sys.exit(1)
    
    # Create LLM provider
    provider = create_llm_provider("openai", openai_key)
    
    # Process CV
    print("Processing CV...")
    cv_parameters = await process_cv(cv_data_base64, provider, "openai")
    print(f"CV Parameters: {json.dumps(cv_parameters, indent=2)}")
    
    # Process prompt if provided
    prompt_parameters = {}
    if prompt:
        print("\nProcessing prompt...")
        prompt_parameters = await process_prompt(prompt, provider)
        print(f"Prompt Parameters: {json.dumps(prompt_parameters, indent=2)}")
    
    # Merge and apply defaults
    parameters = {**cv_parameters, **prompt_parameters}
    final_parameters = apply_parameter_defaults(parameters)
    
    print("\nFinal LinkedIn Search Parameters:")
    print(json.dumps(final_parameters, indent=2))
    
    # Note: This test doesn't actually call the LinkedIn scraper
    print("\nTest complete. To perform a real LinkedIn search, upload this Actor to Apify.")

if __name__ == "__main__":
    asyncio.run(test_cv_processing())
