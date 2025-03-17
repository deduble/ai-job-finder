#!/usr/bin/env python3
from apify import Actor
import logging
import json
import base64
import re
import os
from typing import Dict, List, Any, Optional

# Import providers
from .llm_providers.factory import create_llm_provider
from .cv_processor import process_cv
from .prompt_processor import process_prompt
from .parameter_handler import apply_parameter_defaults

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point for the Actor"""
    # Initialize the Actor
    await Actor.init()
    
    # Get input from the actor
    actor_input = await Actor.get_input() or {}
    
    # Validate input - require at least CV or prompt
    cv_data = actor_input.get("cv")
    prompt = actor_input.get("prompt")
    
    if not cv_data and not prompt:
        raise ValueError("At least one of CV or prompt must be provided")
    
    # Get LLM settings
    llm_settings = actor_input.get("llm_settings", {"provider": "gemini", "model": "gemini-1.5-pro"})
    provider_name = llm_settings.get("provider", "gemini")
    
    # Get API key - first from input, then from environment variables
    api_keys = actor_input.get("api_keys", {})
    api_key = api_keys.get(provider_name)
    
    # If no API key in input, try to get from environment variables
    if not api_key:
        if provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider_name == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
        elif provider_name == "claude":
            api_key = os.getenv("CLAUDE_API_KEY")
    
    # If no API key was found, we can't proceed with LLM processing
    if not api_key:
        logger.warning(f"No API key provided for {provider_name}")
        await Actor.push_data([{
            "title": "LLM API KEY IS NEEDED",
            "description": f"Please provide an API key for {provider_name.upper()} to use this Actor",
            "instructions": f"Set the {provider_name.upper()}_API_KEY environment variable or provide it in the api_keys input parameter",
            "location": "N/A",
            "companyName": "AI Job Finder",
            "experienceLevel": "N/A",
            "workType": "N/A",
            "contractType": "N/A",
            "publishedAt": "N/A",
            "message": f"API key for {provider_name} is required to get real results"
        }])
        logger.info("Returned message indicating API key is needed")
        return
    
    # Create LLM provider for processing
    model = llm_settings.get("model")
    if provider_name == "gemini" and not model:
        model = "gemini-1.5-pro"
        
    logger.info(f"Using LLM provider: {provider_name} with model: {model}")
    llm_provider = create_llm_provider(provider_name, api_key, model)
    
    # Process parameters
    parameters = {}
    
    # Extract parameters from CV and/or prompt
    if cv_data:
        logger.info("Processing CV...")
        cv_parameters = await process_cv(cv_data, llm_provider, provider_name)
        parameters.update(cv_parameters)
    
    if prompt:
        logger.info("Processing prompt...")
        try:
            prompt_parameters = await process_prompt(prompt, llm_provider)
            # Prompt parameters override CV parameters
            parameters.update(prompt_parameters)
        except Exception as e:
            logger.error(f"Error processing prompt: {str(e)}")
            # Continue with default parameters
    
    # Apply any explicit parameters from input
    linkedin_params = actor_input.get("linkedin_search_params", {})
    if linkedin_params:
        parameters.update(linkedin_params)
    
    # Apply defaults for missing parameters
    parameters = apply_parameter_defaults(parameters)
    
    # Set proxy configuration
    if "proxy_configuration" in actor_input:
        parameters["proxy"] = actor_input["proxy_configuration"]
    elif "proxy" in actor_input:
        parameters["proxy"] = actor_input["proxy"]
    
    # Log the parameters we'll use
    logger.info(f"Using LinkedIn search parameters: {json.dumps(parameters, indent=2)}")
    
    # Call LinkedIn scraper
    logger.info("Calling LinkedIn scraper with parameters")
    try:
        jobs = await call_linkedin_scraper(parameters)
        
        # Save output
        await Actor.push_data(jobs)
        logger.info(f"Found {len(jobs)} matching jobs")
    except Exception as e:
        logger.error(f"Error calling LinkedIn scraper: {str(e)}")
        # Return a meaningful error to the user
        await Actor.push_data([{
            "title": "Error Connecting to LinkedIn Scraper",
            "description": f"An error occurred while trying to connect to the LinkedIn Jobs Scraper: {str(e)}",
            "error": True,
            "parameters": parameters
        }])

async def call_linkedin_scraper(parameters):
    """Call the LinkedIn scraper with the given parameters"""
    # Prepare the Actor input
    run_input = {
        "title": parameters.get("title", ""),
        "location": parameters.get("location", ""),
        "companyName": parameters.get("companyName", []),
        "companyId": parameters.get("companyId", []),
        "workType": parameters.get("workType", ""),
        "experienceLevel": parameters.get("experienceLevel", ""),
        "contractType": parameters.get("contractType", ""),
        "publishedAt": parameters.get("publishedAt", ""),
        "rows": parameters.get("rows", 10),
        "proxy": parameters.get("proxy", {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"]
        })
    }
    
    # Run the Actor and wait for it to finish using Actor.apify_client
    # This automatically handles the authentication - no need for explicit API key
    run = await Actor.apify_client.actor("BHzefUZlZRKWxkTck").call(run_input=run_input)
    
    # Fetch and return the Actor's output
    dataset_items = await Actor.apify_client.dataset(run["defaultDatasetId"]).list_items()
    return dataset_items.items
