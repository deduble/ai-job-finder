import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def process_prompt(prompt: str, llm_provider) -> Dict[str, Any]:
    """
    Process user prompt and extract job search parameters
    
    Args:
        prompt: User's job search query
        llm_provider: The LLM provider instance to use
        
    Returns:
        Dictionary of extracted parameters for LinkedIn job search
    """
    try:
        logger.info("Processing user prompt")
        
        # Process prompt with the provider
        prompt_parameters = await llm_provider.process_prompt(prompt)
        
        # Validate and clean the parameters
        prompt_parameters = validate_prompt_parameters(prompt_parameters)
        
        logger.info(f"Successfully extracted parameters from prompt: {json.dumps(prompt_parameters, indent=2)}")
        return prompt_parameters
        
    except Exception as e:
        logger.error(f"Error processing prompt: {str(e)}")
        # Return empty parameters, which will use defaults later
        return {}

def validate_prompt_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean the parameters extracted from the prompt
    
    Args:
        parameters: Raw parameters extracted by the LLM
        
    Returns:
        Cleaned and validated parameters
    """
    cleaned = {}
    
    # Clean and validate title
    if "title" in parameters and parameters["title"]:
        cleaned["title"] = str(parameters["title"]).strip()
    
    # Clean and validate location
    if "location" in parameters and parameters["location"]:
        cleaned["location"] = str(parameters["location"]).strip()
    
    # Clean and validate experienceLevel
    if "experienceLevel" in parameters and parameters["experienceLevel"]:
        exp_level = str(parameters["experienceLevel"]).strip()
        # Ensure it's a number from 1-5
        if exp_level in ["1", "2", "3", "4", "5"]:
            cleaned["experienceLevel"] = exp_level
    
    # Clean and validate workType
    if "workType" in parameters and parameters["workType"]:
        work_type = str(parameters["workType"]).strip()
        # Ensure it's a valid work type (1, 2, or 3)
        if work_type in ["1", "2", "3"]:
            cleaned["workType"] = work_type
    
    # Clean and validate contractType
    if "contractType" in parameters and parameters["contractType"]:
        contract_type = str(parameters["contractType"]).strip().upper()
        # Ensure it's a valid contract type (F, P, C, T, I, or V)
        if contract_type in ["F", "P", "C", "T", "I", "V"]:
            cleaned["contractType"] = contract_type
    
    # Clean and validate publishedAt
    if "publishedAt" in parameters and parameters["publishedAt"]:
        published_at = str(parameters["publishedAt"]).strip()
        # Ensure it's a valid time frame
        if published_at in ["r86400", "r604800", "r2592000", ""]:
            cleaned["publishedAt"] = published_at
    
    # Clean and validate rows
    if "rows" in parameters and parameters["rows"]:
        try:
            rows = int(parameters["rows"])
            if rows > 0:
                cleaned["rows"] = rows
        except (ValueError, TypeError):
            pass
    
    # Clean and validate companyName
    if "companyName" in parameters and isinstance(parameters["companyName"], list):
        cleaned["companyName"] = [str(company).strip() for company in parameters["companyName"] if company]
    
    # Clean and validate companyId
    if "companyId" in parameters and isinstance(parameters["companyId"], list):
        cleaned["companyId"] = [str(company_id).strip() for company_id in parameters["companyId"] if company_id]
    
    return cleaned
