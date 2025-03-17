import logging
import base64
import json
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def process_cv(cv_data: str, llm_provider, provider_name: str) -> Dict[str, Any]:
    """
    Process CV data using the appropriate LLM provider
    
    Args:
        cv_data: CV data (either base64 encoded file or plain text)
        llm_provider: The LLM provider instance to use
        provider_name: Name of the provider ('openai', 'claude', or 'gemini')
        
    Returns:
        Dictionary of extracted parameters for LinkedIn job search
    """
    try:
        logger.info(f"Processing CV with {provider_name} provider")
        
        # Process CV with the provider
        cv_parameters = await llm_provider.process_cv(cv_data)
        
        # Validate and clean the parameters
        cv_parameters = validate_cv_parameters(cv_parameters)
        
        logger.info(f"Successfully extracted parameters from CV: {json.dumps(cv_parameters, indent=2)}")
        return cv_parameters
        
    except Exception as e:
        logger.error(f"Error processing CV: {str(e)}")
        # Return empty parameters, which will use defaults later
        return {}

def validate_cv_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean the parameters extracted from the CV
    
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
    
    # Clean and validate skills (might be used for custom filtering later)
    if "skills" in parameters and isinstance(parameters["skills"], list):
        cleaned["skills"] = [str(skill).strip() for skill in parameters["skills"] if skill]
    
    return cleaned
