import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def apply_parameter_defaults(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply default values for missing parameters

    Args:
        parameters: Current set of parameters

    Returns:
        Parameters with defaults applied
    """
    # Create a copy of the parameters to avoid modifying the original
    final_params = parameters.copy()

    # Check for title (required parameter)
    if "title" not in final_params or not final_params["title"]:
        final_params["title"] = "Software Engineer"  # Default job title
        logger.info("Using default job title: 'Software Engineer'")

    # Set default location if not provided
    if "location" not in final_params or not final_params["location"]:
        final_params["location"] = "United States"  # Country is required, default to United States
        logger.info("Using default location: United States")

    # Set default experience level if not provided
    if "experienceLevel" not in final_params or not final_params["experienceLevel"]:
        final_params["experienceLevel"] = "3"  # Associate
        logger.info("Using default experience level: 3 (Associate)")

    # Set default work type if not provided
    if "workType" not in final_params or not final_params["workType"]:
        final_params["workType"] = ""  # Empty string means any work type
        logger.info("Using default work type: any")

    # Set default contract type if not provided
    if "contractType" not in final_params or not final_params["contractType"]:
        final_params["contractType"] = "F"  # Full-time
        logger.info("Using default contract type: F (Full-Time)")

    # Set default published at if not provided
    if "publishedAt" not in final_params or not final_params["publishedAt"]:
        final_params["publishedAt"] = ""  # Empty string means any time
        logger.info("Using default time frame: any time")

    # Set default company name if not provided
    if "companyName" not in final_params or not final_params["companyName"]:
        final_params["companyName"] = []  # Empty list means any company
        logger.info("Using default company name: any company")

    # Set default company ID if not provided
    if "companyId" not in final_params or not final_params["companyId"]:
        final_params["companyId"] = []  # Empty list means any company ID
        logger.info("Using default company ID: any company ID")

    # Set default rows if not provided
    if "rows" not in final_params or not final_params["rows"]:
        final_params["rows"] = 10  # Default to 10 results
        logger.info("Using default rows: 10")

    # Ensure we have proper proxy configuration
    if "proxy" not in final_params or not final_params["proxy"]:
        final_params["proxy"] = {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"]
        }
        logger.info("Using default proxy configuration")

    return final_params
