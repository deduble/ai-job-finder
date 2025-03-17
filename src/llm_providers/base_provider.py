from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class LLMProvider(ABC):
    """Base abstract class for LLM providers"""
    
    def __init__(self, api_key: str, model: Optional[str] = None):
        """
        Initialize the LLM provider
        
        Args:
            api_key: API key for the provider
            model: Optional specific model to use
        """
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    async def process_cv(self, cv_data: str) -> Dict[str, Any]:
        """
        Process CV data and extract job search parameters
        
        Args:
            cv_data: CV content (could be base64 encoded file or text)
            
        Returns:
            Dictionary of extracted parameters
        """
        pass
    
    @abstractmethod
    async def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Process user prompt and extract job search parameters
        
        Args:
            prompt: User's job search query
            
        Returns:
            Dictionary of extracted parameters
        """
        pass
    
    @abstractmethod
    async def validate_api_key(self) -> bool:
        """
        Validate that the API key is correct
        
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def supports_document_processing(self) -> bool:
        """
        Check if the provider and model support direct document processing
        
        Returns:
            True if document processing is supported, False otherwise
        """
        return False
