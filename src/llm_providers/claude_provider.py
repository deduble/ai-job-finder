import json
import logging
import re
import base64
from typing import Dict, Any, Optional, List

from anthropic import AsyncAnthropic
from src.llm_providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)

class ClaudeProvider(LLMProvider):
    """Implementation of LLM provider for Anthropic Claude"""
    
    def __init__(self, api_key: str, model: Optional[str] = None):
        """Initialize the Claude provider"""
        super().__init__(api_key, model)
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model or "claude-3-opus-20240229"  # Default to most capable model
    
    def supports_document_processing(self) -> bool:
        """Check if this provider/model supports direct document processing"""
        return "claude-3" in self.model  # All Claude 3 models support document processing
    
    async def validate_api_key(self) -> bool:
        """Validate the API key by making a simple models call"""
        try:
            # There's no direct way to validate the key without making a message request
            # Use a minimal request to check if the key works
            await self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return True
        except Exception as e:
            logger.error(f"Claude API key validation failed: {str(e)}")
            return False
    
    async def process_cv(self, cv_data: str) -> Dict[str, Any]:
        """
        Process CV with Claude
        
        Args:
            cv_data: CV content (could be base64 encoded file or text)
            
        Returns:
            Dictionary of extracted parameters
        """
        if self.supports_document_processing() and cv_data.startswith("data:"):
            return await self._process_cv_with_document_api(cv_data)
        else:
            # Assume it's already text
            return await self._process_cv_text(cv_data)
    
    async def _process_cv_with_document_api(self, cv_data: str) -> Dict[str, Any]:
        """Process CV using Claude's document capabilities"""
        try:
            # Extract the mime type and base64 data
            mime_type, encoded_data = cv_data.split(';base64,', 1)
            mime_type = mime_type.replace('data:', '')
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system="Extract job search parameters from this CV/resume.",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": self._get_cv_prompt()},
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": encoded_data
                        }}
                    ]}
                ]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            # Find JSON in the content (handle potential text wrapping)
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON block, try to parse the entire content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Claude document processing failed: {str(e)}")
            raise
    
    async def _process_cv_text(self, cv_text: str) -> Dict[str, Any]:
        """Process CV text with Claude"""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system="Extract job search parameters from this CV/resume.",
                messages=[
                    {"role": "user", "content": self._get_cv_prompt() + f"\n\nCV TEXT:\n{cv_text}"}
                ]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            # Find JSON in the content (handle potential text wrapping)
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON block, try to parse the entire content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Claude text processing failed: {str(e)}")
            raise
    
    async def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """Process user prompt and extract job search parameters"""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system="Extract job search parameters from this query.",
                messages=[
                    {"role": "user", "content": self._get_prompt_extraction_prompt() + f"\n\nUSER QUERY:\n{prompt}"}
                ]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            # Find JSON in the content (handle potential text wrapping)
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON block, try to parse the entire content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Claude prompt processing failed: {str(e)}")
            raise
    
    def _get_cv_prompt(self) -> str:
        """Get the prompt for CV analysis"""
        return """
        Extract the following job search parameters from this CV/resume.
        
        Return your response as valid JSON object inside ```json code blocks with the following structure:
        
        ```json
        {
          "title": "The most recent job title or professional role",
          "location": "Current or preferred location",
          "experienceLevel": "A numeric value from 1-5 where:
            1 = Internship
            2 = Entry Level
            3 = Associate
            4 = Mid-Senior Level
            5 = Director",
          "workType": "Either:
            1 = On-Site
            2 = Remote
            3 = Hybrid
           Based on any workstyle preferences found in the CV",
          "contractType": "A single letter representing employment type preference:
            F = Full-Time
            P = Part-Time
            C = Contract
            T = Temporary
            I = Internship
            V = Volunteer",
          "skills": ["list", "of", "key", "technical", "and", "soft", "skills"]
        }
        ```
        
        If a piece of information is not clearly stated in the CV, make a reasonable inference based on the available information. If inference is not possible, use null.
        
        IMPORTANT: Your output must be a valid JSON object wrapped in ```json code blocks.
        """
    
    def _get_prompt_extraction_prompt(self) -> str:
        """Get the prompt for extracting parameters from user query"""
        return """
        Extract LinkedIn job search parameters from this query.
        
        Return your response as valid JSON object inside ```json code blocks with the following structure:
        
        ```json
        {
          "title": "Job title or role to search for",
          "location": "Geographic location for job search",
          "companyName": ["array of specific companies mentioned"],
          "companyId": ["array of LinkedIn company IDs if mentioned"],
          "workType": "Either:
            1 = On-Site
            2 = Remote
            3 = Hybrid",
          "experienceLevel": "A numeric value from 1-5 where:
            1 = Internship
            2 = Entry Level
            3 = Associate
            4 = Mid-Senior Level
            5 = Director",
          "contractType": "A single letter representing employment type:
            F = Full-Time
            P = Part-Time
            C = Contract
            T = Temporary
            I = Internship
            V = Volunteer",
          "publishedAt": "Time frame:
            r86400 = Last 24 hours
            r604800 = Last week
            r2592000 = Last month
            empty string = Any time",
          "rows": "Number of job listings to return (integer)"
        }
        ```
        
        For any parameters not explicitly mentioned in the query, use null.
        
        IMPORTANT: Your output must be a valid JSON object wrapped in ```json code blocks.
        """
