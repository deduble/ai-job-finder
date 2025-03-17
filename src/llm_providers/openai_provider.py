import json
import logging
import base64
import re
from typing import Dict, Any, Optional, List

from openai import AsyncOpenAI
from src.llm_providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)

class OpenAIProvider(LLMProvider):
    """Implementation of LLM provider for OpenAI"""
    
    def __init__(self, api_key: str, model: Optional[str] = None):
        """Initialize the OpenAI provider"""
        super().__init__(api_key, model)
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model or "gpt-4o"  # Default to most capable model
    
    def supports_document_processing(self) -> bool:
        """Check if this provider/model supports direct document processing"""
        document_capable_models = ["gpt-4-vision", "gpt-4o"]
        return any(model_name in self.model for model_name in document_capable_models)
    
    async def validate_api_key(self) -> bool:
        """Validate the API key by making a simple models.list call"""
        try:
            await self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI API key validation failed: {str(e)}")
            return False
    
    async def process_cv(self, cv_data: str) -> Dict[str, Any]:
        """
        Process CV with OpenAI
        
        Args:
            cv_data: CV content (could be base64 encoded file or text)
            
        Returns:
            Dictionary of extracted parameters
        """
        if self.supports_document_processing() and cv_data.startswith("data:"):
            return await self._process_cv_with_vision(cv_data)
        else:
            # Assume it's already text
            return await self._process_cv_text(cv_data)
    
    async def _process_cv_with_vision(self, cv_data: str) -> Dict[str, Any]:
        """Process CV using OpenAI's vision capabilities"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Extract job search parameters from this CV/resume."},
                    {"role": "user", "content": [
                        {"type": "text", "text": self._get_cv_prompt()},
                        {"type": "image_url", "image_url": {"url": cv_data}}
                    ]}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI vision processing failed: {str(e)}")
            raise
    
    async def _process_cv_text(self, cv_text: str) -> Dict[str, Any]:
        """Process CV text with OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Extract job search parameters from this CV/resume."},
                    {"role": "user", "content": self._get_cv_prompt() + f"\n\nCV TEXT:\n{cv_text}"}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI text processing failed: {str(e)}")
            raise
    
    async def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """Process user prompt and extract job search parameters"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Extract job search parameters from this query."},
                    {"role": "user", "content": self._get_prompt_extraction_prompt() + f"\n\nUSER QUERY:\n{prompt}"}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI prompt processing failed: {str(e)}")
            raise
    
    def _get_cv_prompt(self) -> str:
        """Get the prompt for CV analysis"""
        return """
        Extract the following job search parameters from this CV/resume in JSON format:

        Required JSON format:
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

        If a piece of information is not clearly stated in the CV, make a reasonable inference based on the available information. If inference is not possible, use null.
        """
    
    def _get_prompt_extraction_prompt(self) -> str:
        """Get the prompt for extracting parameters from user query"""
        return """
        Extract LinkedIn job search parameters from this query in JSON format:

        Required JSON format:
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

        For any parameters not explicitly mentioned in the query, use null.
        """
