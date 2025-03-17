import json
import logging
import re
import base64
from typing import Dict, Any, Optional, List

import google.generativeai as genai
from src.llm_providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)

class GeminiProvider(LLMProvider):
    """Implementation of LLM provider for Google Gemini"""

    def __init__(self, api_key: str, model: Optional[str] = None):
        """Initialize the Gemini provider"""
        super().__init__(api_key, model)
        genai.configure(api_key=api_key)
        self.model_name = model or "gemini-1.5-pro"
        self.model = genai.GenerativeModel(self.model_name)

    def supports_document_processing(self) -> bool:
        """Check if this provider/model supports direct document processing"""
        vision_capable_models = ["gemini-pro-vision", "gemini-1.5-pro", "gemini-1.5-flash"]
        return any(model_name in self.model_name for model_name in vision_capable_models)

    async def validate_api_key(self) -> bool:
        """Validate the API key by making a simple models call"""
        try:
            # Gemini doesn't have a dedicated validate endpoint, use a simple generation
            response = self.model.generate_content("Hello")
            return True
        except Exception as e:
            logger.error(f"Gemini API key validation failed: {str(e)}")
            return False

    async def process_cv(self, cv_data: str) -> Dict[str, Any]:
        """
        Process CV with Gemini

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
        """Process CV using Gemini's vision capabilities"""
        try:
            # Extract the mime type and base64 data
            mime_type, encoded_data = cv_data.split(';base64,', 1)
            mime_type = mime_type.replace('data:', '')

            # Create a content parts list with prompt and image
            parts = [
                self._get_cv_prompt(),
                {"mime_type": mime_type, "data": base64.b64decode(encoded_data)}
            ]

            response = self.model.generate_content(
                parts,
                generation_config={
                    "temperature": 0.1
                }
            )

            # Extract JSON from response
            content = response.text

            # Try to parse as JSON directly
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If direct parsing fails, look for JSON in code blocks
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))

                # If still no match, try to find anything that looks like JSON
                json_pattern = r'{.*}'
                json_match = re.search(json_pattern, content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))

                logger.error(f"Could not parse Gemini response as JSON: {content}")
                raise ValueError("Failed to parse Gemini response as JSON")

        except Exception as e:
            logger.error(f"Gemini vision processing failed: {str(e)}")
            raise

    async def _process_cv_text(self, cv_text: str) -> Dict[str, Any]:
        """Process CV text with Gemini"""
        try:
            response = self.model.generate_content(
                self._get_cv_prompt() + f"\n\nCV TEXT:\n{cv_text}",
                generation_config={
                    "temperature": 0.1
                }
            )

            # Extract JSON from response
            content = response.text

            # Try to parse as JSON directly
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If direct parsing fails, look for JSON in code blocks
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))

                # If still no match, try to find anything that looks like JSON
                json_pattern = r'{.*}'
                json_match = re.search(json_pattern, content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))

                logger.error(f"Could not parse Gemini response as JSON: {content}")
                raise ValueError("Failed to parse Gemini response as JSON")

        except Exception as e:
            logger.error(f"Gemini text processing failed: {str(e)}")
            raise

    async def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """Process user prompt and extract job search parameters"""
        try:
            response = self.model.generate_content(
                self._get_prompt_extraction_prompt() + f"\n\nUSER QUERY:\n{prompt}",
                generation_config={
                    "temperature": 0.1
                }
            )

            # Extract JSON from response
            content = response.text

            # Try to parse as JSON directly
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If direct parsing fails, look for JSON in code blocks
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))

                # If still no match, try to find anything that looks like JSON
                json_pattern = r'{.*}'
                json_match = re.search(json_pattern, content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))

                logger.error(f"Could not parse Gemini response as JSON: {content}")
                raise ValueError("Failed to parse Gemini response as JSON")

        except Exception as e:
            logger.error(f"Gemini prompt processing failed: {str(e)}")
            raise

    def _get_cv_prompt(self) -> str:
        """Get the prompt for CV analysis"""
        return """
        Extract the following job search parameters from this CV/resume:

        Follow these steps:
        1. Identify the job title
        2. Determine the location
        3. Assess experience level (1-5)
        4. Identify work type preference (1-3)
        5. Determine contract type (FPCTIV)
        6. List key skills

        Return ONLY a JSON object with this format:
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

        IMPORTANT: Your output must be ONLY the JSON object with no additional text.
        """

    def _get_prompt_extraction_prompt(self) -> str:
        """Get the prompt for extracting parameters from user query"""
        return """
        Extract LinkedIn job search parameters from this query.

        Follow these steps:
        1. Identify job title or role
        2. Determine geographic location
        3. Note any specific companies mentioned
        4. Assess experience level (1-5)
        5. Identify work type (1-3)
        6. Determine contract type (FPCTIV)
        7. Identify time frame for job postings

        Return ONLY a JSON object with this format:
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

        IMPORTANT: Your output must be ONLY the JSON object with no additional text.
        """
