import json
import logging
import base64
import re
import numpy as np
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
            
    async def generate_job_description_from_cv(self, cv_data: str) -> str:
        """Generate an optimized job description based on a CV"""
        # Extract text from CV if needed
        cv_text = cv_data
        if cv_data.startswith("data:"):
            # Use vision capabilities if supported
            logger.info("Using vision model to process CV document")
            return await self._generate_job_description_with_vision(cv_data)
            
        # Use text-based approach
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a job description writer tasked with creating the perfect job description for a candidate based on their CV."},
                    {"role": "user", "content": f"""
                    Create a detailed job description that would be a perfect match for the candidate with this CV.
                    The job description should highlight all their skills and experience, making it an ideal fit.
                    Focus on their technical skills, experience level, and career trajectory.
                    
                    CV:
                    {cv_text}
                    
                    Return only the job description, no comments or explanations.
                    """}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI job description generation failed: {str(e)}")
            raise
            
    async def _generate_job_description_with_vision(self, cv_data: str) -> str:
        """Generate job description using vision capabilities"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a job description writer tasked with creating the perfect job description for a candidate based on their CV."},
                    {"role": "user", "content": [
                        {"type": "text", "text": """
                        Create a detailed job description that would be a perfect match for the candidate with this CV.
                        The job description should highlight all their skills and experience, making it an ideal fit.
                        Focus on their technical skills, experience level, and career trajectory.
                        
                        Return only the job description, no comments or explanations.
                        """},
                        {"type": "image_url", "image_url": {"url": cv_data}}
                    ]}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI vision job description generation failed: {str(e)}")
            raise
            
    async def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the given text using OpenAI's embedding API"""
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-3-small",  # Using OpenAI's latest embedding model
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {str(e)}")
            raise
            
    async def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays for efficient calculation
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # Avoid division by zero
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            
            # Scale to 0-1 range (cosine similarity is between -1 and 1)
            # For text embeddings, we typically expect positive similarity
            return max(0.0, similarity)
        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            # Return 0 on error
            return 0.0
    
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
