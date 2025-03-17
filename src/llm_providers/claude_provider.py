import json
import logging
import re
import base64
import numpy as np
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
            
    async def generate_job_description_from_cv(self, cv_data: str) -> str:
        """Generate an optimized job description based on a CV"""
        # Extract text from CV if needed
        cv_text = cv_data
        if cv_data.startswith("data:"):
            # Use document capabilities if supported
            logger.info("Using Claude to process CV document")
            return await self._generate_job_description_with_document(cv_data)
            
        # Use text-based approach
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system="You are a job description writer tasked with creating the perfect job description for a candidate based on their CV.",
                messages=[
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
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Claude job description generation failed: {str(e)}")
            raise
            
    async def _generate_job_description_with_document(self, cv_data: str) -> str:
        """Generate job description using document capabilities"""
        try:
            # Extract the mime type and base64 data
            mime_type, encoded_data = cv_data.split(';base64,', 1)
            mime_type = mime_type.replace('data:', '')
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system="You are a job description writer tasked with creating the perfect job description for a candidate based on their CV.",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": """
                        Create a detailed job description that would be a perfect match for the candidate with this CV.
                        The job description should highlight all their skills and experience, making it an ideal fit.
                        Focus on their technical skills, experience level, and career trajectory.
                        
                        Return only the job description, no comments or explanations.
                        """},
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": encoded_data
                        }}
                    ]}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Claude document job description generation failed: {str(e)}")
            raise
            
    async def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the given text using Claude's API"""
        try:
            # Claude doesn't have a dedicated embeddings API, so we'll use a workaround
            # Ask the model to create a representation as a list of numbers
            # This is not ideal and a real implementation should use a different model or service
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system="Generate a numerical embedding representation of the text.",
                messages=[
                    {"role": "user", "content": f"""
                    Create a 384-dimensional numerical embedding for the following text. The embedding should be a 
                    list of 384 float numbers between -1 and 1 representing the semantic meaning of the text.
                    Return ONLY the list of numbers (as a valid JSON array) with no explanation or additional text.
                    
                    TEXT: {text}
                    """}
                ]
            )
            
            content = response.content[0].text
            # Find JSON in the content
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                logger.error(f"Failed to parse embeddings from Claude response: {content}")
                # Fallback to a random embedding (this should never happen in a real implementation)
                return [0.0] * 384
        except Exception as e:
            logger.error(f"Claude embedding generation failed: {str(e)}")
            # Return a zero embedding as a fallback
            return [0.0] * 384
            
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
