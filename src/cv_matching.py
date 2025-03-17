import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

async def generate_ideal_job_description(cv_data: str, llm_provider) -> str:
    """
    Generate an ideal job description based on a CV
    
    Args:
        cv_data: CV content (could be base64 encoded file or text)
        llm_provider: The LLM provider instance to use
        
    Returns:
        A job description that would be a perfect match for the candidate
    """
    try:
        logger.info("Generating ideal job description from CV")
        job_description = await llm_provider.generate_job_description_from_cv(cv_data)
        return job_description
    except Exception as e:
        logger.error(f"Error generating job description: {str(e)}")
        return ""

async def calculate_job_match_scores(
    ideal_job_description: str, 
    job_listings: List[Dict[str, Any]], 
    llm_provider
) -> List[Dict[str, Any]]:
    """
    Calculate matching scores between ideal job description and job listings
    
    Args:
        ideal_job_description: The ideal job description generated from CV
        job_listings: List of job listings
        llm_provider: The LLM provider instance to use
        
    Returns:
        List of job listings with match scores added
    """
    try:
        if not ideal_job_description or not job_listings:
            logger.warning("Missing job description or job listings, skipping matching")
            return job_listings
            
        logger.info("Calculating job match scores")
        
        # Generate embedding for the ideal job description
        ideal_embedding = await llm_provider.generate_embeddings(ideal_job_description)
        
        enhanced_listings = []
        
        # For each job listing, calculate similarity score
        for job in job_listings:
            # Extract relevant text for comparison
            job_text = f"{job.get('title', '')} {job.get('description', '')}"
            if not job_text.strip():
                # If no description or title, just add with zero score
                job["match_score"] = 0.0
                enhanced_listings.append(job)
                continue
                
            # Generate embedding for job listing
            job_embedding = await llm_provider.generate_embeddings(job_text)
            
            # Calculate similarity score
            match_score = await llm_provider.calculate_similarity(ideal_embedding, job_embedding)
            
            # Add score to job listing
            job["match_score"] = round(match_score, 2)
            enhanced_listings.append(job)
        
        # Sort by match score (descending)
        enhanced_listings.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        
        logger.info(f"Calculated match scores for {len(enhanced_listings)} job listings")
        return enhanced_listings
        
    except Exception as e:
        logger.error(f"Error calculating job match scores: {str(e)}")
        # Return original listings without scores
        return job_listings
