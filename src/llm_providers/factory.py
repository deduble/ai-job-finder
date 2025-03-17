import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

def create_llm_provider(provider_name: str, api_key: str, model: Optional[str] = None) -> Any:
    """
    Create and return an instance of the specified LLM provider.
    
    Args:
        provider_name: Name of the LLM provider ('openai', 'claude', or 'gemini')
        api_key: API key for the provider
        model: Optional specific model to use
        
    Returns:
        An instance of the appropriate LLM provider
    
    Raises:
        ValueError: If the provider is not supported
    """
    if provider_name.lower() == "openai":
        from src.llm_providers.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key, model)
    elif provider_name.lower() == "claude":
        from src.llm_providers.claude_provider import ClaudeProvider
        return ClaudeProvider(api_key, model)
    elif provider_name.lower() == "gemini":
        from src.llm_providers.gemini_provider import GeminiProvider
        return GeminiProvider(api_key, model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
