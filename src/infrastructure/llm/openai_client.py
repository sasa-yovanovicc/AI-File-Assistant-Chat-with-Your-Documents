"""OpenAI LLM implementation."""

from typing import Dict, Any
import openai

from ...domain.repositories import LLMRepository
from ...config import USE_OPENAI, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS
from ...exceptions import LLMError
from ...error_handler import handle_errors


class OpenAILLMClient(LLMRepository):
    """OpenAI implementation of LLMRepository."""
    
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.model = OPENAI_MODEL
        self.max_tokens = OPENAI_MAX_TOKENS
        
        if self.api_key:
            openai.api_key = self.api_key
    
    def is_available(self) -> bool:
        """Check if OpenAI LLM is available."""
        return USE_OPENAI and bool(self.api_key) and openai is not None
    
    @handle_errors(default_return="", exception_type=LLMError)
    def generate_answer(self, prompt: str, **kwargs) -> str:
        """Generate an answer using OpenAI."""
        if not self.is_available():
            raise LLMError(message="OpenAI LLM is not available")
        
        # Default system message
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based only on provided context. Always try to find relevant information in the context regardless of capitalization differences."
        }
        
        user_message = {"role": "user", "content": prompt}
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[system_message, user_message],
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            temperature=kwargs.get('temperature', 0.3)
        )
        
        return response.choices[0].message.content
