"""Ollama LLM implementation."""

from typing import Dict, Any, Optional
import requests
import json

from ...domain.repositories import LLMRepository
from ...exceptions import LLMError
from ...error_handler import handle_errors
from ...logging_config import get_logger

logger = get_logger(__name__)


class OllamaLLMClient(LLMRepository):
    """Ollama implementation of LLMRepository."""
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model_name: str = "llama3.1:8b"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.api_url = f"{self.base_url}/api/generate"
    
    def _check_connection(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    @handle_errors(default_return="I apologize, but I'm unable to process your request at the moment.", 
                  exception_type=LLMError)
    def generate_response(self, 
                         prompt: str, 
                         system_message: Optional[str] = None,
                         temperature: float = 0.3,
                         max_tokens: Optional[int] = None) -> str:
        """Generate response using Ollama."""
        if not self._check_connection():
            raise LLMError(message=f"Cannot connect to Ollama server at {self.base_url}")
        
        # Build the prompt with system message
        full_prompt = prompt
        if system_message:
            full_prompt = f"System: {system_message}\n\nUser: {prompt}"
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        logger.info(f"Generating Ollama response with model: {self.model_name}")
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120  # Ollama can be slow
            )
            
            if response.status_code != 200:
                raise LLMError(
                    message=f"Ollama API error: {response.status_code}",
                    details={"status_code": response.status_code, "response": response.text}
                )
            
            result = response.json()
            
            if "error" in result:
                raise LLMError(
                    message=f"Ollama error: {result['error']}",
                    details=result
                )
            
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            raise LLMError(
                message=f"Network error connecting to Ollama: {str(e)}",
                details={"error": str(e)}
            )
        except json.JSONDecodeError as e:
            raise LLMError(
                message=f"Invalid JSON response from Ollama: {str(e)}",
                details={"error": str(e)}
            )
    
    @handle_errors(default_return={}, exception_type=LLMError)
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self._check_connection():
            return {"available": False, "error": "Cannot connect to Ollama server"}
        
        try:
            # Get list of available models
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get("models", [])
                
                # Find current model
                current_model = None
                for model in models:
                    if model.get("name") == self.model_name:
                        current_model = model
                        break
                
                return {
                    "available": True,
                    "model_name": self.model_name,
                    "current_model": current_model,
                    "all_models": models,
                    "server_url": self.base_url
                }
            else:
                return {
                    "available": False,
                    "error": f"API returned status {response.status_code}"
                }
                
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    def is_available(self) -> bool:
        """Check if Ollama LLM is available."""
        return self._check_connection()
    
    def generate_answer(self, prompt: str, **kwargs) -> str:
        """Generate an answer using Ollama - alias for generate_response."""
        return self.generate_response(prompt, **kwargs)
