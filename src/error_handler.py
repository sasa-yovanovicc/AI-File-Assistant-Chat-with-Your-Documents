"""Error handling utilities for the AI File Assistant application."""

import logging
import traceback
from typing import Any, Callable, Optional, TypeVar, Union
from functools import wraps

from .exceptions import AIFileAssistantError, LLMError, EmbeddingError, VectorStoreError

# Configure logger
logger = logging.getLogger(__name__)

T = TypeVar('T')


def handle_errors(
    default_return: Any = None,
    exception_type: type = AIFileAssistantError,
    log_level: int = logging.ERROR,
    reraise: bool = False
):
    """Decorator for consistent error handling across the application."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AIFileAssistantError as e:
                # Our custom errors - log with details
                logger.log(log_level, f"{func.__name__} failed: {e.message}", extra={
                    'error_code': e.error_code,
                    'details': e.details,
                    'function': func.__name__
                })
                if reraise:
                    raise
                return default_return
            except Exception as e:
                # Unexpected errors - wrap in our exception type
                error_msg = f"Unexpected error in {func.__name__}: {str(e)}"
                logger.log(log_level, error_msg, extra={
                    'function': func.__name__,
                    'original_error': str(e),
                    'details': {},  # Always provide details field
                    'traceback': traceback.format_exc()
                })
                
                if reraise:
                    # Wrap in our exception type
                    raise exception_type(
                        message=error_msg,
                        details={'original_error': str(e), 'function': func.__name__}
                    ) from e
                return default_return
        return wrapper
    return decorator


def log_error(
    error: Exception,
    context: str,
    details: Optional[dict] = None,
    level: int = logging.ERROR
) -> None:
    """Log an error with consistent formatting."""
    if isinstance(error, AIFileAssistantError):
        logger.log(level, f"{context}: {error.message}", extra={
            'error_code': error.error_code,
            'details': {**(error.details or {}), **(details or {})},
            'context': context
        })
    else:
        logger.log(level, f"{context}: {str(error)}", extra={
            'original_error': str(error),
            'details': details or {},
            'context': context,
            'traceback': traceback.format_exc()
        })


def safe_execute(
    func: Callable[[], T],
    context: str,
    default_return: Any = None,
    exception_type: type = AIFileAssistantError
) -> Union[T, Any]:
    """Safely execute a function with error logging."""
    try:
        return func()
    except AIFileAssistantError as e:
        log_error(e, context)
        return default_return
    except Exception as e:
        wrapped_error = exception_type(
            message=f"Error in {context}: {str(e)}",
            details={'original_error': str(e)}
        )
        log_error(wrapped_error, context)
        return default_return


def validate_config(config_dict: dict, required_keys: list, context: str = "Configuration") -> None:
    """Validate configuration with detailed error messages."""
    missing_keys = [key for key in required_keys if key not in config_dict or config_dict[key] is None]
    
    if missing_keys:
        from .exceptions import ConfigurationError
        raise ConfigurationError(
            message=f"Missing required configuration keys: {', '.join(missing_keys)}",
            details={
                'missing_keys': missing_keys,
                'available_keys': list(config_dict.keys()),
                'context': context
            }
        )
