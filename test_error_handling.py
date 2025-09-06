#!/usr/bin/env python3
"""Test script for error handling."""

from rich import print
from src.embeddings import embed_query
from src.exceptions import EmbeddingError

try:
    # This should raise an EmbeddingError for empty query
    embed_query('')
except EmbeddingError as e:
    print(f'[green]PASS:[/] Caught expected error: {e.error_code} - {e.message}')
    print(f'   Details: {e.details}')
except Exception as e:
    print(f'[red]FAIL:[/] Unexpected error type: {type(e).__name__} - {e}')
else:
    print('[red]FAIL:[/] No error was raised - this is unexpected!')

print('\n[cyan]Error handling test completed![/]')
