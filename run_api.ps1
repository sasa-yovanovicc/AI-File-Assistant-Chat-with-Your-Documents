# Quick helper to run API
param(
    [int]$Port = 8000
)
python -m uvicorn src.api:app --reload --port $Port
