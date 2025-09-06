# Configuration Parameters Guide

## What was improved

Previously, the code contained many "magic numbers" - hardcoded values scattered throughout the codebase. This made the system difficult to tune and maintain. All magic numbers have been extracted to `src/config.py` as configurable environment variables.

## Configuration Categories

### RAG Pipeline Core
- `RETRIEVAL_MIN_SCORE=0.45` - Minimum similarity threshold for chunks
- `KEYWORD_COVERAGE_THRESHOLD=0.25` - Required keyword coverage ratio  
- `STRICT_MODE_ENABLED=false` - Enable strict keyword enforcement

### Answer Extraction
- `LEXICAL_COVERAGE_WEIGHT=0.3` - Weight for keyword coverage in scoring
- `SHORT_SENTENCE_PENALTY=0.5` - Penalty for very short sentences
- `MIN_ANSWER_SCORE=0.2` - Minimum score to return answer
- `LEXICAL_RERANK_WEIGHT=0.08` - Weight for lexical reranking

### Definition Extraction  
- `MIN_DEFINITION_LENGTH=15` - Minimum definition sentence length
- `MAX_DEFINITION_LENGTH=400` - Maximum definition sentence length
- `DEFINITION_NAME_WEIGHT=0.3` - Weight boost for name mentions

### Text Processing
- `MIN_KEYWORD_LENGTH=2` - Minimum keyword length
- `MAX_ANSWER_LENGTH=1200` - Maximum answer length before truncation

### Retrieval Defaults
- `DEFAULT_RETRIEVAL_K=5` - Default number of chunks to retrieve
- `DEFAULT_ANSWER_K=3` - Default k for answer generation

### LLM Parameters
- `OPENAI_TEMPERATURE=0.1` - OpenAI temperature (0.0-1.0)
- `OPENAI_MAX_TOKENS=400` - Max tokens in OpenAI response

### API Configuration
- `API_HOST=127.0.0.1` - API server host
- `API_PORT=8000` - API server port
- `FRONTEND_VITE_PORT=5173` - Vite dev server port
- `FRONTEND_REACT_PORT=3000` - React dev server port

## How to use

1. **Default behavior**: All parameters have sensible defaults - no configuration required
2. **Custom tuning**: Add any parameter to your `.env` file to override defaults
3. **Example configurations**: Use provided `.env.*.example` files as starting points

## Benefits

- **No more magic numbers** - all values are clearly named and documented  
- **Easy tuning** - change behavior without touching code  
- **Environment-specific** - different configs for dev/staging/production  
- **Self-documenting** - parameter names explain their purpose  
- **Type-safe** - proper type conversion (int, float, bool)

## Migration

Old hardcoded values are now configurable:
- `0.45` → `RETRIEVAL_MIN_SCORE`
- `0.25` → `KEYWORD_COVERAGE_THRESHOLD`  
- `0.08` → `LEXICAL_RERANK_WEIGHT`
- etc.

Behavior remains identical unless you override the defaults.
