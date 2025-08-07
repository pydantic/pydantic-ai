# Ollama Local Integration Example

This example demonstrates how to use Pydantic AI with local Ollama models for semantic analysis and structured output generation.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install pydantic-ai with Ollama support
pip install "pydantic-ai[ollama]"

# Or install from source
pip install .
```

### 2. Setup Ollama

```bash
# Install Ollama (if not already installed)
# Visit https://ollama.ai for installation instructions

# Start Ollama server
ollama serve

# Download a model (in another terminal)
ollama pull phi3
```

### 3. Configure the Example

```bash
# Copy the example configuration
cp config.yaml.example config.yaml

# Edit config.yaml to match your setup
# - Change model_name if using a different model
# - Adjust ollama_url if running on a different port
```

### 4. Run the Example

```bash
# Test the integration
python semantic.py

# Or use in your own code
python -c "
from semantic import analyze
from pydantic import BaseModel
from typing import List, Optional

class SearchQuery(BaseModel):
    queries: List[str]
    domain: Optional[str] = None
    year_filter: Optional[int] = None
    tags: List[str] = []

result = analyze('Find documents about AI in 2023', SearchQuery)
print(result.model_dump_json(indent=2))
"
```

## 📁 Files

- **`config.yaml.example`** - Configuration template
- **`system_prompt.txt`** - System prompt for semantic translation
- **`semantic.py`** - Main integration module
- **`README.md`** - This file

## 🔧 Configuration

### config.yaml

```yaml
provider: "openai"              # Use OpenAI-compatible client
model_name: "phi3"              # Your local Ollama model
temperature: 0.0                # Creativity level

# Ollama settings
ollama_url: "http://localhost:11434"
ollama_api_key: "ollama"
```

### Available Models

You can use any Ollama model by changing `model_name`:

```bash
# Download different models
ollama pull llama3.2
ollama pull mistral
ollama pull gemma
ollama pull dolphin-mixtral
```

Then update your `config.yaml`:
```yaml
model_name: "llama3.2"  # or "mistral", "gemma", etc.
```

## 📖 Usage Examples

### Basic Usage

```python
from semantic import analyze
from pydantic import BaseModel
from typing import List, Optional

class SearchQuery(BaseModel):
    queries: List[str]
    domain: Optional[str] = None
    year_filter: Optional[int] = None
    tags: List[str] = []

# Analyze a query
result = analyze(
    "Find documents about the financial crisis in 2008",
    SearchQuery
)

print(result.queries)      # ["financial", "crisis", "2008"]
print(result.year_filter)  # 2008
print(result.tags)         # ["economics"]
```

### Custom Schema

```python
from semantic import analyze
from pydantic import BaseModel
from typing import List

class DocumentClassification(BaseModel):
    category: str
    confidence: float
    tags: List[str]
    summary: str

# Classify a document
result = analyze(
    "This document discusses machine learning algorithms in healthcare",
    DocumentClassification
)

print(result.category)     # "technology"
print(result.confidence)   # 0.92
```

### Quick Analysis

```python
from semantic import quick_analyze

# Get structured output as dictionary
result = quick_analyze("Find AI research papers from 2022")
print(result)
# {
#   "queries": ["AI", "research", "papers"],
#   "domain": null,
#   "year_filter": 2022,
#   "tags": ["technology"]
# }
```

## 🔍 Troubleshooting

### Common Issues

1. **"Connection refused" error**
   ```bash
   # Make sure Ollama is running
   ollama serve
   ```

2. **"Model not found" error**
   ```bash
   # Download the model first
   ollama pull phi3
   ```

3. **"Configuration file not found" error**
   ```bash
   # Copy the example config
   cp config.yaml.example config.yaml
   ```

4. **"Invalid API key" error**
   - This is normal for Ollama - the integration handles it automatically
   - If you see this, check that Ollama is running and accessible

### Debug Mode

Enable verbose output in `config.yaml`:
```yaml
verbose: true
```

## 🧪 Testing

Run the built-in test:
```bash
python semantic.py
```

Expected output:
```
✅ Analysis successful!
Input: Find documents about machine learning in 2023
Output: {'queries': ['machine', 'learning', '2023'], 'domain': None, 'year_filter': 2023, 'tags': ['technology']}
```

## 🔗 Integration with Pydantic AI

This example shows how to:

1. **Configure local models** with Pydantic AI
2. **Use custom schemas** for structured output
3. **Handle environment setup** automatically
4. **Provide fallback mechanisms** for errors

The `analyze()` function wraps Pydantic AI's `Agent` with Ollama-specific configuration, making it easy to use local models with the same interface as cloud providers.

## 📚 Next Steps

- Try different models and compare results
- Create custom schemas for your use case
- Integrate with your existing Pydantic AI workflows
- Explore other Pydantic AI examples

---

**Happy coding with local AI! 🚀**
