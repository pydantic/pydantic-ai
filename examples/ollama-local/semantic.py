#!/usr/bin/env python3
"""
Ollama Local Integration for Pydantic AI
Provides a simple interface for running semantic analysis with local Ollama models.
"""

import os
import yaml
from typing import Optional, Type, TypeVar
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

T = TypeVar('T', bound=BaseModel)

class OllamaModel:
    """Ollama model configuration and setup"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Ollama model with configuration"""
        self.config = self._load_config(config_path)
        self._setup_environment()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    def _setup_environment(self):
        """Setup environment variables for Ollama"""
        ollama_url = self.config["ollama_url"].rstrip("/")
        ollama_key = self.config["ollama_api_key"]
        
        # Set environment variables for OpenAI client
        os.environ["OPENAI_API_BASE"] = f"{ollama_url}/v1"
        os.environ["OPENAI_API_KEY"] = ollama_key
        
        if self.config.get("verbose", False):
            print(f"🔧 Configured for Ollama at: {ollama_url}/v1")
    
    def get_model_name(self) -> str:
        """Get the configured model name"""
        return self.config["model_name"]
    
    def get_temperature(self) -> float:
        """Get the configured temperature"""
        return self.config.get("temperature", 0.0)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt from file"""
        try:
            with open("system_prompt.txt", "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            # Fallback to default prompt
            return "You are a semantic‐layer translator. Convert natural language queries into structured search queries."

def analyze(
    prompt: str,
    output_type: Type[T],
    config_path: str = "config.yaml",
    system_prompt: Optional[str] = None
) -> T:
    """
    Analyze a prompt using local Ollama model and return structured output.
    
    Args:
        prompt: The input text to analyze
        output_type: The Pydantic model class for structured output
        config_path: Path to configuration file
        system_prompt: Optional custom system prompt (overrides file)
    
    Returns:
        Structured output as Pydantic model instance
    
    Example:
        from pydantic import BaseModel
        from typing import List, Optional
        
        class SearchQuery(BaseModel):
            queries: List[str]
            domain: Optional[str] = None
            year_filter: Optional[int] = None
            tags: List[str] = []
        
        result = analyze(
            "Find documents about the financial crisis in 2008",
            SearchQuery
        )
        print(result.queries)  # ["financial", "crisis", "2008"]
    """
    # Initialize Ollama model
    ollama = OllamaModel(config_path)
    
    # Get system prompt
    if system_prompt is None:
        system_prompt = ollama.get_system_prompt()
    
    # Create agent with Ollama model
    model_name = f"openai:{ollama.get_model_name()}"
    agent = Agent(model_name)
    
    # Run analysis
    result = agent.run_sync(
        prompt,
        output_type=output_type,
        model_settings=ModelSettings(
            temperature=ollama.get_temperature()
        ),
    )
    
    return result.output

# Convenience function for quick testing
def quick_analyze(prompt: str, config_path: str = "config.yaml") -> dict:
    """
    Quick analysis function that returns a simple SearchQuery structure.
    
    Args:
        prompt: The input text to analyze
        config_path: Path to configuration file
    
    Returns:
        Dictionary with structured search query
    """
    from pydantic import BaseModel
    from typing import List, Optional
    
    class SearchQuery(BaseModel):
        queries: List[str]
        domain: Optional[str] = None
        year_filter: Optional[int] = None
        tags: List[str] = []
    
    result = analyze(prompt, SearchQuery, config_path)
    return result.model_dump()

if __name__ == "__main__":
    # Example usage
    test_prompt = "Find documents about machine learning in 2023"
    
    try:
        result = quick_analyze(test_prompt)
        print("✅ Analysis successful!")
        print(f"Input: {test_prompt}")
        print(f"Output: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("1. Ollama is running (ollama serve)")
        print("2. Model is downloaded (ollama pull phi3)")
        print("3. config.yaml exists and is properly configured")
