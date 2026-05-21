"""v1: GeminiModel via GoogleGLAProvider."""
import os
os.environ.setdefault('GEMINI_API_KEY', 'dummy')
from pydantic_ai.models.gemini import GeminiModel


def trigger():
    # DEPRECATION: B2_gemini_model
    return GeminiModel('gemini-1.5-pro')


EXPECT = 'Use `GoogleModel` instead'

if __name__ == '__main__':
    trigger()
