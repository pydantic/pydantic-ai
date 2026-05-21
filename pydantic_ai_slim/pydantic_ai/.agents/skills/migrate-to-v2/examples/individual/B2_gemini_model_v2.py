"""v2 form: GoogleModel + GoogleProvider."""
import os
os.environ.setdefault('GOOGLE_API_KEY', 'dummy')
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider


def trigger():
    return GoogleModel('gemini-1.5-pro', provider=GoogleProvider(api_key='dummy'))


if __name__ == '__main__':
    trigger()
