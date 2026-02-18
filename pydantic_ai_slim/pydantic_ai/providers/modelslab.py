"""
ModelsLab Provider for Pydantic AI

A comprehensive OpenAI-compatible provider that integrates ModelsLab's AI APIs
with Pydantic AI's agent framework. Enables both text generation and multi-modal
content creation through a unified, type-safe interface.

Usage:
    from pydantic_ai import Agent
    from modelslab_provider import ModelsLabProvider
    
    # Using provider shorthand
    agent = Agent('modelslab:gpt-4')
    
    # Using provider class directly  
    from pydantic_ai.models.openai import OpenAIChatModel
    
    model = OpenAIChatModel(
        'gpt-4', 
        provider=ModelsLabProvider(api_key='your_api_key')
    )
    agent = Agent(model)
"""

import os
from typing import Optional, Dict, Any
import httpx
from pydantic_ai.providers.openai import OpenAIProvider


class ModelsLabProvider(OpenAIProvider):
    """
    ModelsLab provider for Pydantic AI using OpenAI-compatible API.
    
    Supports ModelsLab's uncensored chat API which is OpenAI-compatible,
    enabling seamless integration with Pydantic AI's OpenAIChatModel.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://modelslab.com/api/v6",
        http_client: Optional[httpx.AsyncClient] = None,
        **kwargs
    ):
        """
        Initialize ModelsLab provider.
        
        Args:
            api_key: ModelsLab API key. If not provided, reads from MODELSLAB_API_KEY env var
            base_url: ModelsLab API base URL (default: https://modelslab.com/api/v6)
            http_client: Optional custom HTTP client
            **kwargs: Additional arguments passed to parent OpenAIProvider
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv('MODELSLAB_API_KEY')
            if not api_key:
                raise ValueError(
                    "ModelsLab API key is required. Set MODELSLAB_API_KEY environment variable "
                    "or pass api_key parameter. Get your key from https://modelslab.com"
                )
        
        # ModelsLab uses different API format, so we need to override the base_url
        # to point to their uncensored chat endpoint
        modelslab_chat_url = f"{base_url.rstrip('/')}/uncensored_chat"
        
        # Initialize parent OpenAIProvider with ModelsLab endpoints
        super().__init__(
            api_key=api_key,
            base_url=modelslab_chat_url,
            http_client=http_client,
            **kwargs
        )
        
        # Store original ModelsLab base URL for potential multi-modal extensions
        self._modelslab_base_url = base_url
    
    @property
    def modelslab_base_url(self) -> str:
        """Get the base ModelsLab API URL for multi-modal endpoints."""
        return self._modelslab_base_url
    
    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"ModelsLabProvider(base_url='{self._modelslab_base_url}')"


# Extended provider with multi-modal capabilities
class ModelsLabMultiModalProvider(ModelsLabProvider):
    """
    Extended ModelsLab provider with multi-modal content generation capabilities.
    
    This provider extends the basic chat functionality with image, video, and audio
    generation methods that can be used by custom agents or tools.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://modelslab.com/api/v6",
        http_client: Optional[httpx.AsyncClient] = None,
        enable_image_generation: bool = True,
        enable_video_generation: bool = True,
        enable_audio_generation: bool = True,
        **kwargs
    ):
        """
        Initialize ModelsLab multi-modal provider.
        
        Args:
            api_key: ModelsLab API key
            base_url: ModelsLab API base URL
            http_client: Optional custom HTTP client
            enable_image_generation: Enable image generation capabilities
            enable_video_generation: Enable video generation capabilities  
            enable_audio_generation: Enable audio/TTS generation capabilities
            **kwargs: Additional arguments
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            **kwargs
        )
        
        self.enable_image_generation = enable_image_generation
        self.enable_video_generation = enable_video_generation
        self.enable_audio_generation = enable_audio_generation
        
        # Set up HTTP client for multi-modal requests
        self._http_client = http_client or httpx.AsyncClient()
    
    async def generate_image(
        self,
        prompt: str,
        model_id: str = "flux",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an image using ModelsLab's text-to-image API.
        
        Args:
            prompt: Text description of the image to generate
            model_id: Model to use for generation (default: "flux")
            width: Image width in pixels
            height: Image height in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing generation result with image URL(s)
        """
        if not self.enable_image_generation:
            raise ValueError("Image generation is disabled for this provider")
        
        url = f"{self.modelslab_base_url}/images/text2img"
        
        payload = {
            "key": self.api_key,
            "model_id": model_id,
            "prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt", "blurry, low quality, distorted"),
            "width": width,
            "height": height,
            "samples": kwargs.get("samples", 1),
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "enhance_prompt": kwargs.get("enhance_prompt", "yes"),
            "safety_checker": kwargs.get("safety_checker", "yes"),
            "webhook": kwargs.get("webhook"),
            "track_id": kwargs.get("track_id")
        }
        
        response = await self._http_client.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    async def generate_video(
        self,
        prompt: str,
        model_id: str = "zeroscope",
        width: int = 576,
        height: int = 320,
        num_frames: int = 24,
        num_inference_steps: int = 20,
        guidance_scale: float = 9.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a video using ModelsLab's text-to-video API.
        
        Args:
            prompt: Text description of the video to generate
            model_id: Model to use for generation (default: "zeroscope")
            width: Video width in pixels
            height: Video height in pixels
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing generation result with video URL
        """
        if not self.enable_video_generation:
            raise ValueError("Video generation is disabled for this provider")
        
        url = f"{self.modelslab_base_url}/video/text2video"
        
        payload = {
            "key": self.api_key,
            "model_id": model_id,
            "prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt", "low quality, blurry, pixelated"),
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "webhook": kwargs.get("webhook"),
            "track_id": kwargs.get("track_id")
        }
        
        response = await self._http_client.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    async def generate_audio(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_multilingual_v2",
        voice_settings: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate audio using ModelsLab's text-to-speech API.
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use for generation
            model_id: TTS model to use
            voice_settings: Voice configuration settings
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing generation result with audio URL
        """
        if not self.enable_audio_generation:
            raise ValueError("Audio generation is disabled for this provider")
        
        if voice_settings is None:
            voice_settings = {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "style": 0.0,
                "use_speaker_boost": True
            }
        
        url = f"{self.modelslab_base_url}/tts"
        
        payload = {
            "key": self.api_key,
            "voice_id": voice_id,
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings
        }
        payload.update(kwargs)
        
        response = await self._http_client.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    async def close(self):
        """Close the HTTP client."""
        if hasattr(self, '_http_client'):
            await self._http_client.aclose()
    
    def __repr__(self) -> str:
        """String representation of the provider."""
        capabilities = []
        if self.enable_image_generation:
            capabilities.append("image")
        if self.enable_video_generation:
            capabilities.append("video")
        if self.enable_audio_generation:
            capabilities.append("audio")
        
        return f"ModelsLabMultiModalProvider(capabilities={capabilities})"


# Provider registration for automatic discovery
def register_modelslab_provider():
    """
    Register ModelsLab provider with Pydantic AI for automatic discovery.
    
    This allows using the 'modelslab:model-name' syntax with Agent initialization.
    """
    try:
        from pydantic_ai.providers import PROVIDER_REGISTRY
        
        # Register the provider
        PROVIDER_REGISTRY['modelslab'] = ModelsLabProvider
        
        # Register multi-modal variant
        PROVIDER_REGISTRY['modelslab-mm'] = ModelsLabMultiModalProvider
        
    except ImportError:
        # Provider registry might not be available in all versions
        # In this case, users need to instantiate providers manually
        pass


# Automatically register providers when module is imported
register_modelslab_provider()


# Export main classes
__all__ = [
    "ModelsLabProvider", 
    "ModelsLabMultiModalProvider", 
    "register_modelslab_provider"
]