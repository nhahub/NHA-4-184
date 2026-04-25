import requests
import logging
import os
import time

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Handles text-to-speech using ElevenLabs API."""

    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
        self.base_url = "https://api.elevenlabs.io/v1"

        if not self.api_key:
            logger.warning("ELEVENLABS_API_KEY not set — TTS will be unavailable")
        else:
            logger.info(f"ElevenLabs TTS initialized: voice_id={self.voice_id}")

    @property
    def is_available(self) -> bool:
        """Check if TTS is configured and ready."""
        return bool(self.api_key)

    def synthesize(self, text: str) -> bytes:
        """Convert text to speech audio bytes (MP3).

        Args:
            text: The text to convert to speech.

        Returns:
            MP3 audio as bytes.

        Raises:
            ValueError: If API key is not configured.
            Exception: If ElevenLabs API returns an error.
        """
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not configured")

        logger.info(f"TTS synthesis started: text_len={len(text)}")
        start_time = time.time()

        url = f"{self.base_url}/text-to-speech/{self.voice_id}"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            logger.error(f"TTS failed: status={response.status_code}, response={response.text[:200]}")
            raise Exception(f"ElevenLabs API error: {response.status_code}")

        audio_bytes = response.content
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"TTS synthesis completed: audio_size={len(audio_bytes)} bytes, duration_ms={duration_ms}")

        return audio_bytes
