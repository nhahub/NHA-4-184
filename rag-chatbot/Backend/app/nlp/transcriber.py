import whisper
import logging
import time
import tempfile
import os

logger = logging.getLogger(__name__)


class Transcriber:
    """Handles speech-to-text using OpenAI Whisper."""

    def __init__(self, model_name: str = "small"):
        logger.info(f"Loading Whisper model: {model_name}")
        start_time = time.time()
        self.model = whisper.load_model(model_name)
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Whisper model loaded successfully: model={model_name}, duration_ms={duration_ms}")

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.webm") -> dict:
        """Transcribe audio bytes to text."""
        logger.info(f"Transcription started: file_size={len(audio_bytes)} bytes")
        start_time = time.time()

        # Save to temp file (Whisper needs a file path)
        suffix = os.path.splitext(filename)[-1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = self.model.transcribe(tmp_path, language="en", fp16=False)
            text = result["text"].strip()
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.info(f"Transcription completed: text_len={len(text)}, duration_ms={duration_ms}")
            return {"text": text, "language": result.get("language", "en")}
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}", exc_info=True)
            raise
        finally:
            os.unlink(tmp_path)
