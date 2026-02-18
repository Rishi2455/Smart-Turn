from abc import ABC, abstractmethod
import os
import asyncio
from groq import Groq
from faster_whisper import WhisperModel
import numpy as np

class ASRService(ABC):
    @abstractmethod
    async def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data (float32 numpy array) to text."""
        pass

class GroqASR(ASRService):
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "whisper-large-v3-turbo"
        print(f"Initialized Groq ASR with model: {self.model}")

    async def transcribe(self, audio_data: np.ndarray) -> str:
        import io
        import wave

        audio_int16 = (audio_data * 32768).astype(np.int16)

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_int16.tobytes())

        wav_io.seek(0)
        wav_io.name = "audio.wav"

        try:
            loop = asyncio.get_event_loop()

            def _call_groq():
                return self.client.audio.transcriptions.create(
                    file=("audio.wav", wav_io),
                    model=self.model,
                    prompt="",
                    response_format="verbose_json",
                    language="en",
                    temperature=0.0
                )

            transcription = await loop.run_in_executor(None, _call_groq)

            if hasattr(transcription, 'text'):
                return transcription.text.strip()
            elif isinstance(transcription, dict) and 'text' in transcription:
                return transcription['text'].strip()
            else:
                return str(transcription).strip()

        except Exception as e:
            print(f"Groq Transcription Error: {e}")
            import traceback
            traceback.print_exc()
            return ""

class LocalWhisperASR(ASRService):
    def __init__(self, model_size="base"):
        print(f"Loading Local Whisper ({model_size})...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Local Whisper Loaded.")

    async def transcribe(self, audio_data: np.ndarray) -> str:
        try:
            loop = asyncio.get_event_loop()

            def _transcribe():
                segments, info = self.model.transcribe(audio_data, beam_size=1, language="en")
                return " ".join([s.text for s in segments]).strip()

            return await loop.run_in_executor(None, _transcribe)
        except Exception as e:
            print(f"Local Whisper Error: {e}")
            return ""
