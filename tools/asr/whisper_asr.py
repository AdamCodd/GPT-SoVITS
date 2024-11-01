"""
WhisperTranscriber: A wrapper class for OpenAI's Whisper model that handles audio transcription and translation.
It provides:
- Single and batch audio file processing
- GPU acceleration with FP16 support
- Memory-efficient batch processing
- Progress tracking for batch operations
- Automatic resource cleanup
- Support for both transcription and translation tasks
"""
from transformers import pipeline
import torch
from typing import Union, List, Dict, Optional
import os
import librosa
import numpy as np

class WhisperPipelineTranscriber:
    def __init__(
        self,
        model_path: str = "openai/whisper-base",
        language: Optional[str] = None,
        use_fp16: Optional[bool] = None,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 if use_fp16 is not None else (self.device == "cuda")
        self.language = language
        self.sample_rate = 16000

        # Determine attention mechanism based on GPU capabilities
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name()
            # Use flash attention 2 for newer GPUs (Ampere and newer)
            use_flash_attention = any(arch in gpu_name.lower() for arch in ['ampere', 'ada', 'hopper'])
            attn_implementation = "flash_attention_2" if use_flash_attention else "sdpa"
        else:
            attn_implementation = "eager"  # Default for CPU

        try:
            model_kwargs = {
                "torch_dtype": torch.float16 if self.use_fp16 else torch.float32,
                "attn_implementation": attn_implementation
            }
            
            if self.language:
                model_kwargs["language"] = self.language

            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_path,
                device=self.device,
                model_kwargs=model_kwargs
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Whisper pipeline: {str(e)}")

    def _validate_audio_path(self, audio_path: str) -> None:
        """Validate audio file existence and basic integrity"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if os.path.getsize(audio_path) == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess a single audio file"""
        try:
            self._validate_audio_path(audio_path)
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            raise Exception(f"Failed to load audio file {audio_path}: {str(e)}")

    def detect_language(
        self,
        audio_input: Union[str, List[str]]
    ) -> Union[Dict[str, float], List[Dict[str, Dict[str, float]]]]:
        """
        Detect the language(s) spoken in the audio file(s)
        """
        try:
            if isinstance(audio_input, list):
                # Handle multiple files
                audio_inputs = [self._load_audio(path) for path in audio_input]
                predictions = self.lang_pipeline(audio_inputs)
                
                return [
                    {
                        "file": audio_path,
                        "languages": {
                            pred["label"]: pred["score"]
                            for pred in file_pred
                        }
                    }
                    for audio_path, file_pred in zip(audio_input, predictions)
                ]
            else:
                # Handle single file
                audio = self._load_audio(audio_input)
                predictions = self.lang_pipeline(audio)
                
                return {
                    pred["label"]: pred["score"]
                    for pred in predictions
                }

        except Exception as e:
            raise Exception(f"Language detection failed: {str(e)}")
    
    def transcribe(
        self,
        audio_input: Union[str, List[str]],
        translate: bool = False
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Transcribe single audio file or multiple files with optional translation
        """
        try:
            if isinstance(audio_input, list):
                # Handle multiple files
                audio_inputs = [self._load_audio(path) for path in audio_input]
                outputs = self.pipeline(
                    audio_inputs,
                    generate_kwargs={
                        "task": "translate" if translate else "transcribe",
                        "language": self.language if self.language else None
                    },
                    return_timestamps=False
                )
                
                return [
                    {
                        "file": audio_path,
                        "text": output["text"].strip(),
                        "type": "translation" if translate else "transcription"
                    }
                    for audio_path, output in zip(audio_input, outputs)
                ]
            else:
                # Handle single file
                audio = self._load_audio(audio_input)
                output = self.pipeline(
                    audio,
                    generate_kwargs={
                        "task": "translate" if translate else "transcribe",
                        "language": self.language if self.language else None
                    },
                    return_timestamps=False
                )
                
                return output["text"].strip()

        except Exception as e:
            raise Exception(f"Processing failed: {str(e)}")

    def cleanup(self):
        if hasattr(self, 'pipeline'):
            del self.pipeline
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

"""
Usage Examples:

# Initialize transcriber
transcriber = WhisperPipelineTranscriber(
    model_path="openai/whisper-base",
    language="fr"  # Optional: specify source language
)

1. # Single file transcription
result = transcriber.transcribe("audio.wav")

2. # Multiple files transcription with translation
results = transcriber.transcribe(
    ["audio1.wav", "audio2.wav"],
    translate=True, # Always translate to english regardless of the source language
)

### Language detection
# Detect language for a single file
lang_result = transcriber.detect_language("audio.wav")
# Returns: {"en": 0.98, "fr": 0.01, ...}

# Detect language for multiple files
lang_results = transcriber.detect_language(["audio1.wav", "audio2.wav"])
# Returns: [
#     {"file": "audio1.wav", "languages": {"en": 0.98, "fr": 0.01, ...}},
#     {"file": "audio2.wav", "languages": {"es": 0.95, "pt": 0.03, ...}}
# ]

"""
