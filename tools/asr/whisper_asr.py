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
from pathlib import Path

class WhisperPipelineTranscriber:
    def __init__(
        self,
        model_path: str = "openai/whisper-large-v3",
        language: Optional[str] = None,
        use_fp16: Optional[bool] = None,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 if use_fp16 is not None else (self.device == "cuda")
        self.language = language
        self.sample_rate = 16000

        # Set up local model paths relative to this file's location
        current_file_dir = Path(__file__).parent  # gets tools/asr directory
        self.local_model_dir = current_file_dir / "models"
        self.model_name = model_path.split('/')[-1]  # Get just 'whisper-large-v3'
        
        # Define possible local paths
        possible_paths = [
            self.local_model_dir / self.model_name,  # tools/asr/models/whisper-large-v3
            self.local_model_dir / model_path.replace('/', '-'),  # tools/asr/models/openai-whisper-large-v3
        ]

        # Try to find existing local model
        self.local_model_path = None
        for path in possible_paths:
            if path.exists() and (path / "config.json").exists():  # Verify it's a valid model directory
                self.local_model_path = path
                print(f"Found local model at: {path}")
                break

        if not self.local_model_path:
            print(f"Local model not found, will download from Hugging Face: {model_path}")

        # Determine attention mechanism based on GPU capabilities
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name()
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

            # Use local path if found, otherwise use HF model path
            effective_model_path = str(self.local_model_path) if self.local_model_path else model_path
            
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=effective_model_path,
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
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Detect the primary language spoken in the audio file(s).
        """
        try:
            if isinstance(audio_input, list):
                # Handle multiple files
                audio_inputs = [self._load_audio(path) for path in audio_input]
                predictions = self.lang_pipeline(audio_inputs)
                
                # Simplify output to only return top language for each file
                return [
                    {
                        "file": audio_path,
                        "languages": max(
                            file_pred, key=lambda x: x["score"]
                        )["label"]
                    }
                    for audio_path, file_pred in zip(audio_input, predictions)
                ]
            else:
                # Handle single file
                audio = self._load_audio(audio_input)
                predictions = self.lang_pipeline(audio)
                
                # Return the language with the highest score
                return max(predictions, key=lambda x: x["score"])["label"]

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
    model_path="openai/whisper-base", # Optional: Will use the default model 'whisper-v3-large' if not provided
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
# Returns: "en"

# Detect language for multiple files
lang_results = transcriber.detect_language(["audio1.wav", "audio2.wav"])
# Returns: [
#     {"file": "audio1.wav", "languages": "en"},
#     {"file": "audio2.wav", "languages": "en"}
# ]

"""
