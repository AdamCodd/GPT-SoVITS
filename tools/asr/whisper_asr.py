"""
WhisperTranscriber: A wrapper class for OpenAI's Whisper model that handles audio transcription and translation.
It provides:
- Single and batch audio file processing
- GPU acceleration with FP16 support
- Memory-efficient batch processing
- Progress tracking for batch operations
- Automatic resource cleanup
- Support for both transcription and translation tasks

Return Types:
1. Single file processing (transcribe with string input):
   - Returns str: The transcribed/translated text
   
2. Batch processing (transcribe with list input or transcribe_batch):
   - Returns List[Dict[str, str]] where each dict contains:
     {
         "file": str,  # Original audio file path
         "text": str,  # Transcribed/translated text
         "type": str   # Either "transcription" or "translation"
     }
"""
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
from typing import Union, List, Dict, Optional, Callable
import numpy as np
import os

class WhisperTranscriber:
    def __init__(self, model_path=None, language=None, use_fp16=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = use_fp16 if use_fp16 is not None else (self.device == "cuda")
        
        if self.use_fp16 and self.device == "cpu":
            print("Warning: FP16 on CPU may not provide benefits. Switching to FP32.")
            self.use_fp16 = False

        try:
            model_id = model_path or "openai/whisper-base"
            self.processor = WhisperProcessor.from_pretrained(model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.use_fp16 else torch.float32
            ).to(self.device).eval()
        except Exception as e:
            raise Exception(f"Failed to load Whisper model: {str(e)}")
        
        self.language = language
        self.sample_rate = 16000

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

    def _generate_with_model(self, features: torch.Tensor, num_beams: int, task: str) -> torch.Tensor:
        """Generate model output with given parameters"""
        ctx = torch.cuda.amp.autocast(enabled=self.use_fp16) if self.device == "cuda" \
              else torch.cuda.amp.autocast(enabled=False)
        
        with ctx:
            predicted_ids = self.model.generate(
                features,
                task=task,
                language=self.language,
                return_timestamps=False,
                num_beams=num_beams,
                length_penalty=1.0,
                do_sample=False,
                early_stopping=True
            )
        return predicted_ids

    @torch.no_grad()
    def transcribe_batch(
        self, 
        audio_paths: List[str], 
        batch_size: int = 8, 
        num_beams: int = 5,
        translate: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[Dict[str, str]]:
        """
        Transcribe multiple audio files in parallel batches with optional translation
        Args:
            audio_paths (List[str]): List of paths to audio files
            batch_size (int): Number of audio files to process simultaneously
            num_beams (int): Number of beams for beam search
            translate (bool): Whether to translate to English (requires language to be set)
            progress_callback (Optional[Callable[[float], None]]): Optional callback for progress updates
        Returns:
            List[Dict[str, str]]: List of dictionaries containing file paths and transcriptions/translations
        """
        if translate and not self.language:
            raise ValueError("Language must be set for translation")

        try:
            results = []
            task = "translate" if translate else "transcribe"
            total_files = len(audio_paths)
            
            for i in range(0, total_files, batch_size):
                if progress_callback:
                    progress = (i / total_files) * 100
                    progress_callback(progress)

                batch_paths = audio_paths[i:i + batch_size]
                audio_inputs = [self._load_audio(path) for path in batch_paths]
                
                features = self.processor(
                    audio_inputs,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True
                ).input_features.to(self.device)

                predicted_ids = self._generate_with_model(features, num_beams, task)
                
                transcriptions = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )
                
                batch_results = [
                    {
                        "file": audio_path, 
                        "text": text.strip(),
                        "type": "translation" if translate else "transcription"
                    }
                    for audio_path, text in zip(batch_paths, transcriptions)
                ]
                results.extend(batch_results)
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            if progress_callback:
                progress_callback(100.0)
            
            return results
                
        except Exception as e:
            raise Exception(f"Batch processing failed: {str(e)}")

    @torch.no_grad()
    def transcribe(
        self, 
        audio_input: Union[str, List[str]], 
        batch_size: int = 8,
        num_beams: int = 5,
        translate: bool = False
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Transcribe single audio file or multiple files with optional translation
        Args:
            audio_input (Union[str, List[str]]): Single audio path or list of audio paths
            batch_size (int): Number of audio files to process simultaneously (for batch processing)
            num_beams (int): Number of beams for beam search
            translate (bool): Whether to translate to English (requires language to be set)
        Returns:
            Union[str, List[Dict[str, str]]]: Single transcription/translation or list of results
        """
        if translate and not self.language:
            raise ValueError("Language must be set for translation")

        if isinstance(audio_input, list):
            return self.transcribe_batch(audio_input, batch_size, num_beams, translate)
        
        try:
            audio = self._load_audio(audio_input)
            task = "translate" if translate else "transcribe"
            
            input_features = self.processor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features.to(self.device)

            predicted_ids = self._generate_with_model(input_features, num_beams, task)
            
            result = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            return result.strip()
            
        except Exception as e:
            raise Exception(f"Processing failed: {str(e)}")

    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

"""
Usage Examples:

1. Basic transcription of a single file:
transcriber = WhisperTranscriber()
text = transcriber.transcribe("audio.mp3")
print(text)  # Output: "The transcribed text"

2. Transcription with specific language:
transcriber = WhisperTranscriber(language="fr")
text = transcriber.transcribe("french_audio.mp3")
print(text)  # Output: "Le texte transcrit"

3. Translation to English:
transcriber = WhisperTranscriber(language="es")
text = transcriber.transcribe("spanish_audio.mp3", translate=True)
print(text)  # Output: "The translated text in English"

4. Batch processing with progress tracking:
def progress_handler(progress: float):
    print(f"Progress: {progress:.2f}%")

transcriber = WhisperTranscriber()
audio_files = ["file1.mp3", "file2.wav", "file3.ogg"]
results = transcriber.transcribe_batch(
    audio_files,
    batch_size=2,
    progress_callback=progress_handler
)
# Output: [
#     {"file": "file1.mp3", "text": "First transcription", "type": "transcription"},
#     {"file": "file2.wav", "text": "Second transcription", "type": "transcription"},
#     {"file": "file3.ogg", "text": "Third transcription", "type": "transcription"}
# ]

5. Using context manager for automatic cleanup:
with WhisperTranscriber(use_fp16=True) as transcriber:
    text = transcriber.transcribe("audio.mp3")
    print(text)  # Resources automatically cleaned up after this block

6. Using a different model:
transcriber = WhisperTranscriber(
    model_path="openai/whisper-large-v2",
    language="ja",
    use_fp16=True
)
text = transcriber.transcribe("japanese_audio.mp3")
print(text)  # Output: "日本語の文章"

7. Batch processing with translation:
audio_files = ["german1.mp3", "german2.mp3"]
with WhisperTranscriber(language="de") as transcriber:
    results = transcriber.transcribe_batch(
        audio_files,
        translate=True,
        num_beams=5
    )
    # Output: [
    #     {"file": "german1.mp3", "text": "First text in English", "type": "translation"},
    #     {"file": "german2.mp3", "text": "Second text in English", "type": "translation"}
    # ]

8. Using main transcribe method for both single and batch:
transcriber = WhisperTranscriber(language="fr")

# Single file - returns string
text = transcriber.transcribe("speech.mp3")
print(text)  # Output: "The transcribed text"

# Multiple files - returns list of dicts
results = transcriber.transcribe(["speech1.mp3", "speech2.mp3"])
# Output: [
#     {"file": "speech1.mp3", "text": "First text", "type": "transcription"},
#     {"file": "speech2.mp3", "text": "Second text", "type": "transcription"}
# ]

Note: 
- Supported audio formats depend on librosa (mp3, wav, ogg, flac, etc.)
- GPU acceleration is automatic if available
- FP16 is enabled by default on GPU for better performance and memory usage
- Batch size should be adjusted based on available memory
"""
