from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

class WhisperTranscriber:
    def __init__(self, model_path=None):
        """
        Initialize the Whisper transcriber with an optional model path
        """
        try:
            if model_path:
                self.processor = WhisperProcessor.from_pretrained(model_path)
                self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
            else:
                self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
                self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
        except Exception as e:
            raise Exception(f"Failed to load Whisper model: {str(e)}")
        
        # Set device (GPU if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def transcribe(self, audio_path):
        """
        Transcribe audio file to text
        Args:
            audio_path (str): Path to audio file
        Returns:
            str: Transcribed text
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Process audio
            input_features = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device)

            # Generate token ids
            predicted_ids = self.model.generate(
                input_features,
                task="transcribe",
                language=None,  # Auto detect language
                return_timestamps=False
            )
            
            # Decode the token ids to text
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")
