# Complete refactor of api_v2.py

import asyncio
import hashlib
import os
import signal
import subprocess
import sys
import time
import traceback
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Generator, Optional, Dict, Any, Union, List

import numpy as np
import psutil
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Response, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator

# Add project root to path
NOW_DIR = Path(__file__).parent.absolute()
sys.path.extend([str(NOW_DIR), str(NOW_DIR / "GPT_SoVITS")])

from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names

class MediaType(str, Enum):
    WAV = "wav"
    RAW = "raw"
    OGG = "ogg"
    AAC = "aac"

@dataclass
class AppConfig:
    UPLOAD_DIR: Path = Path("uploaded_audio")
    TEMP_DIR: Path = Path("temp")
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    CLEANUP_INTERVAL: int = 3600  # 1 hour
    MAX_FILE_AGE: int = 86400  # 24 hours
    THREAD_POOL_SIZE: int = max(4, (os.cpu_count() or 4) // 2)
    LRU_CACHE_SIZE: int = 128
    
    def __post_init__(self):
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        self.TEMP_DIR.mkdir(exist_ok=True)

@dataclass
class GPTSoVitsConfig:
    gpt_window_size: int = 1024
    max_symbols_per_phrase: int = 150
    hz_per_symbol: float = 150.0
    sample_rate: int = 44100
    bytes_per_sample: int = 4
    gpu_memory_fraction: float = 0.9
    cpu_memory_fraction: float = 0.8
    memory_per_char_base: int = 25000

class ResourceManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.lock = Lock()
        self._last_cleanup = datetime.now()
        self.executor = ThreadPoolExecutor(max_workers=config.THREAD_POOL_SIZE)

    async def cleanup_old_files(self):
        """Periodically clean up old files from upload and temp directories"""
        async with self.lock:
            now = datetime.now()
            if (now - self._last_cleanup).total_seconds() < self.config.CLEANUP_INTERVAL:
                return

            for directory in [self.config.UPLOAD_DIR, self.config.TEMP_DIR]:
                for file_path in directory.glob("*"):
                    if file_path.is_file():
                        file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_age.total_seconds() > self.config.MAX_FILE_AGE:
                            try:
                                file_path.unlink()
                            except OSError as e:
                                print(f"Error deleting {file_path}: {e}")

            self._last_cleanup = now

class BatchSizeManager:
    def __init__(self, config: GPTSoVitsConfig):
        self.config = config
        self.lock = Lock()
        self._last_refresh = datetime.now()
        self.refresh_interval = timedelta(minutes=5)
        self._available_memory = self._get_available_memory()

    def _format_memory_size(self, bytes_size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024 or unit == 'GB':
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024

    def _get_available_memory(self) -> int:
        with self.lock:
            if torch.cuda.is_available():
                memory_available = int(
                    torch.cuda.get_device_properties(0).total_memory * 
                    self.config.gpu_memory_fraction
                )
                print(f"GPU Memory available: {self._format_memory_size(memory_available)}")
                return memory_available

            memory_available = int(
                psutil.virtual_memory().total * 
                self.config.cpu_memory_fraction
            )
            print(f"System Memory available: {self._format_memory_size(memory_available)}")
            return memory_available

    def get_batch_size(self, text_length: int, streaming: bool = False) -> int:
        now = datetime.now()
        with self.lock:
            if now - self._last_refresh > self.refresh_interval:
                self._available_memory = self._get_available_memory()
                self._last_refresh = now

            # Calculate memory requirements with safety margin
            estimated_memory = text_length * self.config.memory_per_char_base * 1.2
            max_batch = max(1, int(self._available_memory / estimated_memory))

            # Adjust batch size based on text length and streaming mode
            if text_length > 500:
                max_batch = int(max_batch * 0.75)
            
            if streaming:
                max_batch = max(1, int(max_batch * 0.5))

            return max_batch

class AudioProcessor:
    @staticmethod
    def pack_ogg(data: np.ndarray, rate: int) -> BytesIO:
        buffer = BytesIO()
        with sf.SoundFile(buffer, mode='w', samplerate=rate, 
                         channels=1, format='ogg') as audio_file:
            audio_file.write(data)
        buffer.seek(0)
        return buffer

    @staticmethod
    def pack_wav(data: np.ndarray, rate: int) -> BytesIO:
        buffer = BytesIO()
        sf.write(buffer, data, rate, format='wav')
        buffer.seek(0)
        return buffer

    @staticmethod
    def pack_aac(data: np.ndarray, rate: int) -> BytesIO:
        buffer = BytesIO()
        chunk_size = 1024 * 1024  # 1MB chunks

        try:
            process = subprocess.Popen([
                'ffmpeg',
                '-f', 's16le',
                '-ar', str(rate),
                '-ac', '1',
                '-i', 'pipe:0',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-vn',
                '-f', 'adts',
                'pipe:1'
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Process in chunks
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                process.stdin.write(chunk.tobytes())
            
            process.stdin.close()
            out, err = process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {err.decode()}")
                
            buffer.write(out)
            buffer.seek(0)
            return buffer
            
        except Exception as e:
            raise RuntimeError(f"AAC encoding failed: {str(e)}")

    @staticmethod
    def create_wave_header(channels: int = 1, 
                          sample_width: int = 2, 
                          sample_rate: int = 32000) -> bytes:
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"")
        return buffer.getvalue()
        
class TTSRequest(BaseModel):
    text: str
    text_lang: str
    ref_audio_path: Optional[str] = None
    aux_ref_audio_paths: Optional[List[str]] = None
    prompt_text: str = ""
    prompt_lang: str
    top_k: int = 5
    top_p: float = Field(default=1.0, gt=0)
    temperature: float = Field(default=1.0, gt=0)
    text_split_method: str = "cut5"
    batch_size: int = Field(default=1, gt=0)
    batch_threshold: float = Field(default=0.75, gt=0)
    split_bucket: bool = True
    speed_factor: float = Field(default=1.0, gt=0)
    fragment_interval: float = Field(default=0.3, gt=0)
    seed: int = -1
    media_type: MediaType = MediaType.WAV
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = Field(default=1.35, gt=0)

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

    @field_validator('text_lang', 'prompt_lang')
    @classmethod
    def validate_language(cls, v: str) -> str:
        if not v:
            raise ValueError('Language cannot be empty')
        return v.lower()

    @field_validator('media_type', 'streaming_mode')
    @classmethod
    def validate_media_streaming(cls, v, info):
        values = info.data
        if 'media_type' in values and 'streaming_mode' in values:
            if values['media_type'] == MediaType.OGG and not values['streaming_mode']:
                raise ValueError('OGG format requires streaming mode')
        return v

class TTSService:
    def __init__(self, config_path: str, app_config: AppConfig):
        self.config = TTS_Config(config_path)
        self.pipeline = TTS(self.config)
        self.app_config = app_config
        self.batch_manager = BatchSizeManager(GPTSoVitsConfig())
        self.resource_manager = ResourceManager(app_config)
        self.audio_processor = AudioProcessor()
        self.supported_languages = set(lang.lower() for lang in self.config.languages)

    async def validate_languages(self, text_lang: str, prompt_lang: str):
        if text_lang.lower() not in self.supported_languages:
            raise ValueError(f"Unsupported text language: {text_lang}")
        if prompt_lang.lower() not in self.supported_languages:
            raise ValueError(f"Unsupported prompt language: {prompt_lang}")

    async def process_audio(self, audio_data: np.ndarray, 
                          sample_rate: int, 
                          media_type: MediaType) -> bytes:
        if media_type == MediaType.OGG:
            buffer = self.audio_processor.pack_ogg(audio_data, sample_rate)
        elif media_type == MediaType.AAC:
            buffer = self.audio_processor.pack_aac(audio_data, sample_rate)
        elif media_type == MediaType.WAV:
            buffer = self.audio_processor.pack_wav(audio_data, sample_rate)
        else:  # RAW
            buffer = BytesIO(audio_data.tobytes())
        
        return buffer.getvalue()

    async def generate_audio(self, request: TTSRequest) -> Generator:
        await self.validate_languages(request.text_lang, request.prompt_lang)
        
        # Optimize batch size
        optimal_batch_size = self.batch_manager.get_batch_size(
            len(request.text), 
            request.streaming_mode
        )
        request.batch_size = min(request.batch_size, optimal_batch_size)

        # Convert request to dict for pipeline
        req_dict = request.dict()
        req_dict['return_fragment'] = request.streaming_mode

        try:
            return self.pipeline.run(req_dict)
        except Exception as e:
            raise RuntimeError(f"TTS generation failed: {str(e)}")

APP = FastAPI(title="GPT-SoVITS API")

@APP.on_event("startup")
async def startup_event():
    APP.state.config = AppConfig()
    APP.state.tts_service = TTSService(
        config_path=os.getenv("TTS_CONFIG", "GPT_SoVITS/configs/tts_infer.yaml"),
        app_config=APP.state.config
    )

@APP.on_event("shutdown")
async def shutdown_event():
    if hasattr(APP.state, 'tts_service'):
        # Cleanup resources
        await APP.state.tts_service.resource_manager.executor.shutdown(wait=True)

@APP.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": [{"loc": err["loc"], "msg": err["msg"]} for err in exc.errors()]
        }
    )

@APP.get("/tts")
async def tts_get_endpoint(request: TTSRequest = Depends()):
    try:
        tts_service = APP.state.tts_service
        generator = await tts_service.generate_audio(request)

        if request.streaming_mode:
            async def stream_generator():
                if request.media_type == MediaType.WAV:
                    yield AudioProcessor.create_wave_header()
                
                for sample_rate, chunk in generator:
                    audio_data = await tts_service.process_audio(
                        chunk, sample_rate, 
                        MediaType.RAW if request.media_type == MediaType.WAV 
                        else request.media_type
                    )
                    yield audio_data

            return StreamingResponse(
                stream_generator(),
                media_type=f"audio/{request.media_type.value}"
            )
        else:
            sample_rate, audio_data = next(generator)
            processed_audio = await tts_service.process_audio(
                audio_data, sample_rate, request.media_type
            )
            return Response(
                processed_audio,
                media_type=f"audio/{request.media_type.value}"
            )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )
        
class FileUploadManager:
    def __init__(self, upload_dir: Path):
        self.upload_dir = upload_dir
        self._lock = Lock()
        self._async_lock = asyncio.Lock()  # Add asyncio Lock for async operations

    @lru_cache(maxsize=128)
    def _sanitize_filename(self, filename: str) -> str:
        """Create a safe filename while keeping the extension"""
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(f"{name}{timestamp}".encode()).hexdigest()[:10]
        return f"{name_hash}{ext}"

    async def save_file(self, file: UploadFile) -> Path:
        """Save uploaded file with thread safety and proper cleanup"""
        if not file.filename:
            raise ValueError("Filename is required")

        safe_filename = self._sanitize_filename(file.filename)
        file_path = self.upload_dir / safe_filename

        async with self._async_lock:  # Use the async lock here
            try:
                # Read in chunks to handle large files
                with file_path.open("wb") as buffer:
                    while chunk := await file.read(8192):  # 8KB chunks
                        buffer.write(chunk)
                return file_path
            except Exception as e:
                # Cleanup on failure
                if file_path.exists():
                    file_path.unlink()
                raise IOError(f"Failed to save file: {str(e)}")

class TTSPostRequest(BaseModel):
    text: str
    text_lang: str
    prompt_lang: str
    prompt_text: str = ""
    top_k: int = Field(default=5, gt=0)
    top_p: float = Field(default=1.0, gt=0)
    temperature: float = Field(default=1.0, gt=0)
    text_split_method: str = "cut5"
    batch_size: int = Field(default=1, gt=0)
    batch_threshold: float = Field(default=0.75, gt=0)
    split_bucket: bool = True
    speed_factor: float = Field(default=1.0, gt=0)
    fragment_interval: float = Field(default=0.3, gt=0)
    seed: int = -1
    media_type: MediaType = MediaType.WAV
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = Field(default=1.35, gt=0)

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

    @field_validator('text_lang', 'prompt_lang')
    @classmethod
    def validate_language(cls, v: str) -> str:
        if not v:
            raise ValueError('Language cannot be empty')
        return v.lower()

    @field_validator('media_type', 'streaming_mode')
    @classmethod
    def validate_media_streaming(cls, v, info):
        values = info.data
        if 'media_type' in values and 'streaming_mode' in values:
            if values['media_type'] == MediaType.OGG and not values['streaming_mode']:
                raise ValueError('OGG format requires streaming mode')
        return v

    class Config:
        use_enum_values = True

@APP.post("/tts")
async def tts_post_endpoint(
    text: str = Form(...),
    text_lang: str = Form(...),
    prompt_lang: str = Form(...),
    ref_audio_file: UploadFile = File(...),
    prompt_text: str = Form(""),
    top_k: int = Form(5),
    top_p: float = Form(1.0),
    temperature: float = Form(1.0),
    text_split_method: str = Form("cut5"),
    batch_size: int = Form(1),
    batch_threshold: float = Form(0.75),
    split_bucket: bool = Form(True),
    speed_factor: float = Form(0.7),
    fragment_interval: float = Form(0.3),
    seed: int = Form(42),
    media_type: str = Form("wav"),
    streaming_mode: bool = Form(False),
    parallel_infer: bool = Form(True),
    repetition_penalty: float = Form(1.35)
):
    try:
        file_manager = FileUploadManager(APP.state.config.UPLOAD_DIR)
        tts_service = APP.state.tts_service

        # Process file upload
        ref_audio_path = await file_manager.save_file(ref_audio_file)

        # Convert media_type string to enum
        try:
            media_type_enum = MediaType(media_type.lower())
        except ValueError:
            raise ValueError(f"Invalid media type: {media_type}")

        # Create full TTS request
        tts_request = TTSRequest(
            text=text,
            text_lang=text_lang,
            prompt_lang=prompt_lang,
            ref_audio_path=str(ref_audio_path),
            prompt_text=prompt_text,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            text_split_method=text_split_method,
            batch_size=batch_size,
            batch_threshold=batch_threshold,
            split_bucket=split_bucket,
            speed_factor=speed_factor,
            fragment_interval=fragment_interval,
            seed=seed,
            media_type=media_type_enum,
            streaming_mode=streaming_mode,
            parallel_infer=parallel_infer,
            repetition_penalty=repetition_penalty
        )

        # Generate audio using the same logic as GET endpoint
        generator = await tts_service.generate_audio(tts_request)

        if streaming_mode:
            async def stream_generator():
                try:
                    if media_type_enum == MediaType.WAV:
                        yield AudioProcessor.create_wave_header()
                    
                    for sample_rate, chunk in generator:
                        audio_data = await tts_service.process_audio(
                            chunk, sample_rate,
                            MediaType.RAW if media_type_enum == MediaType.WAV 
                            else media_type_enum
                        )
                        yield audio_data
                finally:
                    # Cleanup uploaded file after streaming is done
                    ref_audio_path.unlink(missing_ok=True)

            return StreamingResponse(
                stream_generator(),
                media_type=f"audio/{media_type_enum.value}"
            )
        else:
            try:
                sample_rate, audio_data = next(generator)
                processed_audio = await tts_service.process_audio(
                    audio_data, sample_rate, media_type_enum
                )
                return Response(
                    processed_audio,
                    media_type=f"audio/{media_type_enum.value}"
                )
            finally:
                # Cleanup uploaded file after processing
                ref_audio_path.unlink(missing_ok=True)

    except Exception as e:
        # Ensure cleanup on error
        if 'ref_audio_path' in locals():
            Path(ref_audio_path).unlink(missing_ok=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )

@APP.post("/set_weights")
async def set_weights(
    weight_type: str = Form(...),
    weights_path: str = Form(...)
):
    """Update model weights"""
    try:
        tts_service = APP.state.tts_service
        
        if not Path(weights_path).exists():
            raise ValueError(f"Weights file not found: {weights_path}")

        if weight_type.lower() == "gpt":
            await asyncio.to_thread(tts_service.pipeline.init_t2s_weights, weights_path)
        elif weight_type.lower() == "sovits":
            await asyncio.to_thread(tts_service.pipeline.init_vits_weights, weights_path)
        else:
            raise ValueError(f"Invalid weight type: {weight_type}")

        return JSONResponse(
            status_code=200,
            content={"message": f"Successfully updated {weight_type} weights"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )

@APP.post("/set_reference_audio")
async def set_reference_audio(
    audio_file: UploadFile = File(...),
):
    """Update reference audio"""
    try:
        file_manager = FileUploadManager(APP.state.config.UPLOAD_DIR)
        tts_service = APP.state.tts_service
        
        # Save the file permanently for future reference
        audio_path = await file_manager.save_file(audio_file)
        
        try:
            await asyncio.to_thread(tts_service.pipeline.set_ref_audio, str(audio_path))
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Successfully updated reference audio",
                    "reference_path": str(audio_path)
                }
            )
        except Exception as e:
            # Only cleanup the file if setting it as reference failed
            audio_path.unlink(missing_ok=True)
            raise e
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )

@APP.post("/control")
async def control_endpoint(command: str = Form(...)):
    """System control endpoint"""
    try:
        if command == "restart":
            # Graceful shutdown
            await shutdown_event()
            # Restart process
            os.execl(sys.executable, sys.executable, *sys.argv)
        elif command == "shutdown":
            # Graceful shutdown
            await shutdown_event()
            # Send termination signal
            os.kill(os.getpid(), signal.SIGTERM)
        else:
            raise ValueError(f"Unknown command: {command}")
            
        return JSONResponse(
            status_code=200,
            content={"message": f"Successfully executed command: {command}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-SoVITS API Server")
    parser.add_argument(
        "-c", "--tts_config",
        type=str,
        default="GPT_SoVITS/configs/tts_infer.yaml",
        help="Path to TTS config file"
    )
    parser.add_argument(
        "--bind_addr",
        type=str,
        default="127.0.0.1",
        help="Host to bind to"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=9880,
        help="Port to listen on"
    )
    
    args = parser.parse_args()
    
    # Set environment variable for TTS config
    os.environ["TTS_CONFIG"] = args.tts_config
    
    try:
        uvicorn.run(
            app=APP,
            host=None if args.bind_addr.lower() == "none" else args.bind_addr,
            port=args.port,
            workers=1
        )
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        sys.exit(1)   
