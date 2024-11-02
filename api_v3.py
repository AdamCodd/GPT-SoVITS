import os
import io
import sys
import traceback
from typing import Dict, Generator, Optional, Union, List

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException, Response, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from tools.asr.whisper_asr import WhisperPipelineTranscriber
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from tools.audio_checker import AudioQualityChecker, QualityCheckConfig, AudioQualityParams
from pydantic import BaseModel
from datetime import datetime
import hashlib
from pathlib import Path
import shutil

# Constants
SUPPORTED_MEDIA_TYPES = {"wav", "raw", "ogg", "aac"}
UPLOAD_DIR = Path("uploaded_audio")
UPLOAD_DIR.mkdir(exist_ok=True)

i18n = I18nAuto()
cut_method_names = get_cut_method_names()

# Command line argument setup
parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
print(tts_config)
tts_pipeline = TTS(tts_config)

APP = FastAPI()

class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_file: Optional[UploadFile] = None
    ref_audio_path: Optional[str] = None
    aux_ref_audio_files: Optional[List[UploadFile]] = []
    aux_ref_audio_paths: Optional[List[str]] = []
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 5
    min_p: float = 0.0
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: Optional[str] = None
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35

    class Config:
        arbitrary_types_allowed = True

    async def process_audio_files(self) -> tuple[str, list[str]]:
        # Handle main reference file/path
        if self.ref_audio_file and self.ref_audio_path:
            raise ValueError("Cannot provide both ref_audio_file and ref_audio_path")
        
        main_ref_path = (await save_uploaded_file(self.ref_audio_file) 
                        if self.ref_audio_file 
                        else self.ref_audio_path)

        # Handle auxiliary files/paths
        if self.aux_ref_audio_files and self.aux_ref_audio_paths:
            raise ValueError("Cannot provide both aux_ref_audio_files and aux_ref_audio_paths")
        
        aux_paths = []
        if self.aux_ref_audio_files:
            aux_paths = [str(await save_uploaded_file(file)) for file in self.aux_ref_audio_files]
        elif self.aux_ref_audio_paths:
            aux_paths = self.aux_ref_audio_paths

        return main_ref_path, aux_paths

    async def process_asr(self, main_ref_path: str) -> None:
        """Process ASR if needed and populate prompt_lang and prompt_text"""
        if not self.prompt_lang or not self.prompt_text:
            transcriber = WhisperPipelineTranscriber()
            
            if not self.prompt_text:
                self.prompt_text = transcriber.transcribe(main_ref_path)
            
            if not self.prompt_lang:
                self.prompt_lang = transcriber.detect_language(main_ref_path)


def wave_header_chunk(frame_input: bytes = b"", channels: int = 1, 
                     sample_width: int = 2, sample_rate: int = 32000) -> bytes:
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()

class AudioProcessor:
    @staticmethod
    def pack_audio(data: np.ndarray, rate: int, media_type: str) -> BytesIO:
        io_buffer = BytesIO()
        try:
            packers = {
                "wav": AudioProcessor.pack_wav,
                "raw": AudioProcessor.pack_raw
            }
            packer = packers.get(media_type, AudioProcessor.pack_raw)
            io_buffer = packer(io_buffer, data, rate)
            io_buffer.seek(0)
            return io_buffer
        except Exception as e:
            io_buffer.close()
            raise e

    @staticmethod
    def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
        try:
            wav_data = wave_header_chunk(
                frame_input=data.tobytes(),
                channels=1,
                sample_width=2,
                sample_rate=rate
            )
            io_buffer.write(wav_data)
            return io_buffer
        except Exception as e:
            raise e

    @staticmethod
    def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
        try:
            io_buffer.write(data.tobytes())
            return io_buffer
        except Exception as e:
            raise e

def handle_control(command: str) -> None:
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)

def check_params(req: Dict) -> Optional[JSONResponse]:
    text = req.get("text", "")
    text_lang = req.get("text_lang", "").lower()
    ref_audio_path = req.get("ref_audio_path", "")
    prompt_lang = req.get("prompt_lang", "").lower()
    media_type = req.get("media_type", "wav")
    streaming_mode = req.get("streaming_mode", False)
    text_split_method = req.get("text_split_method", "cut5")

    # Basic validation
    if not text:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if not ref_audio_path:
        return JSONResponse(status_code=400, content={"message": "Must provide ref_audio_path"})
        
    # Language validation
    if not text_lang:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang not in tts_config.languages:
        return JSONResponse(status_code=400, content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"})
    if prompt_lang and prompt_lang not in tts_config.languages:
        return JSONResponse(status_code=400, content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"})
    
    # Media type validation
    if media_type not in SUPPORTED_MEDIA_TYPES:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    elif media_type == "ogg" and not streaming_mode:
        return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})
    
    # Text split method validation
    if text_split_method not in cut_method_names:
        return JSONResponse(status_code=400, content={"message": f"text_split_method: {text_split_method} is not supported"})

    return None

def sanitize_filename(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_hash = hashlib.md5(f"{name}{timestamp}".encode()).hexdigest()[:10]
    return f"{name_hash}{ext}"

async def save_uploaded_file(file: UploadFile) -> Path:
    safe_filename = sanitize_filename(file.filename)
    file_path = UPLOAD_DIR / safe_filename
    
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    
    return file_path

async def tts_handle(req: Dict) -> Union[StreamingResponse, Response, JSONResponse]:
    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type = req.get("media_type", "wav")

    if streaming_mode or return_fragment:
        req["return_fragment"] = True
    
    try:
        tts_generator = tts_pipeline.run(req)
        
        if streaming_mode:
            async def streaming_generator(tts_generator: Generator, media_type: str):
                try:
                    if media_type == "wav":
                        yield wave_header_chunk()
                        media_type = "raw"
                    async for sr, chunk in tts_generator:
                        yield AudioProcessor.pack_audio(chunk, sr, media_type).getvalue()
                except Exception as e:
                    # Clean up generator
                    if hasattr(tts_generator, 'close'):
                        tts_generator.close()
                    raise e
                finally:
                    # Ensure cleanup happens
                    if hasattr(tts_generator, 'aclose'):
                        await tts_generator.aclose()
            
            return StreamingResponse(
                streaming_generator(tts_generator, media_type),
                media_type=f"audio/{media_type}",
                status_code=200
            )
        else:
            try:
                sr, audio_data = next(tts_generator)
                audio_data = AudioProcessor.pack_audio(audio_data, sr, media_type).getvalue()
                return Response(audio_data, media_type=f"audio/{media_type}", status_code=200)
            finally:
                if hasattr(tts_generator, 'close'):
                    tts_generator.close()
    except Exception as e:
        # Clean up any remaining resources
        if 'tts_generator' in locals() and hasattr(tts_generator, 'close'):
            tts_generator.close()
        return JSONResponse(
            status_code=500,
            content={
                "message": "TTS processing failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

# FastAPI endpoints
@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)

@APP.get("/tts")
async def tts_get_endpoint(**params):
    return await tts_handle(params)

@APP.post("/tts")
async def tts_post_endpoint(
    text: str = Form(...),
    text_lang: str = Form(...),
    ref_audio_file: Optional[UploadFile] = File(None),
    ref_audio_path: Optional[str] = Form(None),
    aux_ref_audio_files: List[UploadFile] = File(default=[]),
    aux_ref_audio_paths: Optional[List[str]] = Form(None),
    prompt_lang: Optional[str] = Form(None),
    prompt_text: str = Form(""),
    top_k: int = Form(5),
    min_p: float = Form(0.0),
    top_p: float = Form(1.0),
    temperature: float = Form(1.0),
    text_split_method: Optional[str] = Form(None),
    batch_size: int = Form(1),
    batch_threshold: float = Form(0.75),
    split_bucket: bool = Form(True),
    speed_factor: float = Form(1.0),
    fragment_interval: float = Form(0.3),
    seed: int = Form(-1),
    media_type: str = Form("wav"),
    streaming_mode: bool = Form(False),
    parallel_infer: bool = Form(True),
    repetition_penalty: float = Form(1.35)
):
    try:
        # Debug printing
        if ref_audio_file:
            print(f"Received main reference file: {ref_audio_file.filename}")
        if aux_ref_audio_files:
            print(f"Received {len(aux_ref_audio_files)} auxiliary files")
            for idx, aux_file in enumerate(aux_ref_audio_files):
                print(f"Auxiliary file {idx}: {aux_file.filename}")

        # Create request object
        request = TTS_Request(
            text=text,
            text_lang=text_lang,
            ref_audio_file=ref_audio_file,
            ref_audio_path=ref_audio_path,
            aux_ref_audio_files=aux_ref_audio_files,
            aux_ref_audio_paths=aux_ref_audio_paths,
            prompt_lang=prompt_lang,
            prompt_text=prompt_text,
            top_k=top_k,
            min_p=min_p,
            top_p=top_p,
            temperature=temperature,
            text_split_method=text_split_method,
            batch_size=batch_size,
            batch_threshold=batch_threshold,
            split_bucket=split_bucket,
            speed_factor=speed_factor,
            fragment_interval=fragment_interval,
            seed=seed,
            media_type=media_type,
            streaming_mode=streaming_mode,
            parallel_infer=parallel_infer,
            repetition_penalty=repetition_penalty
        )

        # Process audio files first
        main_ref_path, aux_paths = await request.process_audio_files()
        
        # Run ASR if needed
        await request.process_asr(main_ref_path)

        # Auto-adjust text_split_method if not provided
        if not request.text_split_method:
            if text_lang.lower() == 'en':
                request.text_split_method = 'cut6'
            elif text_lang.lower() in ['ja', 'all_ja']:
                request.text_split_method = 'cut7'
            else:
                request.text_split_method = 'cut5'

        # Prepare final request dict and validate
        final_req = request.dict(exclude={'ref_audio_file', 'aux_ref_audio_files'})
        final_req['ref_audio_path'] = main_ref_path
        final_req['aux_ref_audio_paths'] = aux_paths

        check_result = check_params(final_req)
        if check_result is not None:
            return check_result

        print(f"Final request configuration: {final_req}")
        return await tts_handle(final_req)
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"Internal server error: {str(e)}"}
        )

class AudioCheckRequest(BaseModel):
    checks: Optional[List[str]] = None  # Optional, if None all checks will be performed
    params: Optional[AudioQualityParams] = None  # Optional, will use defaults if not provided

@APP.post("/checkaudio")
async def check_audio(
    file: UploadFile = File(...),
    config: Optional[AudioCheckRequest] = Form(None)
):
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    try:
        # Check file size from headers (if available)
        file_size = file.size  # FastAPI provides this from Content-Length header
        if file_size and file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Check file type
        content_type = file.content_type
        if not content_type or not content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")

        # Read in chunks and process
        CHUNK_SIZE = 1024 * 1024  # 1MB chunks
        audio_stream = io.BytesIO()
        first_chunk = True

        while chunk := await file.read(CHUNK_SIZE):
            if first_chunk and not chunk:  # Check if first chunk is empty
                raise HTTPException(status_code=400, detail="Empty audio file")
            first_chunk = False
            audio_stream.write(chunk)

        audio_stream.seek(0)  # Reset stream position to beginning

        # Initialize checker
        checker = AudioQualityChecker(
            params=config.params if config and config.params else None
        )
        
        # Process audio and get results
        passed, metrics, analysis = checker.process_audio(audio_stream)

        # Return results directly since they're already converted to Python types
        return JSONResponse({
            "passed": bool(passed),
            "metrics": metrics,
            "analysis": analysis
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if not weights_path:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change gpt weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})

@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if not weights_path:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})

if __name__ == "__main__":
    try:
        if host == 'None':
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0) 
