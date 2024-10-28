import os
import sys
import traceback
from typing import Dict, Generator, Optional, Union

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
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
    ref_audio_path: str = None
    ref_audio_file: UploadFile = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
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

    async def process_audio_file(self) -> str:
        if self.ref_audio_file:
            if self.ref_audio_path:
                raise ValueError("Cannot provide both ref_audio_file and ref_audio_path")
            file_path = await save_uploaded_file(self.ref_audio_file)
            return str(file_path)
        return self.ref_audio_path

class AudioProcessor:
    @staticmethod
    def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
        with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
            audio_file.write(data)
        return io_buffer

    @staticmethod
    def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
        io_buffer.write(data.tobytes())
        return io_buffer

    @staticmethod
    def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
        io_buffer = BytesIO()
        sf.write(io_buffer, data, rate, format='wav')
        return io_buffer

    @staticmethod
    def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int) -> BytesIO:
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
        out, _ = process.communicate(input=data.tobytes())
        io_buffer.write(out)
        return io_buffer

    @staticmethod
    def pack_audio(data: np.ndarray, rate: int, media_type: str) -> BytesIO:
        io_buffer = BytesIO()
        packers = {
            "ogg": AudioProcessor.pack_ogg,
            "aac": AudioProcessor.pack_aac,
            "wav": AudioProcessor.pack_wav,
            "raw": AudioProcessor.pack_raw
        }
        packer = packers.get(media_type, AudioProcessor.pack_raw)
        io_buffer = packer(io_buffer, data, rate)
        io_buffer.seek(0)
        return io_buffer

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

def handle_control(command: str) -> None:
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)

def check_params(req: Dict) -> Optional[JSONResponse]:
    text = req.get("text", "")
    text_lang = req.get("text_lang", "")
    ref_audio_path = req.get("ref_audio_path", "")
    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")
    prompt_lang = req.get("prompt_lang", "")
    text_split_method = req.get("text_split_method", "cut5")

    if not ref_audio_path:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if not text:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if not text_lang:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(status_code=400, content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"})
    if not prompt_lang:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(status_code=400, content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"})
    if media_type not in SUPPORTED_MEDIA_TYPES:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    elif media_type == "ogg" and not streaming_mode:
        return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})
    
    if text_split_method not in cut_method_names:
        return JSONResponse(status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"})

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

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode or return_fragment:
        req["return_fragment"] = True
    
    try:
        tts_generator = tts_pipeline.run(req)
        
        if streaming_mode:
            def streaming_generator(tts_generator: Generator, media_type: str):
                if media_type == "wav":
                    yield wave_header_chunk()
                    media_type = "raw"
                for sr, chunk in tts_generator:
                    yield AudioProcessor.pack_audio(chunk, sr, media_type).getvalue()
            
            return StreamingResponse(
                streaming_generator(tts_generator, media_type),
                media_type=f"audio/{media_type}"
            )
        else:
            sr, audio_data = next(tts_generator)
            audio_data = AudioProcessor.pack_audio(audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})

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
    ref_audio_file: UploadFile = File(...),
    prompt_lang: str = Form(...),
    prompt_text: str = Form(""),
    top_k: int = Form(5),
    top_p: float = Form(1.0),
    temperature: float = Form(1.0),
    text_split_method: str = Form("cut5"),
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
        request = TTS_Request(
            text=text,
            text_lang=text_lang,
            ref_audio_file=ref_audio_file,
            prompt_lang=prompt_lang,
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
            media_type=media_type,
            streaming_mode=streaming_mode,
            parallel_infer=parallel_infer,
            repetition_penalty=repetition_penalty
        )

        ref_audio_path = await request.process_audio_file()
        if not ref_audio_path:
            return JSONResponse(
                status_code=400,
                content={"message": "Must provide either ref_audio_file or ref_audio_path"}
            )

        req = request.dict(exclude={'ref_audio_file'})
        req['ref_audio_path'] = ref_audio_path

        return await tts_handle(req)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Internal server error: {str(e)}"}
        )

@APP.get("/set_refer_audio")
async def set_refer_audio(refer_audio_path: str = None):
    try:
        tts_pipeline.set_ref_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})

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
