import io
import os
import shutil
import tempfile
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from transformers import AutoModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set the path to your downloaded IndicF5 directory
MODEL_DIR = "D:\IndicF5-main"  # Replace with your actual local folder path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading local IndicF5 model from '{MODEL_DIR}' on {device}...")

try:
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True).to(device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model from {MODEL_DIR}: {e}")
    model = None

@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    ref_text: str = Form(...),
    ref_audio: UploadFile = File(...)
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    if not text.strip() or not ref_text.strip():
        raise HTTPException(status_code=400, detail="Text and Reference Text cannot be empty.")

    # Save incoming reference audio to a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, ref_audio.filename)

    try:
        with open(temp_audio_path, "wb") as f:
            shutil.copyfileobj(ref_audio.file, f)

        # Run IndicF5 inference
        audio = model(
            text,
            ref_audio_path=temp_audio_path,
            ref_text=ref_text
        )

        # Normalize audio output to 32-bit float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # Encode to WAV buffer (default IndicF5 sampling rate: 24000 Hz)
        buffer = io.BytesIO()
        sf.write(buffer, np.array(audio, dtype=np.float32), samplerate=24000, format="WAV")
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
