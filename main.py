from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
from moviepy.video.VideoClip import ColorClip
from moviepy.video.fx import all as vfx
import librosa
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import tensorflow_hub as hub
import tensorflow as tf
import scipy.signal.windows
import requests
from moviepy.video.compositing.concatenate import concatenate_videoclips
from multiprocessing import Pool
import pandas as pd
from video_prompt_generator import generate_video_prompt
import math
import os
from sklearn.cluster import KMeans
from tempfile import NamedTemporaryFile
import logging
import uuid
import traceback
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from functools import lru_cache
import librosa
from Emotion import Extract_emotion
import requests
import json
import re
# Configure logging
logging.basicConfig(level=logging.ERROR)

app = FastAPI()

#CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, should be configured in production.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods.
    allow_headers=["*"],  # Allows all headers.
)

# Ensure necessary directories exist
if not os.path.exists("temp_audio"):
    os.makedirs("temp_audio")
if not os.path.exists("temp_video"):
    os.makedirs("temp_video")

# Helper Functions (same as before)

@lru_cache(maxsize=1)  # Load model once in cache
def load_whisper_model(model_id="openai/whisper-tiny", device="cpu", torch_dtype=torch.float32):
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def Extract_lyrics(audio_path, model_id="openai/whisper-tiny", language=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load model once
    model, processor = load_whisper_model(model_id, device, torch_dtype)
    
    # Load audio from the path
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None
    
    # Prepare input
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    
    # Dynamically access correct input key
    if "input_values" in inputs:
        input_ids = inputs["input_values"].to(device, torch_dtype)
    elif "input_features" in inputs:
        input_ids = inputs["input_features"].to(device, torch_dtype)
    else:
        print(f"Error: Neither 'input_values' nor 'input_features' found in processor output.")
        return None
    
    # Generate transcription
    if language:
        lang_ids = list(processor.tokenizer.get_vocab().values()) # gets list of all ids
        if language in processor.tokenizer.lang_code_to_id: # check to see if the language is supported
            lang_id = processor.tokenizer.lang_code_to_id[language]
            if device == "cuda:0":
                generated_ids = model.generate(
                    inputs=input_ids,
                    max_new_tokens=400, # Reduced max_new_tokens
                    num_beams=1,
                     forced_bos_token_id=lang_id, # Use language ID only if available
                    do_sample=False, 
                )
            else:
                generated_ids = model.generate(
                    inputs=input_ids,
                    max_new_tokens=400, # Reduced max_new_tokens
                    num_beams=1,
                    forced_bos_token_id=lang_id,  # Use language ID only if available
                   do_sample=False,
                )
        else:
            print(f"Warning: Language '{language}' is not supported by the tokenizer. Ignoring language.")
            if device == "cuda:0":
                 generated_ids = model.generate(
                    inputs=input_ids,
                    max_new_tokens=400,  # Reduced max_new_tokens
                    num_beams=1,
                    do_sample=False,
                 )
            else:
                generated_ids = model.generate(
                    inputs=input_ids,
                    max_new_tokens=400, # Reduced max_new_tokens
                    num_beams=1,
                    do_sample=False,
             )
    else:
          if device == "cuda:0":
            generated_ids = model.generate(
                inputs=input_ids,
                max_new_tokens=400,  # Reduced max_new_tokens
                num_beams=1,
                do_sample=False,
             )
          else:
            generated_ids = model.generate(
                inputs=input_ids,
                max_new_tokens=400, # Reduced max_new_tokens
                num_beams=1,
                do_sample=False,
             )

    # Decode
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return transcription

# Step 1: Extract audio properties and user input for customization
def analyze_audio(audio_path):

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        logging.error(f"Error loading audio file: {e}")
        raise

    # Calculate audio properties
    duration_seconds = librosa.get_duration(y=y, sr=sr)
    channels = 1  # Librosa loads audio as mono by default
    frame_rate = sr
    average_amplitude = np.mean(np.abs(y))

    # Calculate approximate loudness over chunks
    chunk_duration = 1.0  # seconds
    chunk_size = int(sr * chunk_duration)
    num_chunks = math.ceil(len(y) / chunk_size)
    chunk_loudness = [
        np.mean(np.abs(y[i * chunk_size: (i + 1) * chunk_size]))
        for i in range(num_chunks)
    ]
    
    # Identify significant loudness peaks
    loud_significant_points = [
        i for i, loudness in enumerate(chunk_loudness) if loudness > 1.5 * average_amplitude
    ]

    # Estimate tempo
    scipy.signal.hann = scipy.signal.windows.hann
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # Convert beat frames to time (in seconds)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # Convert beat times to significant points (in sample indices)
    significant_points = beat_times.tolist() #beat_times=timestamps


    # Detect rhythmic patterns
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        rhythmic_patterns = len(onset_frames) > 0
    except Exception as e:
        rhythmic_patterns = False
        logging.error(f"Error detecting rhythmic patterns: {e}")

    # Estimate key using chroma features
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_index = np.argmax(np.mean(chroma, axis=1))
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_estimate = keys[key_index]
    except Exception as e:
        key_estimate = "Unknown"
        logging.error(f"Error estimating key: {e}")

    # Determine energy level
    min_loudness = min(chunk_loudness) if chunk_loudness else 0.01  # Avoid division by zero
    dynamic_range = max(chunk_loudness) / min_loudness
    energy_level = "dynamic" if dynamic_range > 2 else "steady"

    # Analyze harmonic content
    try:
        harmonic = librosa.effects.harmonic(y=y)
        harmonic_content = np.mean(np.abs(harmonic)) > 0.1
    except Exception as e:
        harmonic_content = False
        logging.error(f"Error analyzing harmonic content: {e}")
    # Generate the audio text description

    audio_text_description = f"""
    The audio file has a duration of approximately {duration_seconds:.2f} seconds.
    It has {channels} channel(s) with a frame rate of {frame_rate} Hz.
    The approximate tempo is {tempo} BPM, and it is in the key of {key_estimate}.
    The energy levels in the audio are {energy_level}, with {len(loud_significant_points)} significant peaks in loudness.
    Rhythmic patterns {'are' if rhythmic_patterns else 'are not'} detected, and harmonic content {'is' if harmonic_content else 'is not'} prominent.
    """
    return audio_text_description, tempo, frame_rate, significant_points

def cluster_near_beats(significant_points):
    data = np.array(significant_points)
    data = data.reshape(-1, 1)
    kmeans = KMeans(n_clusters=10, random_state=42)
    labels = kmeans.fit_predict(data)
    clustered_beats=list(kmeans.cluster_centers_.flatten())
    return clustered_beats


import moviepy.video.fx.all as vfx
from moviepy.editor import VideoClip, CompositeVideoClip, ColorClip
import numpy as np



def apply_beat_sync_effects(clip, timestamps, beat_duration=0.002):
    """
    Synchronize visual effects with the provided beat timestamps.
    Effects:
    1. Zoom in on the current frame.
    2. Pulse (increase brightness) on the current frame.
    """
    effects = []

    for ts in timestamps:
        if ts <= clip.duration:
            # Apply Zoom Effect (scale from 1.0 to 1.2 over beat duration)
            zoom_clip = clip.subclip(ts, ts + beat_duration).fx(vfx.resize, 1.2)

            # Apply Pulsing Effect (brightness increase from 1.0 to 1.5)
            pulse_clip = clip.subclip(ts, ts + beat_duration).fx(vfx.colorx, 1.5)

            effects.append(zoom_clip)
            effects.append(pulse_clip)

    return CompositeVideoClip([clip] + effects)


# FastAPI Endpoints

@app.post("/generate_prompt/")
async def generate_prompt(
    audio_file: UploadFile = File(None),
    url: str = Form(None),
    theme: str = Form(...),
    color_pattern: str = Form(...),
    primary_color: Optional[str] = Form(None),
    secondary_color: Optional[str] = Form(None),
    animation_type: str = Form(...)
):
    audio_path = None
    try:
        # Handle URL download or file upload
        if url:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)

            # Extract the .mp3 link using regex
            match = re.search(r'"audio_file":"(https://[\w./%-]+\.mp3)"', response.text)

            if match:
                mp3_url = match.group(1)
                print("MP3 URL:", mp3_url)

                # Download the MP3 file
                mp3_response = requests.get(mp3_url, headers=headers)
                
                audio_temp = NamedTemporaryFile(delete=False, suffix=".mp3", dir="temp_audio")
                audio_temp.write(mp3_response.content)
                audio_temp.close()
                
                audio_path = audio_temp.name
                print(f"Downloaded: {audio_path}")
            else:
                print("MP3 URL not found!")
        elif audio_file:
            audio_temp = NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1], dir="temp_audio")
            audio_temp.write(await audio_file.read())
            audio_temp.close()
            audio_path = audio_temp.name
            print("Audio: ",type(audio_temp))
            print("Audio_path_type: ",type(audio_path))
            print("Audio_path: ",audio_path)
        else:
            raise HTTPException(status_code=400, detail="No audio file or URL provided.")
         
        # 2. Analyze the audio
        audio_description, tempo, sr, significant_points = analyze_audio(audio_path)
        print('Audio Details: ', audio_description)
        print("Time Stamps: ",significant_points)

        # 3. Determine Colors
        colors = []
        if color_pattern == "Custom":
            if primary_color and secondary_color:
                colors = [primary_color, secondary_color]
            else:
                raise HTTPException(status_code=400, detail="Primary and secondary colors are required for custom color pattern.")
        else:
            palette = {
                "Vibrant": ["#FF5733", "#FFC300", "#DAF7A6"],
                "Pastel": ["#F7CAC9", "#92A8D1", "#F4E1D2"],
                "Monochrome": ["#2C2C2C", "#4F4F4F", "#BFBFBF"],
            }
            colors = palette.get(color_pattern, [])

        # 4. Extract Lyrics and Emotion
        lyrics = Extract_lyrics(audio_path)
        valence = Extract_emotion(audio_path)
        if 1.0 <= valence < 3.0:
            emotion = "Anger"
        elif 3.0 <= valence < 5.0:
            emotion = "Sad"
        elif 5.0 <= valence < 7.0:
            emotion = "Neutral"
        elif 7.0 <= valence <= 9.0:
            emotion = "Happy"
        else:
            emotion = "Excited"

        
        print("AGENTIC-AI: ")

        # 5. Generate Video Prompt
        input_data = {
            "audio_description": audio_description,
            "valence": valence,
            "emotion": emotion,
            "lyrics": lyrics,
            "theme": theme,
            "colors": colors,
            "animation": animation_type
        }
        output = generate_video_prompt(input_data)
        print("Generated Video Prompt:")
        print(output)
        
        # 6. Return the generated prompt and other data
        return JSONResponse({
            "prompt": str(output),
            "audio_path": audio_path,
            "tempo": tempo,
            "sr": sr,
            "significant_points": significant_points
        })

    except Exception as e:
        logging.error(f"Error in generate prompt endpoint: {e}") #Log exception
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_video/")
async def process_video(
    video_file: UploadFile = File(...),
    audio_path: str = Form(...), # Change audio file to audio_path and make it string
    tempo: float = Form(...),
    sr: int = Form(...),
    significant_points_str: str = Form(...)
):
    output_file = None
    video_clip = None
    audio_clip = None
    #audio_path = audio_file.filename
    video_path = video_file.filename

    print("Audio Path: ",audio_path)
    print("Completed..........")
  
    # Save uploaded files
    with open(video_path, "wb") as f:
        f.write(await video_file.read())  # Use 'await' for async reading


    print("File Has Been Written...")
    video_clip = VideoFileClip(video_path).without_audio()

    audio_duration = AudioFileClip(audio_path).duration
    video_duration = video_clip.duration
    # Determine the shorter duration
    min_duration = min(audio_duration, video_duration)

    # Trim both the video and audio to the shorter duration
    video_clip = video_clip.subclip(0, min_duration)
    audio_clip = AudioFileClip(audio_path).subclip(0, min_duration)

    # Merge the trimmed audio with the video
    final_video = video_clip.set_audio(audio_clip)
    print("Music and Video Merged Successfully")

    # Parse significant points
    if significant_points_str == "[]":
        significant_points_str = significant_points_str.replace("[]", "[0]")
    significant_points = list(map(float, significant_points_str.strip("[]").split(",")))
    print("Timestamps: ",significant_points)
    # Downsample significant points for optimization
    if len(significant_points)>10:
        timestamps=cluster_near_beats(significant_points)
    else:
        timestamps=significant_points

    print('Timestamps: ',timestamps)
    print('sr = ',sr)
    print("Started Beat Synching...")

    # Process the entire video in one go
    final_video = apply_beat_sync_effects(final_video, timestamps)


    output_file = "final_video_with_audio.mp4"
    print("Beat Synchronization Complete. Writing output...")

    # Output the final video
    final_video.write_videofile(
    output_file, 
    codec="libx264", 
    audio_codec="aac", 
    preset="ultrafast",  # Fastest encoding preset
    bitrate="5000k",     # Lower bitrate speeds up rendering
    threads=4,           # Use multi-threading for faster processing
    fps=24,              # Set FPS to a reasonable value
    temp_audiofile='temp-audio.m4a',  # Temporary audio file for faster muxing
    remove_temp=True
    )

    return FileResponse(output_file, media_type="video/mp4", filename="final_video.mp4")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)