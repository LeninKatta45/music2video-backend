import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import StandardScaler
import joblib
import time  # Added to measure inference time

# Load VGGish model
def load_vggish_model():
    print("Loading VGGish model...")
    model = hub.load("https://tfhub.dev/google/vggish/1")
    return model

# Extract VGGish embeddings from new audio
def extract_vggish_embeddings(audio_path, model):
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Ensure the audio is 16kHz
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    # Extract embeddings using VGGish
    embeddings = model(audio)
    return embeddings.numpy()

# Predict emotion from new audio
def predict_emotion(audio_path, scaler, regressor):
    # Load VGGish model
    vggish_model = load_vggish_model()
    
    # Extract embeddings
    embeddings = extract_vggish_embeddings(audio_path, vggish_model)
    
    # Average the embeddings over time to get a single 128-dimensional vector
    avg_embeddings = np.mean(embeddings, axis=0)
    
    # Normalize embeddings
    avg_embeddings = scaler.transform([avg_embeddings])
    
    # Predict emotion (valence/arousal)
    emotion = regressor.predict(avg_embeddings)[0]
    return emotion

def Extract_emotion(audio_file):
    # Load the trained scaler and regressor
    scaler = joblib.load("vggish_emotion_scaler.pkl")  # Replace with the correct path
    regressor = joblib.load("vggish_emotion_regressor.pkl")  # Replace with the correct path
    # Predict emotion
    start = time.time()
    emotion = predict_emotion(audio_file, scaler, regressor)
    end = time.time()
   
    print(f"Inference time for Emotion Detection: {end - start:.4f} seconds")
    return emotion
if __name__ == "__main__":
    Extract_emotion("audio.mp3")

