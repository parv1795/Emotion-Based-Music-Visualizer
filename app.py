import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import yt_dlp
from deepface import DeepFace
from PIL import Image
import pygame
import requests
from io import BytesIO

# Initialize pygame for music playback
pygame.mixer.init()

# Define emotions and their corresponding moods
emotion_map = {
    "angry": "calm",
    "disgust": "relaxing",
    "fear": "soothing",
    "happy": "happy",
    "neutral": "peaceful",
    "sad": "uplifting",
    "surprise": "exciting"
}

def detect_emotion(frame):
    """Analyze the facial expression in the given frame and return the detected emotion."""
    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        emotion = result[0]["dominant_emotion"]
        return emotion
    except Exception as e:
        st.error(f"⚠️ Error detecting emotion: {str(e)}")
        return "neutral"  # Default to neutral if detection fails

def get_youtube_url(emotion):
    """Search YouTube for a song matching the detected emotion and return the first result's URL."""
    search_query = f"{emotion_map.get(emotion, 'neutral')} mood music"

    ydl_opts = {
        'default_search': 'ytsearch1',
        'quiet': True,
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=False)
            if "entries" in info and len(info["entries"]) > 0:
                return info["entries"][0]["webpage_url"]
            else:
                return None  # No results found
    except Exception as e:
        st.error(f"❌ Error searching YouTube: {str(e)}")
        return None

def download_audio(youtube_url):
    """Download the audio from a YouTube video URL using yt-dlp."""
    temp_dir = tempfile.gettempdir()
    output_file = os.path.join(temp_dir, "audio.mp3")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_file,
        'quiet': True,
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return output_file
    except Exception as e:
        st.error(f"❌ Failed to download music: {str(e)}")
        return None

def play_music(youtube_url):
    """Play the downloaded music using pygame."""
    audio_file = download_audio(youtube_url)
    
    if audio_file:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        return audio_file
    else:
        return None

# Streamlit UI
st.title("🎶 Emotion-Based Music Visualizer")
st.write("Capture your emotions and listen to music that matches your mood! 😊")

# Webcam capture
cap = cv2.VideoCapture(0)

if st.button("📷 Capture & Analyze Emotion"):
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        emotion = detect_emotion(frame)

        # Display the captured image
        st.image(frame, caption=f"Detected Emotion: {emotion}", use_container_width=True)

        # Fetch music
        st.write("🔎 Searching for a song...")
        youtube_url = get_youtube_url(emotion)

        if youtube_url:
            st.success(f"🎵 Now Playing: {emotion_map.get(emotion, 'neutral')} Mood Music")
            st.video(youtube_url)  # Embed YouTube video in Streamlit
        else:
            st.warning("❌ No suitable music found.")
    else:
        st.error("❌ Failed to capture image.")

cap.release()
