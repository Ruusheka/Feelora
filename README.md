# 🌈 Feelora

![Feelora Banner](https://raw.githubusercontent.com/Ruusheka/Feelora/main/image.png)

> **Feelora** is an AI-powered mood detection app that reads your emotion from a selfie and vibes back with personalized music, scent, color themes, and a downloadable mood card — all in real time.

---

## ✨ Features

- 🧠 **Emotion Detection**: Upload a selfie and Feelora detects your current mood using state-of-the-art AI models.
- 🎵 **Music Vibes**: Get a curated playlist that matches your mood and language (English, Tamil, Hindi).
- 🎨 **Mood Color & Wallpaper**: Your detected mood generates matching background color and optional AI wallpaper.
- 🌸 **Mood Scent (Concept)**: Suggests a scent profile based on your emotional state.
- 🖼️ **Mood Card Generator**: Download a personalized card with your mood, emotion color, and emoji avatar.
- 🪄 **Daily NFT Concept**: Each mood can become a collectible — a unique visual avatar NFT (future feature).

---

## 🚀 Live Demo

🔗 [Try Feelora on Hugging Face Spaces](https://huggingface.co/spaces/Ruusheka/Feelora)

---

## 🛠️ Tech Stack

- `Python` | `Gradio` | `Hugging Face Transformers`
- `DeepFace` – for face-based emotion detection
- `Spotify API` – for real-time playlist generation
- `Stable Diffusion` – for AI-generated wallpapers
- `OpenCV` + `Pillow` – for image manipulation and mood card rendering

---

## 📸 Screenshots

| Mood Detection | Playlist Suggestion | Mood Card |
|----------------|---------------------|-----------|
| ![Detect](https://raw.githubusercontent.com/Ruusheka/Feelora/main/assets/detect.png) | ![Music](https://raw.githubusercontent.com/Ruusheka/Feelora/main/assets/music.png) | ![Card](https://raw.githubusercontent.com/Ruusheka/Feelora/main/assets/card.png) |

---

## 🧪 How to Run Locally

```bash
git clone https://github.com/Ruusheka/Feelora.git
cd Feelora
pip install -r requirements.txt
python app.py
