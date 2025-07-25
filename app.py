import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import gradio as gr
import requests
from io import BytesIO
import random
import base64
import os
from datetime import datetime


model_id = "jlassi/vit-Facial-Expression-Recognition"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)
labels = model.config.id2label


emoji_map = {
    "happy": "😊", "sad": "😢", "angry": "😠", "surprised": "😲",
    "neutral": "😐", "disgust": "🤢", "fearful": "😨", "calm": "😌",
    "romantic": "😍"
}


SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")

def get_spotify_token():
    token_url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    }
    data = {"grant_type": "client_credentials"}
    response = requests.post(token_url, headers=headers, data=data)
    return response.json().get("access_token", None)


EXCLUDED_KEYWORDS = ["birthday", "happy birthday", "kids", "nursery"]

mood_scent_map = {
    "happy": ("Citrus Burst", "Energizing and joyful with a zesty twist"),
    "sad": ("Rainy Lavender", "Calming floral aroma with notes of melancholy"),
    "angry": ("Smoked Oud", "Deep woody tones to help you unwind"),
    "calm": ("Mint Breeze", "Refreshing and soothing like a gentle stream"),
    "romantic": ("Rose Ember", "Sweet romantic floral with warm undertones"),
    "fearful": ("Pine Noir", "Grounding scent to ease tension"),
    "disgust": ("Ocean Escape", "Clean and purifying marine blend"),
    "surprised": ("Sparkle Spice", "Bright notes of cinnamon and surprise"),
    "neutral": ("Soft Linen", "Simple, clean and balanced")
}


mood_palette = {
    "happy": ["#FFF176", "#FFD54F", "#FFB300"],
    "sad": ["#90A4AE", "#78909C", "#546E7A"],
    "angry": ["#EF5350", "#E53935", "#B71C1C"],
    "calm": ["#81D4FA", "#4FC3F7", "#29B6F6"],
    "romantic": ["#F48FB1", "#EC407A", "#AD1457"],
    "fearful": ["#A1887F", "#8D6E63", "#6D4C41"],
    "disgust": ["#A5D6A7", "#81C784", "#66BB6A"],
    "surprised": ["#FFD54F", "#FFCA28", "#FFA000"],
    "neutral": ["#E0E0E0", "#BDBDBD", "#9E9E9E"]
}


mood_affirmations = {
    "happy": "Keep spreading that sunshine! ☀️", 
    "sad": "You're strong, even when it feels heavy. 💙",
    "angry": "Pause. Breathe. You’ve got control. 🔥",
    "calm": "Stay centered, you radiate peace. 🌿",
    "romantic": "Love is in the air — and in your smile. 💖",
    "fearful": "Even fear is a sign of care. You've got this. 🌌",
    "disgust": "Let go of what doesn't serve you. 🌊",
    "surprised": "Embrace the unexpected — magic lives there! ✨",
    "neutral": "Balance is beautiful. 🧘‍♀️"
}

def search_spotify_track(emotion, language):
    token = get_spotify_token()
    if not token:
        return {"title": "No token", "artist": "Check credentials", "link": "https://spotify.com", "note": ""}

    def get_valid_track(query, note_msg=""):
        try:
            search_url = f"https://api.spotify.com/v1/search?q={query}&type=playlist&limit=10"
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(search_url, headers=headers, timeout=5)
    
            if response.status_code != 200:
                print("Spotify search failed:", response.status_code, response.text)
                return None
    
            playlists = response.json().get("playlists", {}).get("items", [])
            if not playlists:
                return None
    
            playlist = random.choice(playlists)
            playlist_id = playlist.get("id")
            if not playlist_id:
                return None
    
            tracks_url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
            tracks_resp = requests.get(tracks_url, headers=headers, timeout=5)
    
            if tracks_resp.status_code != 200:
                print("Spotify track fetch failed:", tracks_resp.status_code, tracks_resp.text)
                return None
    
            tracks_data = tracks_resp.json()
            tracks = tracks_data.get("items", [])
    
            valid_tracks = [
                t["track"] for t in tracks
                if t.get("track") and t["track"].get("name") and
                not any(kw in t["track"]["name"].lower() for kw in EXCLUDED_KEYWORDS)
            ]
    
            if not valid_tracks:
                return None
    
            selected = random.choice(valid_tracks)
            return {
                "title": selected["name"],
                "artist": selected["artists"][0]["name"],
                "link": selected["external_urls"]["spotify"],
                "note": note_msg
            }
    
        except Exception as e:
            print(f"[ERROR] Exception in get_valid_track: {e}")
            return None



    # First attempt with emotion + language
    track = get_valid_track(f"{emotion} {language}")

    # Fallback with only language
    if not track:
        track = get_valid_track(language, note_msg="🎵 Couldn't find a perfect match for your mood. Here's a cool track in your preferred language!")

    # Final fallback
    if not track:
        return {"title": "No match found", "artist": "Try different mood/language", "link": "https://spotify.com", "note": ""}

    return track



def predict_emotion(image, language):
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.nn.functional.softmax(logits, dim=-1)
    top5 = probs.topk(5)

    top_idx = top5.indices[0].item()
    top_label = labels[top_idx]
    top_conf = round(float(top5.values[0].item()) * 100, 2)
    top_emoji = emoji_map.get(top_label, "")

    scent_name, scent_desc = mood_scent_map.get(top_label, ("Classic Air", "A neutral blend."))
    track = search_spotify_track(top_label, language)
    current_date = datetime.now().strftime("%d %b %Y")
    affirmation = mood_affirmations.get(top_label, "You're doing great. 💛")
    palette = mood_palette.get(top_label, ["#DDD", "#AAA", "#777"])

    html = f"""
    <div style='font-family: Fredoka; padding: 1rem;'>
        <div style='text-align:center; padding: 12px; border-radius: 10px; background-color: #f8f0ff; margin-bottom: 20px; box-shadow: 0 0 12px rgba(106,13,173,0.3);'>
            <h2 style='margin-bottom: 10px;'>🎯 Final Mood Detected:</h2>
            <div style='font-size: 30px; font-weight: bold; color: #6a0dad;'>{top_label.capitalize()} {top_emoji} ({top_conf}%)</div>
        </div>
    """

    for i, p in zip(top5.indices[1:], top5.values[1:]):
        label = labels[i.item()]
        emoji = emoji_map.get(label, "")
        percent = round(float(p.item()) * 100, 2)
        html += f"""
            <div style='margin-bottom: 10px;'>
                <strong style='font-size: 18px;'>{emoji} {label.capitalize()}</strong>
                <div style='background: #eee; border-radius: 10px; overflow: hidden; height: 20px; width: 100%;'>
                    <div style='background: #6a0dad; height: 100%; width: {percent}%; color: white; text-align: right; padding-right: 8px; border-radius: 10px 0 0 10px;'>{percent}%</div>
                </div>
            </div>
        """

    html += f"""
        <div style='margin-top:30px; padding:1rem; background:#fdf6ff; border-radius:10px;'>
            <h3 style='color:#6a0dad;'>🎵 Mood-based Spotify Track:</h3>
            <p><strong>{track['title']}</strong> by {track['artist']}</p>
            {"<p style='font-size:13px; color:#777; margin-top:6px;'>" + track.get("note", "") + "</p>" if track.get("note") else ""}
            <a href='{track['link']}' target='_blank' style='display:inline-block; margin-top:10px; padding:10px 20px; background:#1db954; color:white; border-radius:8px;'>🎧 Listen on Spotify</a>
        </div>
        <div style='margin-top:30px; padding:1rem; background:#fff2fc; border-radius:10px;'>
            <h3 style='color:#c71585;'>🌸 Suggested Mood Scent:</h3>
            <p><strong>{scent_name}</strong> - {scent_desc}</p>
        </div>
        <div style='margin-top:30px; padding:1rem; background:#e3f9ff; border-radius:10px;'>
            <h3 style='color:#039be5;'>🎨 Mood Color Palette:</h3>
            <div style='display:flex; gap:10px;'>
                {''.join([f"<div style='width:60px;height:30px;border-radius:5px;background:{c}' title='{c}'></div>" for c in palette])}
            </div>
        </div>
        <div style='margin-top:30px; padding:1.5rem; background:linear-gradient(145deg, #fdf6ff, #fceaff); border-radius:16px; box-shadow:0 4px 20px rgba(106,13,173,0.15); font-family: Fredoka;'>
            <h2 style='text-align:center; color:#6a0dad;'>🧾 Mood Receipt</h2>
            <hr style='border:none; height:2px; background:#d1c4e9; margin:10px 0 20px 0;' />
            <div style='font-size:16px; line-height:1.8;'>
                <p><strong>📅 Date:</strong> {current_date}</p>
                <p><strong>😀 Emotion:</strong> <span style='color:#6a0dad;'>{top_label.capitalize()} {top_emoji}</span></p>
                <p><strong>🎶 Track:</strong> <em>{track['title']}</em> by <em>{track['artist']}</em></p>
                <p><strong>🌸 Scent:</strong> <em>{scent_name}</em> - {scent_desc}</p>
                <p><strong>💬 Affirmation:</strong> <q>{affirmation}</q></p>
            </div>
        </div>
    </div>
    """
    return html

# Sample & UI

def sample_image():
    url = "https://media.istockphoto.com/id/640307210/photo/choose-positivity-every-morning.jpg?s=612x612&w=0&k=20&c=lCu1J4s_3tFPAjTzMI8koWcyiAGZvkYxnaiR8GJWDco="
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def clear_fields():
    return None, ""

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500&display=swap');
h1, .custom-desc {
    text-align: center;
    font-family: 'Fredoka', sans-serif !important;
    color: #4b0082;
}
h4, .custom-desc {
    text-align: center;
    font-family: 'Fredoka', sans-serif !important;
    color: #c71585;
}
.custom-desc {
    font-size: 12px;
    color: #444;
    max-width: 900px;
    margin: 0 auto;
    padding-bottom: 1.5rem;
}
#language-input {
    font-weight: bold;
    color: #6a0dad;
    font-size: 16px;
    margin-top: 1rem;
    border: 2px solid #6a0dad;
    border-radius: 10px;
    padding: 10px;
    font-family: 'Fredoka', sans-serif;
    background: #f4edfa;
    color: #333;
    transition: all 0.3s ease;
}
#language-input:focus {
    border-color: #9b30ff;
    box-shadow: 0 0 10px rgba(106, 13, 173, 0.3);
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("""
    <h1>🪄 Feelora 🔮 </h1>
    <h4>Where your mood meets music, scent, and soul.💫<h4>
    <p class='custom-desc'>
        Feelora 😊✨ reads your selfie to discover how you’re feeling right now — happy, calm, or maybe surprised! 🎭🎶 It then picks the perfect Spotify song 🎧, suggests a soothing virtual scent 🌸, and shows colors that match your mood 🎨. All this comes together in a stylish mood receipt 🧾, made just for you to brighten your day! 🌟
    </p>
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="📸 Upload Your Selfie")
            language_input = gr.Dropdown(
                choices=[
                    "english", "hindi", "tamil", "telugu", "kannada", "malayalam",
                    "bengali", "spanish", "french", "german", "japanese", "korean", "arabic", "portuguese", "russian"
                ],
                value="english",
                label="🎧 Preferred Language for Music",
                interactive=True,
                elem_id="language-input"
            )
            with gr.Row():
                sample_btn = gr.Button("🎁 Try Sample")
                clear_btn = gr.Button("🧼 Clear All")
                predict_btn = gr.Button("🔍 Analyze Mood")
        with gr.Column():
            output_html = gr.HTML(label="🧠 Vibe Breakdown")

    predict_btn.click(fn=predict_emotion, inputs=[image_input, language_input], outputs=output_html)
    sample_btn.click(fn=sample_image, outputs=image_input)
    clear_btn.click(fn=clear_fields, outputs=[image_input, output_html])

demo.launch(share=True)
