import streamlit as st
import soundfile as sf
import io
import numpy as np
import librosa
from collections import defaultdict, Counter
import pickle
import re
import os

from fingerprinting import process_segment, extract_peaks_bandwise, generate_pair_hashes
from spotify_util import get_spotify_tracks  # to get album art & spotify link

LOCAL_DB_PATH = "fingerprint_db_hamming.pkl"  # in current directory


@st.cache_resource
def load_fingerprint_db():
    with open(LOCAL_DB_PATH, "rb") as f:
        return pickle.load(f)


fingerprint_db = load_fingerprint_db()

st.title("🎤 BeatFinder")
st.write("Record audio or upload a file to recognize it from the song database!")

# --- Testing controls ---
with st.expander("🧪 Testing Features (for experimentation only)"):
    st.warning("These features are for testing only. For normal use, leave at default values!")
    snr_db = st.slider(
        "Simulate Noise: SNR (dB) - higher = less noise", min_value=-10, max_value=10, value=10
    )
    min_match_thresh = st.number_input(
        "Detection Threshold", min_value=1, max_value=1000, value=500
    )

# --- Input tabs ---
tab1, tab2 = st.tabs(["Record Audio", "Upload File"])

audio_bytes = None

with tab1:
    st.write("Record a 10-second audio sample:")
    recorded_audio = st.audio_input("Record a 10s audio sample", sample_rate=16000, key="recorder")
    if recorded_audio:
        audio_bytes = recorded_audio

with tab2:
    st.write("Upload a WAV or MP3 audio file (max 10 seconds will be processed):")
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"], key="uploader")
    if uploaded_file:
        audio_bytes = uploaded_file.getvalue()


def add_noise_to_signal(signal, snr_db):
    if snr_db >= 40:  # Treat 40dB or more as no noise
        return signal
    rms_signal = np.sqrt(np.mean(signal**2))
    snr_linear = 10 ** (snr_db / 10)
    rms_noise = rms_signal / np.sqrt(snr_linear)
    noise = np.random.normal(0, rms_noise, signal.shape[0])
    noisy_signal = signal + noise
    return noisy_signal


def get_song_info(song_id, client_id, client_secret, cache={}):
    if song_id in cache:
        return cache[song_id]
    playlist_url = "https://open.spotify.com/playlist/7AFzTreiVAZpYU8wYW3fp9?si=AvcUBT7ZTfyKhMBMepcmKw"
    tracks = get_spotify_tracks(playlist_url, client_id, client_secret)
    for track in tracks:
        if track["id"] == song_id:
            cache[song_id] = track
            return track
    return None


if audio_bytes is not None:
    # Properly wrap bytes to BytesIO only when needed
    if isinstance(audio_bytes, (bytes, bytearray)):
        audio_stream = io.BytesIO(audio_bytes)
    else:
        audio_stream = audio_bytes

    y, sr = sf.read(audio_stream)

    # Resample to 16kHz if needed
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    # If stereo, convert to mono by averaging channels
    if y.ndim > 1:
        y = y.mean(axis=1)

    max_samples = sr * 10
    y = y[:max_samples]

    # Apply noise if requested
    y = add_noise_to_signal(y, snr_db)

    S_mag = process_segment(y, sr)
    peaks = extract_peaks_bandwise(S_mag)
    pair_hashes = generate_pair_hashes(peaks)
    st.write(f"Fingerprinted {len(pair_hashes)} pairs.")

    if st.button("Find best match"):
        offset_diffs = defaultdict(Counter)
        hash_cache = {}

        for h, t in pair_hashes:
            if h not in hash_cache:
                hash_cache[h] = fingerprint_db.get(h, [])
            entries = hash_cache[h]
            for song, song_time in entries:
                offset_diff = round(song_time - t, 2)
                offset_diffs[song][offset_diff] += 1

        best_matches = {}
        for song, offsets in offset_diffs.items():
            if not offsets:
                continue
            best_offset, best_count = max(offsets.items(), key=lambda x: x[1])
            best_matches[song] = best_count

        filtered = {k: v for k, v in best_matches.items() if v >= min_match_thresh}

        if filtered:
            top_matches = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:3]
            st.write("Top matches:")

            match_counts = [count for _, count in top_matches]

            multiple_strong_condition = False
            if len(match_counts) > 1:
                if match_counts[0] >= 50 and match_counts[1] >= 50:
                    diff = abs(match_counts[0] - match_counts[1])
                    if diff < 0.2 * match_counts[0]:  # within 20%
                        multiple_strong_condition = True
                        st.warning(
                            "⚠ Multiple strong matches detected. The audio sample may contain mixed sources or be ambiguous."
                        )

            for i, (song_id, count) in enumerate(top_matches, 1):
                if count >= 100:
                    confidence_label = "Very High Confidence"
                elif count >= 50:
                    confidence_label = "High Confidence"
                elif count >= 20:
                    confidence_label = "Medium Confidence"
                else:
                    confidence_label = "Low Confidence"

                percent = (count / sum(match_counts)) * 100 if sum(match_counts) > 0 else 0
                try:
                    client_id = st.secrets["CLIENT_ID"]
                    client_secret = st.secrets["CLIENT_SECRET"]
                except Exception:
                    client_id = os.getenv("CLIENT_ID")
                    client_secret = os.getenv("CLIENT_SECRET")

                track = get_song_info(song_id, client_id, client_secret)

                if i == 1:
                    st.markdown(
                        f"<div style='background-color:#FFD700; padding:10px; border-radius:5px;'>",
                        unsafe_allow_html=True,
                    )

                if track:
                    st.markdown(f"### {i}. {track['title']} - {track['artist']}")
                    st.markdown(f"<i style='font-size:small;'>ID: {song_id}</i>", unsafe_allow_html=True)
                else:
                    st.markdown(f"### {i}. Unknown Track")
                    st.markdown(f"<i style='font-size:small;'>ID: {song_id}</i>", unsafe_allow_html=True)

                st.write(f"Matching hashes: {count}")
                st.write(f"Match Confidence: {confidence_label}")
                st.write(f"Match percentage: {percent:.1f}%")

                if track and "album" in track and "images" in track["album"] and track["album"]["images"]:
                    img_url = track["album"]["images"][0]["url"]
                    st.image(img_url, width=150)

                if track and "external_urls" in track and "spotify" in track["external_urls"]:
                    url = track["external_urls"]["spotify"]
                    st.markdown(f"[Listen on Spotify]({url})", unsafe_allow_html=True)

                if i == 1:
                    st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.write("No matches found above the threshold.")
