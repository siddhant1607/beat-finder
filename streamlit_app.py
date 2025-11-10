# streamlit run streamlit_app.py
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

LOCAL_DB_PATH = "fingerprint_db.pkl"  # in current directory

def load_fingerprint_db():
    with open(LOCAL_DB_PATH, "rb") as f:
        return pickle.load(f)

fingerprint_db = load_fingerprint_db()

st.title("🎤 BeatFinder")
st.write("Record audio and try to recognize it from the song database!")

audio_bytes = st.audio_input("Record a 10s audio sample", sample_rate=16000)

def get_song_info(song_id, client_id, client_secret, cache={}):
    if song_id in cache:
        return cache[song_id]
    playlist_url = "https://open.spotify.com/playlist/7AFzTreiVAZpYU8wYW3fp9?si=AvcUBT7ZTfyKhMBMepcmKw"
    tracks = get_spotify_tracks(playlist_url, client_id, client_secret)
    for track in tracks:
        if track['id'] == song_id:
            cache[song_id] = track
            return track
    return None

if audio_bytes is not None:
    y, sr = sf.read(audio_bytes)
    max_samples = sr * 10
    y = y[:max_samples]

    S_mag = process_segment(y, sr)
    peaks = extract_peaks_bandwise(S_mag,sr)
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
            best_offset, best_count = max(offsets.items(), key=lambda x: x[1])
            best_matches[song] = best_count

        # Exclude matches below 500 hashes
        filtered = {k: v for k, v in best_matches.items() if v >= 500}

        if filtered:
            top_matches = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:3]
            st.write("Top matches:")

            match_counts = [count for _, count in top_matches]

            for i, (song_id, count) in enumerate(top_matches, 1):
                if count >= 1000:
                    confidence_label = "High Confidence"
                elif 750 <= count < 1000:
                    confidence_label = "Medium Confidence"
                elif 500 <= count < 750:
                    confidence_label = "Low Confidence"
                else:
                    confidence_label = "Unknown Confidence"

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

                # Always show the detected song_id below the track name
                if track:
                    st.markdown(f"### {i}. {track['title']} - {track['artist']}")
                    st.markdown(f"<i style='font-size:small;'>ID: {song_id}</i>", unsafe_allow_html=True)
                else:
                    st.markdown(f"### {i}. Unknown Track")
                    st.markdown(f"<i style='font-size:small;'>ID: {song_id}</i>", unsafe_allow_html=True)

                st.write(f"Matching hashes: {count}")
                st.write(f"Match Confidence: {confidence_label}")
                st.write(f"Match percentage: {percent:.1f}%")
                # Show album art if present
                if (
                    track
                    and "album" in track
                    and "images" in track["album"]
                    and track["album"]["images"]
                ):
                    img_url = track["album"]["images"][0]["url"]
                    st.image(img_url, width=150)
                # Show Spotify link if present
                if (
                    track
                    and "external_urls" in track
                    and "spotify" in track["external_urls"]
                ):
                    url = track["external_urls"]["spotify"]
                    st.markdown(f"[Listen on Spotify]({url})", unsafe_allow_html=True)

                if i == 1:
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("No matches found above threshold.")
