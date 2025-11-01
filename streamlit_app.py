import streamlit as st
import soundfile as sf
import io
import numpy as np
import librosa
from collections import defaultdict
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

st.title("🎤 BeatFinder Demo")
st.write("Record audio and try to recognize it from the song database!")

# Audio input
audio_bytes = st.audio_input("Record a 10s audio sample (or upload WAV)", sample_rate=16000)

def get_song_info(song_id, client_id, client_secret, cache={}):
    if song_id in cache:
        return cache[song_id]
    playlist_url = "https://open.spotify.com/playlist/7AFzTreiVAZpYU8wYW3fp9?si=AvcUBT7ZTfyKhMBMepcmKw"
    tracks = get_spotify_tracks(playlist_url, client_id, client_secret)
    for track in tracks:
        title = track.get('title', '')
        artist = track.get('artist', '')
        generated_id = re.sub(r'\W+', '_', title.lower()) + "__" + re.sub(r'\W+', '_', artist.lower())
        if generated_id == song_id:
            cache[song_id] = track
            return track
    return None

if audio_bytes is not None:
    y, sr = sf.read(audio_bytes)

    S_mag = process_segment(y, sr)
    peaks = extract_peaks_bandwise(S_mag)
    pair_hashes = generate_pair_hashes(peaks)
    st.write(f"Fingerprinted {len(pair_hashes)} pairs.")

    if st.button("Find best match"):
        score = defaultdict(int)
        offset_diffs = defaultdict(lambda: defaultdict(int))
        for h, t in pair_hashes:
            entries = fingerprint_db.get(h, [])
            for song, song_time in entries:
                offset_diff = round(song_time - t, 2)
                offset_diffs[song][offset_diff] += 1

        best_matches = {}
        for song, offsets in offset_diffs.items():
            best_offset, best_count = max(offsets.items(), key=lambda x: x[1])
            best_matches[song] = best_count

        threshold = 10
        filtered = {k: v for k, v in best_matches.items() if v >= threshold}

        if filtered:
            top5 = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:5]
            total_matches = sum(count for _, count in top5)
            st.write("Top matches:")
            for i, (song_id, count) in enumerate(top5, 1):
                percent = (count / total_matches) * 100 if total_matches > 0 else 0
                track = get_song_info(song_id, st.secrets["CLIENT_ID"], st.secrets["CLIENT_SECRET"])

                # Highlight top song
                if i == 1:
                    st.markdown(f"<div style='background-color:#FFD700; padding:10px; border-radius:5px;'>", unsafe_allow_html=True)
                    st.markdown(f"### **{i}. {track['title']} - {track['artist']}**")
                else:
                    if track:
                        st.markdown(f"### {i}. {track['title']} - {track['artist']}")

                st.write(f"Matching hashes: {count}")
                st.write(f"Match confidence: {percent:.1f}%")
                if track and 'album' in track and 'images' in track['album']:
                    img_url = track['album']['images'][0]['url']
                    st.image(img_url, width=150)
                if track and 'external_urls' in track and 'spotify' in track['external_urls']:
                    st.markdown("Listen on Spotify")

                if i == 1:
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("No matches found above threshold.")
