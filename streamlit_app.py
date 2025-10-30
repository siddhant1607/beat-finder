import streamlit as st
import soundfile as sf
import io
import numpy as np
import librosa
from collections import defaultdict
import pickle
import re
import os
import requests

from fingerprinting import process_segment, extract_peaks_bandwise, generate_pair_hashes
from spotify_util import get_spotify_tracks  # to get album art & spotify link

LOCAL_DB_PATH = "fingerprint_db.pkl"  # in current directory
GOOGLE_DRIVE_FILE_ID = "1Nn4VWd97KENZRggSIEvZJhB2AoWDnRXe"
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)
    st.write("Downloaded fingerprint database from Google Drive.")

def download_fingerprint_db():
    directory = os.path.dirname(LOCAL_DB_PATH)
    if not os.path.exists(directory):
        os.makedirs(directory)
    download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, LOCAL_DB_PATH)

def load_fingerprint_db():
    if not os.path.exists(LOCAL_DB_PATH):
        st.write("Fingerprint DB not present locally, downloading...")
        download_fingerprint_db()
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
                if track:
                    st.markdown(f"### {i}. {track['title']} - {track['artist']}")
                    st.write(f"Matching hashes: {count}")
                    st.write(f"Match confidence: {percent:.1f}%")
                    if 'album' in track and 'images' in track['album']:
                        img_url = track['album']['images'][0]['url']
                        st.image(img_url, width=150)
                    if 'external_urls' in track and 'spotify' in track['external_urls']:
                        st.markdown("Listen on Spotify")
                else:
                    st.write(f"{i}. {song_id} - {count} matching hashes (track info not found)")
                    st.write(f"Match confidence: {percent:.1f}%")
        else:
            st.write("No matches found above threshold.")
