import streamlit as st
import soundfile as sf
import io
import numpy as np
import librosa
from collections import defaultdict
import pickle
import re

from fingerprinting import process_segment, extract_peaks_bandwise, generate_pair_hashes
from spotify_util import get_spotify_tracks  # to get album art & spotify link

# Constants and config - replace paths as needed
FINGERPRINT_DB_PATH = "/workspaces/beat-finder/fingerprint_db.pkl"

# Load fingerprint db once (cache for session)
@st.cache(allow_output_mutation=True)
def load_fingerprint_db():
    with open(FINGERPRINT_DB_PATH, "rb") as f:
        return pickle.load(f)

fingerprint_db = load_fingerprint_db()

st.title("🎤 BeatFinder Demo")
st.write("Record audio and try to recognize it from the song database!")

# Audio input
audio_bytes = st.audio_input("Record a 10s audio sample (or upload WAV)", sample_rate=16000)

def get_song_info(song_id, client_id, client_secret, cache={}):
    # Cache API results for performance
    if song_id in cache:
        return cache[song_id]

    # Reverse song_id back to title and artist if possible (based on your sanitize logic)
    # Here we assume a function to get playlist tracks info is available from spotify_util
    playlist_url = "YOUR_SPOTIFY_PLAYLIST_URL"  # put your playlist url here
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
    # Decode uploaded audio bytes
    y, sr = sf.read(io.BytesIO(audio_bytes))

    # Fingerprint
    S_mag = process_segment(y, sr)
    peaks = extract_peaks_bandwise(S_mag)
    pair_hashes = generate_pair_hashes(peaks)
    st.write(f"Fingerprinted {len(pair_hashes)} pairs.")

    if st.button("Find best match"):
        # Matching
        score = defaultdict(int)
        offset_diffs = defaultdict(lambda: defaultdict(int))  # for time offset filtering
        for h, t in pair_hashes:
            entries = fingerprint_db.get(h, [])
            for song, song_time in entries:
                offset_diff = round(song_time - t, 2)
                offset_diffs[song][offset_diff] += 1

        best_matches = {}
        for song, offsets in offset_diffs.items():
            best_offset, best_count = max(offsets.items(), key=lambda x: x[1])
            best_matches[song] = best_count

        # Filter results by threshold
        threshold = 10
        filtered = {k: v for k, v in best_matches.items() if v >= threshold}

        if filtered:
            top3 = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:3]
            st.write("Top matches:")
            for i, (song_id, count) in enumerate(top3, 1):
                track = get_song_info(song_id, st.secrets["CLIENT_ID"], st.secrets["CLIENT_SECRET"])
                if track:
                    st.markdown(f"### {i}. {track['title']} - {track['artist']}")
                    st.write(f"Matching hashes: {count}")
                    if 'album' in track and 'images' in track['album']:
                        img_url = track['album']['images'][0]['url']
                        st.image(img_url, width=150)
                    if 'external_urls' in track and 'spotify' in track['external_urls']:
                        st.markdown(f"[Listen on Spotify]({track['external_urls']['spotify']})")
                else:
                    st.write(f"{i}. {song_id} - {count} matching hashes (track info not found)")
        else:
            st.write("No matches found above threshold.")

