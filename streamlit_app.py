import streamlit as st
import soundfile as sf
import io
import numpy as np
from collections import defaultdict
import pickle
import re
import os

import fingerprinting
from spotify_util import get_spotify_tracks

LOCAL_DB_PATH = "fingerprint_db.pkl"


def load_fingerprint_db():
    with open(LOCAL_DB_PATH, "rb") as f:
        return pickle.load(f)


fingerprint_db = load_fingerprint_db()

st.title("🎤 BeatFinder")
st.write("Record audio and try to recognize it from the song database!")

audio_bytes = st.audio_input("Record a 10s audio sample", sample_rate=44100)


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
    # Save audio bytes to temporary file for fingerprinting
    temp_audio_path = "temp_query.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_bytes.getbuffer())
    
    hashes = fingerprinting.create_audio_fingerprint(temp_audio_path)
    os.remove(temp_audio_path)  # Clean up
    
    st.write(f"Fingerprinted {len(hashes)} hashes.")

    if st.button("Find best match"):
        offset_diffs = defaultdict(lambda: defaultdict(int))
        
        for h, t in hashes:
            entries = fingerprint_db.get(h, [])
            for song, song_time in entries:
                offset_diff = round(song_time - t, 2)
                offset_diffs[song][offset_diff] += 1

        clustered_matches = {}
        cluster_info = {}
        
        # For each song, find the dominant cluster
        for song, offset_counts in offset_diffs.items():
            total_matches = sum(offset_counts.values())
            
            # Dynamically determine number of clusters based on data distribution
            # Use more clusters to isolate true matches from noise
            n_clusters = min(5, max(1, total_matches // 50))
            
            best_cluster_count = fingerprinting.cluster_time_stamps(offset_counts, n_clusters=n_clusters)
            
            # Calculate cluster quality (ratio of largest cluster to total matches)
            cluster_quality = best_cluster_count / total_matches if total_matches > 0 else 0
            
            clustered_matches[song] = best_cluster_count
            cluster_info[song] = {
                'total': total_matches,
                'quality': cluster_quality,
                'best_cluster': best_cluster_count
            }

        # Filter by minimum threshold (500 matches in dominant cluster)
        filtered = {k: v for k, v in clustered_matches.items() if v >= 500}

        if filtered:
            top_matches = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:3]
            st.write("Top matches:")

            match_counts = [count for _, count in top_matches]

            # Check for multiple strong matches
            if len(match_counts) > 1:
                if match_counts[0] >= 1000 and match_counts[1] >= 1000:
                    diff = abs(match_counts[0] - match_counts[1])
                    if diff < 0.1 * match_counts[0]:
                        st.warning("⚠️ Multiple strong matches detected. Audio Sample may contain mixed sources or noise.")

            for i, (song_id, count) in enumerate(top_matches, 1):
                # Confidence labels based on dominant cluster size
                if count >= 1000:
                    confidence_label = "High Confidence"
                elif 750 <= count < 1000:
                    confidence_label = "Medium Confidence"
                elif 500 <= count < 750:
                    confidence_label = "Low Confidence"
                else:
                    confidence_label = "Unknown"

                percent = (count / sum(match_counts)) * 100 if sum(match_counts) > 0 else 0
                quality = cluster_info[song_id]['quality']
                
                # Get secrets with fallback
                client_id = st.secrets["CLIENT_ID"] if "CLIENT_ID" in st.secrets else os.getenv("CLIENT_ID")
                client_secret = st.secrets["CLIENT_SECRET"] if "CLIENT_SECRET" in st.secrets else os.getenv("CLIENT_SECRET")

                track = get_song_info(song_id, client_id, client_secret)

                if i == 1:
                    st.markdown(f"<div style='background-color:#FFD700; padding:10px; border-radius:5px;'>", unsafe_allow_html=True)
                    st.markdown(f"### **{i}. {track['title']} - {track['artist']}**")
                else:
                    if track:
                        st.markdown(f"### {i}. {track['title']} - {track['artist']}")

                st.write(f"Dominant cluster matches: {count}")
                st.write(f"Match Confidence: {confidence_label}")
                st.write(f"Cluster quality: {quality:.1%}")
                st.write(f"Match percentage: {percent:.1f}%")
                
                if track and 'album' in track and 'images' in track['album']:
                    img_url = track['album']['images'][0]['url']
                    st.image(img_url, width=150)
                    
                if track and 'external_urls' in track and 'spotify' in track['external_urls']:
                    url = track['external_urls']['spotify']
                    st.markdown(f"[Listen on Spotify]({url})")

                if i == 1:
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("No matches found above threshold.")
