import streamlit as st
import soundfile as sf
import io
import numpy as np
from collections import defaultdict, Counter
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
    # Read and trim audio
    data, samplerate = sf.read(io.BytesIO(audio_bytes.getvalue()))
    max_length = 10 * samplerate
    trimmed_data = data[:max_length]
    
    temp_audio_path = "temp_query.wav"
    sf.write(temp_audio_path, trimmed_data, samplerate)
    
    # Generate fingerprint hashes
    query_hashes = fingerprinting.create_audio_fingerprint(temp_audio_path)
    os.remove(temp_audio_path)

    st.write(f"Fingerprinted {len(query_hashes)} hashes.")

    if st.button("Find best match"):
        # ===== MATCHING LOGIC FROM SCRATCH =====
        
        # Step 1: For each query hash, find all database entries
        # Store: song_id -> list of (anchor_time_diff) for all matching hashes
        song_time_diffs = defaultdict(list)
        
        hash_matches = 0
        for query_hash, query_anchor_time in query_hashes:
            if query_hash in fingerprint_db:
                # This hash exists in database
                db_entries = fingerprint_db[query_hash]
                for song_id, db_anchor_time in db_entries:
                    # Calculate the time difference (alignment offset)
                    # If query matches db at this offset, it helps identify song position
                    time_diff = db_anchor_time - query_anchor_time
                    song_time_diffs[song_id].append(time_diff)
                    hash_matches += 1

        st.write(f"Found {hash_matches} hash matches across songs.")

        if not song_time_diffs:
            st.write("No matches found in database.")
        else:
            # Step 2: For each song, convert time diffs to Counter and cluster
            song_results = {}
            
            for song_id, time_diffs in song_time_diffs.items():
                # Round time differences to group similar offsets
                rounded_diffs = [round(t, 0) for t in time_diffs]
                diff_counter = Counter(rounded_diffs)
                
                # Cluster to find dominant time offset
                clustered_count = fingerprinting.cluster_time_stamps(diff_counter, n_clusters=2)
                
                song_results[song_id] = {
                    'dominant_matches': clustered_count,
                    'total_matches': len(time_diffs),
                    'unique_offsets': len(diff_counter),
                    'top_offset': diff_counter.most_common(1)[0][0] if diff_counter else 0
                }

            # Step 3: Filter by minimum threshold (500 matches in dominant cluster)
            filtered_songs = {
                k: v['dominant_matches'] 
                for k, v in song_results.items() 
                if v['dominant_matches'] >= 500
            }

            if filtered_songs:
                # Sort by dominant cluster count
                top_matches = sorted(filtered_songs.items(), key=lambda x: x[1], reverse=True)[:3]
                
                st.write("## Top matches:")

                match_counts = [count for _, count in top_matches]

                # Check for ambiguity (multiple strong matches)
                if len(match_counts) > 1:
                    if match_counts[0] >= 1000 and match_counts[1] >= 1000:
                        diff = abs(match_counts[0] - match_counts[1])
                        if diff < 0.1 * match_counts[0]:
                            st.warning("⚠️ Multiple strong matches detected. Audio Sample may contain mixed sources or noise.")

                # Display results
                for i, (song_id, dominant_count) in enumerate(top_matches, 1):
                    result = song_results[song_id]
                    total_matches = result['total_matches']
                    cluster_quality = dominant_count / total_matches if total_matches > 0 else 0
                    percent = (dominant_count / sum(match_counts)) * 100 if sum(match_counts) > 0 else 0
                    
                    # Confidence based on dominant cluster count
                    if dominant_count >= 1000:
                        confidence_label = "High Confidence"
                    elif 750 <= dominant_count < 1000:
                        confidence_label = "Medium Confidence"
                    elif 500 <= dominant_count < 750:
                        confidence_label = "Low Confidence"
                    else:
                        confidence_label = "Very Low"

                    # Get credentials
                    try:
                        client_id = st.secrets["CLIENT_ID"]
                        client_secret = st.secrets["CLIENT_SECRET"]
                    except:
                        client_id = os.getenv("CLIENT_ID")
                        client_secret = os.getenv("CLIENT_SECRET")

                    track = get_song_info(song_id, client_id, client_secret)

                    if track:
                        if i == 1:
                            st.markdown(f"<div style='background-color:#FFD700; padding:10px; border-radius:5px;'>", unsafe_allow_html=True)
                            st.markdown(f"### **{i}. {track['title']} - {track['artist']}**")
                        else:
                            st.markdown(f"### {i}. {track['title']} - {track['artist']}")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Dominant cluster matches:** {dominant_count}")
                            st.write(f"**Total hash matches:** {total_matches}")
                            st.write(f"**Cluster quality:** {cluster_quality:.1%}")
                            st.write(f"**Match confidence:** {confidence_label}")
                            st.write(f"**Relative score:** {percent:.1f}%")
                        
                        with col2:
                            if track.get('album') and track['album'].get('images'):
                                img_url = track['album']['images'][0]['url']
                                st.image(img_url, width=150)

                        if track.get('external_urls') and track['external_urls'].get('spotify'):
                            url = track['external_urls']['spotify']
                            st.markdown(f"[🎵 Listen on Spotify]({url})")

                        if i == 1:
                            st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.write("No strong matches found (threshold: 500 dominant cluster matches).")
