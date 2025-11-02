# streamlit run streamlit_app.py

import streamlit as st
import soundfile as sf
import io
import numpy as np
from collections import defaultdict, Counter
import pickle
import re
import os

import fingerprinting as fingerprinting
from fingerprinting import STANDARD_SR
from spotify_util import get_spotify_tracks

LOCAL_DB_PATH = "fingerprint_db.pkl"

def load_fingerprint_db():
    with open(LOCAL_DB_PATH, "rb") as f:
        return pickle.load(f)

fingerprint_db = load_fingerprint_db()

st.title("🎤 BeatFinder")
st.write("Record a 10-15 second audio sample to identify songs from the database!")

# Settings sidebar
st.sidebar.header("Settings")
match_threshold = st.sidebar.slider("Match threshold (minimum dominant matches)", 
                                     min_value=30, max_value=500, value=100, step=10)
time_diff_tolerance = st.sidebar.slider("Time difference tolerance (frames)", 
                                        min_value=1, max_value=10, value=2, step=1)

audio_bytes = st.audio_input("Record a 10-15s audio sample", sample_rate=STANDARD_SR)

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

# ============================================================================
# FIX #7: IMPROVED CONFIDENCE SCORING
# ============================================================================
def calculate_confidence(dominant_matches, total_matches, unique_offsets, 
                        time_tightness, abs_threshold=100):
    """
    Calculate a more meaningful confidence score.
    
    Considers:
    1. Absolute number of dominant matches (must be high)
    2. Ratio of dominant to total matches (must be concentrated)
    3. Time offset tightness (matches should cluster well)
    4. Uniqueness in offsets (too many unique offsets = noise)
    """
    
    # Confidence factors:
    factor1 = min(dominant_matches / abs_threshold, 1.0)  # How many matches vs minimum
    factor2 = dominant_matches / max(total_matches, 1)    # How concentrated (0-1)
    factor3 = 1.0 - min(unique_offsets / max(total_matches, 1) * 0.5, 1.0)  # Offset concentration
    factor4 = min(time_tightness, 1.0)  # How tight is the cluster
    
    # Weighted average
    confidence = (factor1 * 0.4 + factor2 * 0.3 + factor3 * 0.2 + factor4 * 0.1) * 100
    return confidence

if audio_bytes is not None:
    # Read and trim audio
    data, samplerate = sf.read(io.BytesIO(audio_bytes.getvalue()))
    
    # Resample if necessary (should already be at STANDARD_SR but just in case)
    if samplerate != STANDARD_SR:
        data = librosa.resample(data, orig_sr=samplerate, target_sr=STANDARD_SR)
        samplerate = STANDARD_SR
    
    max_length = 15 * samplerate
    trimmed_data = data[:max_length]
    
    temp_audio_path = "temp_query.wav"
    sf.write(temp_audio_path, trimmed_data, samplerate)
    
    # Generate fingerprint hashes
    query_hashes = fingerprinting.create_audio_fingerprint(temp_audio_path)
    os.remove(temp_audio_path)

    st.write(f"**Fingerprinted {len(query_hashes)} hashes from query**")
    
    # Analyze fingerprint quality
    if query_hashes:
        unique_query_hashes = len(set([h[0] for h in query_hashes]))
        collision_rate = 1 - (unique_query_hashes / len(query_hashes))
        st.write(f"*Unique hashes: {unique_query_hashes} (collision rate: {collision_rate*100:.1f}%)*")

    if st.button("🔍 Find best match"):
        if not query_hashes:
            st.error("No hashes generated from query. Try a clearer recording.")
        else:
            # Step 1: For each query hash, find all database entries
            song_time_diffs = defaultdict(list)
            
            hash_matches = 0
            for query_hash, query_anchor_time in query_hashes:
                hash_key = str(query_hash)
                if hash_key in fingerprint_db:
                    db_entries = fingerprint_db[hash_key]
                    for song_id, db_anchor_time in db_entries:
                        time_diff = db_anchor_time - query_anchor_time
                        song_time_diffs[song_id].append(time_diff)
                        hash_matches += 1

            st.write(f"**Found {hash_matches} hash matches across songs**")

            if not song_time_diffs:
                st.error("❌ No matches found in database.")
                st.info("Possible reasons: Database might be empty, or query audio is too different from database samples.")
            else:
                # Step 2: For each song, cluster time differences
                song_results = {}
                
                for song_id, time_diffs in song_time_diffs.items():
                    # Group similar offsets with tolerance
                    grouped_diffs = []
                    for td in time_diffs:
                        grouped_diffs.append(round(td / time_diff_tolerance) * time_diff_tolerance)
                    
                    diff_counter = Counter(grouped_diffs)
                    
                    # Cluster the time differences
                    clustered_count, best_offset = fingerprinting.cluster_time_stamps(
                        diff_counter, n_clusters=3, min_cluster_size=5
                    )
                    
                    # Measure time tightness (how concentrated are matches around best offset)
                    if diff_counter:
                        offset_deviations = [abs(od - best_offset) for od in grouped_diffs]
                        avg_deviation = np.mean(offset_deviations)
                        time_tightness = 1.0 / (1.0 + avg_deviation / 10.0)  # Normalize to 0-1
                    else:
                        time_tightness = 0
                    
                    confidence = calculate_confidence(
                        clustered_count, len(time_diffs), len(diff_counter),
                        time_tightness, abs_threshold=match_threshold
                    )
                    
                    song_results[song_id] = {
                        'dominant_matches': clustered_count,
                        'total_matches': len(time_diffs),
                        'unique_offsets': len(diff_counter),
                        'top_offset': best_offset,
                        'confidence': confidence,
                        'time_tightness': time_tightness
                    }

                # Step 3: Filter by threshold
                filtered_songs = {
                    k: v for k, v in song_results.items()
                    if v['dominant_matches'] >= match_threshold
                }

                if filtered_songs:
                    top_matches = sorted(
                        filtered_songs.items(), 
                        key=lambda x: x[1]['confidence'], 
                        reverse=True
                    )[:3]
                    
                    st.markdown("## 🎵 Top Matches:")
                    
                    for i, (song_id, result) in enumerate(top_matches, 1):
                        confidence = result['confidence']
                        
                        # Confidence color coding
                        if confidence >= 80:
                            conf_label = "🟢 Very High"
                            color = "#00FF00"
                        elif confidence >= 60:
                            conf_label = "🟡 High"
                            color = "#FFD700"
                        elif confidence >= 40:
                            conf_label = "🟠 Medium"
                            color = "#FFA500"
                        else:
                            conf_label = "🔴 Low"
                            color = "#FF6B6B"
                        
                        try:
                            client_id = st.secrets["CLIENT_ID"]
                            client_secret = st.secrets["CLIENT_SECRET"]
                        except:
                            client_id = os.getenv("CLIENT_ID")
                            client_secret = os.getenv("CLIENT_SECRET")

                        track = get_song_info(song_id, client_id, client_secret)

                        if track:
                            if i == 1:
                                st.markdown(
                                    f"<div style='background-color:{color}; padding:15px; border-radius:8px; border:2px solid black;'>",
                                    unsafe_allow_html=True
                                )
                            
                            st.markdown(f"### #{i} - {track['title']} by {track['artist']}")
                            
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                st.metric("Dominant Matches", result['dominant_matches'])
                                st.metric("Confidence", f"{confidence:.1f}%")
                            with col2:
                                st.metric("Time Tightness", f"{result['time_tightness']:.2f}")
                                st.metric("Total Matches", result['total_matches'])
                            
                            if track.get('external_urls') and track['external_urls'].get('spotify'):
                                st.markdown(f"[🎵 Listen on Spotify]({track['external_urls']['spotify']})")
                            
                            if i == 1:
                                st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.divider()
                else:
                    st.warning(f"⚠️ No strong matches found (threshold: {match_threshold} dominant matches).")
                    st.info("Try adjusting the match threshold in the sidebar or check if the database has enough songs.")
