import streamlit as st
import soundfile as sf
import io
from fingerprinting import process_segment, extract_peaks_bandwise, generate_pair_hashes
from db import get_db
import os

client_id = st.secrets["SPOTIPY_CLIENT_ID"]
client_secret = st.secrets["SPOTIPY_CLIENT_SECRET"]
mongo_url = st.secrets["MONGO_URL"]

st.title("🎤 BeatFinder Demo")
st.write("Record audio and try to recognize it from your 20-song database!")

# Record or upload audio sample
audio_bytes = st.audio_input("Record a 10s audio sample (or upload any WAV file)", sample_rate=16000)

if audio_bytes is not None:
    # Decode to numpy array
    y, sr = sf.read(io.BytesIO(audio_bytes))
    
    # Generate hashes
    S_mag = process_segment(y, sr)
    peaks = extract_peaks_bandwise(S_mag)
    pair_hashes = generate_pair_hashes(peaks)
    st.write(f"Fingerprinted {len(pair_hashes)} pairs.")

    # Matching
    if st.button("Find best match"):
        db_collection = get_db()
        # build a counter for (song_id, match count)
        from collections import defaultdict
        score = defaultdict(int)
        for h, t in pair_hashes:
            results = db_collection.find_one({"hash": int(h)})
            if results:
                for entry in results["entries"]:
                    score[entry["song_id"]] += 1
        if score:
            sorted_scores = sorted(score.items(), key=lambda x: x[1], reverse=True)
            st.write("Top matches:")
            for song_id, count in sorted_scores[:3]:
                st.write(f"{song_id} - {count} matching hashes")
        else:
            st.write("No matches found.")
