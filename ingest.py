import os
import re
import librosa
from fingerprinting import process_segment, extract_peaks_bandwise, generate_pair_hashes
from db import get_db
from spotify_util import get_spotify_tracks

# Folder where your manual WAV files are stored
AUDIO_FOLDER = "/workspaces/beat-finder/SongDB"

def sanitize(text):
    return re.sub(r'\W+', '_', text.lower())

def generate_song_id(title, artist):
    return f"{sanitize(title)}__{sanitize(artist)}"

def already_fingerprinted(song_id, db_collection):
    return db_collection.find_one({"entries.song_id": song_id}) is not None

def process_spotify_playlist_to_db(playlist_url, client_id, client_secret, db_collection):
    tracks = get_spotify_tracks(playlist_url, client_id, client_secret)
    missing_files = []

    for track in tracks:
        title = track.get('title', '')
        artist = track.get('artist', '')
        unique_song_id = generate_song_id(title, artist)

        wav_filename = f"{unique_song_id}.wav"
        wav_path = os.path.join(AUDIO_FOLDER, wav_filename)

        if not os.path.isfile(wav_path):
            missing_files.append(wav_filename)
            continue

        if already_fingerprinted(unique_song_id, db_collection):
            print(f"Skipping '{unique_song_id}', already fingerprinted in DB.")
            continue

        print(f"Processing: {unique_song_id}")
        y, sr = librosa.load(wav_path, sr=16000)
        S_mag = process_segment(y, sr)
        peaks = extract_peaks_bandwise(S_mag)
        pair_hashes = generate_pair_hashes(peaks)

        for h, t in pair_hashes:
            db_collection.update_one(
                {
                    "hash": int(h),
                    "entries.song_id": {"$ne": unique_song_id}
                },
                {"$addToSet": {"entries": {"song_id": unique_song_id, "offset": float(t)}}},
                upsert=True
            )
        # Delete WAV to save storage after processing
        os.remove(wav_path)
        print(f"Finished and deleted: {wav_filename}")

    if missing_files:
        print("\nThe following audio files are missing. Please add them to the downloads folder:")
        for fname in missing_files:
            print(" -", fname)
    else:
        print("\nAll audio files found and processed.")

if __name__ == "__main__":
    client_id = os.environ.get("SPOTIPY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET must be set as environment variables.")

    db_collection = get_db()

    PLAYLIST_URL = "https://open.spotify.com/playlist/7AFzTreiVAZpYU8wYW3fp9?si=mS5NsOgnTFCgl-QrzVEUxw"

    process_spotify_playlist_to_db(PLAYLIST_URL, client_id, client_secret, db_collection)

    print("Batch ingest complete!")
