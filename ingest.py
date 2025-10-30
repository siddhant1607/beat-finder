import os
import re
import pickle
from collections import defaultdict
import librosa

from fingerprinting import process_segment, extract_peaks_bandwise, generate_pair_hashes
from spotify_util import get_spotify_tracks


AUDIO_FOLDER = "/workspaces/beat-finder/SongDB"
FINGERPRINT_DB_PATH = "/workspaces/beat-finder/fingerprint_db.pkl"

def load_fingerprint_db(db_path):
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            fingerprint_db = pickle.load(f)
        print(f"Loaded fingerprint database with {len(fingerprint_db)} unique hashes.")
    else:
        fingerprint_db = defaultdict(list)
        print("Creating new fingerprint database.")
    return fingerprint_db


def save_fingerprint_db(db_path, fingerprint_db):
    with open(db_path, "wb") as f:
        pickle.dump(fingerprint_db, f)
    print(f"Fingerprint database saved: {len(fingerprint_db)} unique hashes.")

def sanitize(text):
    return re.sub(r'\W+', '_', text.lower())

def generate_song_id(title, artist):
    return f"{sanitize(title)}__{sanitize(artist)}"

def already_fingerprinted(song_id, fingerprint_db):
    return any(entry[0] == song_id for entries in fingerprint_db.values() for entry in entries)

def process_playlist(playlist_url, client_id, client_secret):
    fingerprint_db = load_fingerprint_db(FINGERPRINT_DB_PATH)
    tracks = get_spotify_tracks(playlist_url, client_id, client_secret)

    for track in tracks:
        title = track.get('title', '')
        artist = track.get('artist', '')
        song_id = generate_song_id(title, artist)
        audio_filename = f"{song_id}.mp3"
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)

        while not os.path.isfile(audio_path):
            print(f"\nMissing MP3 file for '{song_id}'.")
            print(f"Please add the file named exactly: '{audio_filename}' to folder: '{AUDIO_FOLDER}'")
            choice = input("Press Enter once you have added the file, or type 'skip' to move to the next song: ").strip().lower()
            if choice == 'skip':
                print(f"Skipping '{song_id}' as requested.")
                break
        else:
            print(f"Processing '{song_id}'")
            y, sr = librosa.load(audio_path, sr=16000)
            S_mag = process_segment(y, sr)
            peaks = extract_peaks_bandwise(S_mag)
            pair_hashes = generate_pair_hashes(peaks)

            for h, t in pair_hashes:
                fingerprint_db[int(h)].append((song_id, t))

            os.remove(audio_path)
            print(f"Finished and deleted '{audio_filename}'")

    save_fingerprint_db(FINGERPRINT_DB_PATH, fingerprint_db)
    print("\nAll tracks processed.")

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
playlist_url = "https://open.spotify.com/playlist/7AFzTreiVAZpYU8wYW3fp9?si=2uheFNb4Q7a-LDcWcufb5g"

process_playlist(playlist_url, client_id, client_secret)
