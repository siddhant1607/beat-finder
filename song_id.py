import re
from spotify_util import get_spotify_tracks  # Assuming you have this
import os

def sanitize(text):
    # Lowercase, replace non-alphanumeric with underscore
    return re.sub(r'\W+', '_', text.lower())

def generate_song_id(title, artist):
    return f"{sanitize(title)}__{sanitize(artist)}"

def print_song_ids(playlist_url, client_id, client_secret):
    tracks = get_spotify_tracks(playlist_url, client_id, client_secret)
    for track in tracks:
        title = track.get('title', '')
        artist = track.get('artist', '')
        song_id = generate_song_id(title, artist)
        print(f"Title: '{title}' | Artist: '{artist}' | Song ID for renaming: '{song_id}'")

if __name__ == "__main__":
    # Read Spotify credentials from environment variables
    client_id = os.environ.get("SPOTIPY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
    playlist_url = "https://open.spotify.com/playlist/7AFzTreiVAZpYU8wYW3fp9?si=mf4MJmI3Skmx_J_MAwP-CA"  # Replace with your playlist URL

    if not client_id or not client_secret:
        raise ValueError("Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET as environment variables")

    print_song_ids(playlist_url, client_id, client_secret)
