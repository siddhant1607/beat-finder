import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re

def sanitize(text):
    return re.sub(r'\W+', '_', text.lower())

def generate_song_id(title, artist):
    return f"{sanitize(title)}__{sanitize(artist)}"

def get_spotify_tracks(playlist_url, client_id, client_secret):
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))
    results = sp.playlist_items(playlist_url)
    tracks = []
    while results:
        for item in results['items']:
            track = item['track']
            if track:
                title = track['name']
                artist = track['artists'][0]['name']
                song_id = generate_song_id(title, artist)
                # Add the properly normalized ID
                tracks.append({
                    'title': title,
                    'artist': artist,
                    'duration': track['duration_ms'] / 1000,
                    'album': track.get('album', {}),
                    'external_urls': track.get('external_urls', {}),
                    'id': song_id
                })
        if results.get('next'):
            results = sp.next(results)
        else:
            break
    return tracks
