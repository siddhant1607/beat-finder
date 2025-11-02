import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re

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
                # Consistent generated_id
                generated_id = re.sub(r'\W+', '', title.lower()) + re.sub(r'\W+', '', artist.lower())
                tracks.append({
                    'title': title,
                    'artist': artist,
                    'duration': track['duration_ms'] / 1000,
                    'id': generated_id
                })
        if results.get('next'):
            results = sp.next(results)
        else:
            break
    return tracks
