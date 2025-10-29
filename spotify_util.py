import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

def get_spotify_tracks(playlist_url, client_id, client_secret):
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))
    tracks = []
    results = sp.playlist_items(playlist_url)
    while results:
        for item in results['items']:
            track = item['track']
            if track is None:
                continue
            tracks.append({
                'title': track['name'],
                'artist': track['artists'][0]['name'],
                'duration': track['duration_ms'] / 1000,
            })
        # Paginate if needed
        if results['next']:
            results = sp.next(results)
        else:
            break
    return tracks
