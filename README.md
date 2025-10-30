# BeatFinder 🎤

BeatFinder is a music recognition web app built with Streamlit. It allows you to record or upload a short audio clip and tries to identify the song by matching audio fingerprints against a pre-built song database. The results show the most likely matches along with confidence percentages and Spotify links.

---

## Features

- Record or upload a 10-second audio sample directly in the browser.
- Extract audio fingerprints based on spectral peak pairs.
- Efficiently match audio fingerprints to a local song database (`fingerprint_db.pkl`).
- Show top 5 matching songs with confidence percentages.
- Display song metadata including title, artist, album art, and Spotify links.
- Utilizes Spotify API for detailed song information.
- Pitch-shift compensation to improve recognition on slightly off-pitch inputs.

---

## Technologies Used

- Python 3
- Streamlit for the web UI
- librosa for audio processing and spectral analysis
- numpy and scipy for signal processing
- Spotify Web API for song metadata retrieval
- Pickle for local fingerprint database storage and loading

---

## Setup and Installation

1. Clone this repository.

2. Install dependencies:

pip install -r requirements.txt

text

3. Create a `.streamlit/secrets.toml` file with your Spotify API credentials:

CLIENT_ID = "your_spotify_client_id"
CLIENT_SECRET = "your_spotify_client_secret"

text

4. Run the Streamlit app:

streamlit run streamlit_app.py

text

5. Use the web UI to record or upload audio and identify songs.

---

## Project Structure

- `streamlit_app.py` - Main Streamlit web app frontend.
- `fingerprinting.py` - Module for audio fingerprint extraction.
- `spotify_util.py` - Spotify API integration and track info fetching.
- `fingerprint_db.pkl` - Precomputed fingerprint database file.
- `.streamlit/secrets.toml` - Secure storage of Spotify API secrets.

---

## Limitations and Future Work

- Current fingerprinting relies on spectral peak pairs and is sensitive to pitch and noise changes.
- Pitch-shift compensation is applied, but more advanced invariant features could improve robustness.
- Database size is limited compared to commercial systems like Shazam.
- Future improvements may use machine learning for enhanced matching and noise robustness.

---

## Acknowledgements

This project is inspired by classic audio fingerprinting research and the Shazam music recognition algorithm. Thanks to open-source libraries like Streamlit, librosa, and Spotipy.

---

## License

This project is released under the MIT License.

---

Enjoy discovering your music with BeatFinder!
