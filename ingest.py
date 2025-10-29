def process_playlist(playlist_url, client_id, client_secret):
    fingerprint_db = load_fingerprint_db(FINGERPRINT_DB_PATH)
    tracks = get_spotify_tracks(playlist_url, client_id, client_secret)

    for track in tracks:
        title = track.get('title', '')
        artist = track.get('artist', '')
        song_id = generate_song_id(title, artist)
        wav_filename = f"{song_id}.wav"
        wav_path = os.path.join(AUDIO_FOLDER, wav_filename)

        while not os.path.isfile(wav_path):
            print(f"Missing WAV file for '{song_id}'.")
            print(f"Please add the file named exactly: '{wav_filename}' to '{AUDIO_FOLDER}'")
            input("Press Enter once you have added the file...")

        if already_fingerprinted(song_id, fingerprint_db):
            print(f"Skipping '{song_id}', already fingerprinted.")
            continue

        print(f"Processing '{song_id}'")
        y, sr = librosa.load(wav_path, sr=16000)
        S_mag = process_segment(y, sr)
        peaks = extract_peaks_bandwise(S_mag)
        pair_hashes = generate_pair_hashes(peaks)

        for h, t in pair_hashes:
            fingerprint_db[int(h)].append((song_id, t))

        os.remove(wav_path)
        print(f"Finished and deleted '{wav_filename}'")

    save_fingerprint_db(FINGERPRINT_DB_PATH, fingerprint_db)
    print("\nAll tracks processed.")
