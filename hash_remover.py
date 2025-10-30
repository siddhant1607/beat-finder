import os
import pickle
from collections import defaultdict

FINGERPRINT_DB_PATH = "fingerprint_db.pkl"

def load_fingerprint_db(db_path):
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            fingerprint_db = pickle.load(f)
        print(f"Loaded fingerprint database with {len(fingerprint_db)} unique hashes.")
    else:
        fingerprint_db = defaultdict(list)
        print("No existing fingerprint database found. Created new empty database.")
    return fingerprint_db

def save_fingerprint_db(db_path, fingerprint_db):
    with open(db_path, "wb") as f:
        pickle.dump(fingerprint_db, f)
    print(f"Fingerprint database saved: {len(fingerprint_db)} unique hashes.")

def list_song_ids(fingerprint_db):
    song_ids = set()
    for hashes in fingerprint_db.values():
        for song_id, _ in hashes:
            song_ids.add(song_id)
    return sorted(list(song_ids))

def delete_hashes_for_song(fingerprint_db, song_id_to_delete):
    hashes_removed = 0
    keys_to_delete = []
    for h, entries in fingerprint_db.items():
        # Filter out entries matching song_id_to_delete
        new_entries = [entry for entry in entries if entry[0] != song_id_to_delete]
        if len(new_entries) != len(entries):
            hashes_removed += (len(entries) - len(new_entries))
            if new_entries:
                fingerprint_db[h] = new_entries
            else:
                keys_to_delete.append(h)
    # Remove keys that have empty lists after deletion
    for k in keys_to_delete:
        del fingerprint_db[k]
    return hashes_removed

def main():
    print("Loading fingerprint database...")
    fingerprint_db = load_fingerprint_db(FINGERPRINT_DB_PATH)

    print("\nSong IDs in database:")
    song_ids = list_song_ids(fingerprint_db)
    for sid in song_ids:
        print("-", sid)

    song_id_to_delete = input("\nEnter the exact song ID to delete all associated hashes (or 'exit' to quit): ").strip()
    if song_id_to_delete.lower() == "exit":
        print("Exiting without changes.")
        return

    if song_id_to_delete not in song_ids:
        print(f"Song ID '{song_id_to_delete}' not found in database. No changes made.")
        return

    removed = delete_hashes_for_song(fingerprint_db, song_id_to_delete)
    if removed > 0:
        print(f"Removed {removed} hashes for song ID '{song_id_to_delete}'.")
        save_fingerprint_db(FINGERPRINT_DB_PATH, fingerprint_db)
    else:
        print("No hashes were removed.")

if __name__ == "__main__":
    main()
