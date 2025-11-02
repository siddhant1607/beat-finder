import numpy as np
import scipy.signal
import librosa
import scipy.ndimage as ndi
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

STANDARD_SR = 22050  # Use ONE sample rate for EVERYTHING
STANDARD_N_FFT = 4096
STANDARD_HOP_LENGTH = 512  # Changed from 1024 for finer time resolution
PEAK_NEIGHBORHOOD_SIZE = (10, 10)  # Reduced from (20,20) for better sensitivity
PEAK_THRESHOLD_DB = -30  # Raised from -40 (less aggressive)
FAN_VALUE = 15  # Increased from 10 for better coverage


def generate_audio_hashes_improved(peaks, fan_value=FAN_VALUE):
    hashes = []
    for i in range(len(peaks)):
        anchor_t, anchor_f = peaks[i]

        for j in range(1, min(fan_value + 1, len(peaks) - i)):
            target_t, target_f = peaks[i + j]
            delta_t = target_t - anchor_t

            if delta_t > 0 and delta_t < 512:
                h = (int(anchor_f), int(target_f), int(delta_t))
                hashes.append((h, anchor_t))

        if i > 0:
            for j in range(1, min(5, i + 1)):
                target_t, target_f = peaks[i - j]
                delta_t = anchor_t - target_t

                if delta_t > 0 and delta_t < 256:
                    h = (int(target_f), int(anchor_f), int(delta_t))
                    hashes.append((h, anchor_t))
    return hashes


def generate_spectrogram(audio_path, sr=STANDARD_SR, n_fft=STANDARD_N_FFT, hop_length=STANDARD_HOP_LENGTH):
    try:
        x, _ = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

    S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window=scipy.signal.windows.hamming)

    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    S_db_normalized = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8) * 80 - 40

    return S_db_normalized, sr, hop_length


def find_spectrogram_peaks(spectrogram_db, structure_size=PEAK_NEIGHBORHOOD_SIZE, threshold_db=PEAK_THRESHOLD_DB):
    is_local_max = ndi.maximum_filter(spectrogram_db, size=structure_size) == spectrogram_db
    meets_threshold = spectrogram_db > threshold_db
    peaks_mask = is_local_max & meets_threshold
    peak_indices = np.where(peaks_mask)
    peaks = list(zip(peak_indices[0], peak_indices[1]))
    peaks_m_omega = [(t, f) for f, t in peaks]
    return sorted(peaks_m_omega, key=lambda x: x[0])


def cluster_time_stamps(offset_diff_counts, n_clusters=3, min_cluster_size=5):
    if not offset_diff_counts:
        return 0, 0

    raw_data = []
    offset_map = {}
    idx = 0
    for offset_diff, count in offset_diff_counts.items():
        offset_map[idx] = offset_diff
        raw_data.extend([idx] * count)
        idx += 1

    if not raw_data:
        return 0, 0

    data = np.array(raw_data).reshape(-1, 1)
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    n_clusters = min(n_clusters, len(np.unique(data)))
    if n_clusters < 1:
        n_clusters = 1

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                            n_init='auto', batch_size=256)
    kmeans.fit(standardized_data)

    labels = kmeans.labels_
    label_counts = np.bincount(labels)
    best_label = np.argmax(label_counts)
    best_count = label_counts[best_label]

    cluster_mask = labels == best_label
    cluster_indices = np.where(cluster_mask)[0]
    original_offsets = [data[i, 0] for i in cluster_indices]
    best_offset = int(np.median(original_offsets))

    return best_count, best_offset


def create_audio_fingerprint(audio_path):
    result = generate_spectrogram(audio_path, sr=STANDARD_SR,
                                 n_fft=STANDARD_N_FFT,
                                 hop_length=STANDARD_HOP_LENGTH)
    if result is None:
        return []

    spectrogram_db, sr, hop_length = result

    print(f"Spectrogram: {spectrogram_db.shape[1]} frames, {spectrogram_db.shape[0]} bins")

    peaks = find_spectrogram_peaks(spectrogram_db)
    print(f"Extracted {len(peaks)} peaks")

    fingerprint_hashes = generate_audio_hashes_improved(peaks, fan_value=FAN_VALUE)
    print(f"Generated {len(fingerprint_hashes)} hashes")

    return fingerprint_hashes


def analyze_fingerprint_quality(hashes):
    if not hashes:
        print("No hashes to analyze")
        return

    hash_values = [h[0] for h in hashes]
    unique_hashes = len(set(hash_values))
    collision_rate = 1 - (unique_hashes / len(hashes))

    print(f"\n--- Fingerprint Quality Analysis ---")
    print(f"Total hashes: {len(hashes)}")
    print(f"Unique hashes: {unique_hashes}")
    print(f"Collision rate: {collision_rate*100:.2f}%")
    print(f"(Collision rate > 5% indicates hash quality issues)")

    times = [t for _, t in hashes]
    print(f"Time range: {min(times)} to {max(times)}")
    print(f"Average time between hashes: {(max(times) - min(times)) / len(times):.2f}")
