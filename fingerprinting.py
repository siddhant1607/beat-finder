import numpy as np
import scipy.signal
import librosa
import scipy.ndimage as ndi
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# --- I. Spectrogram Generation (Section II.A) ---
def generate_spectrogram(audio_path, sr=44100, n_fft=4096, hop_length=1024):
    try:
        # Load audio signal (x(t))
        x, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

    S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window=scipy.signal.windows.hamming)
    
    S_mag = np.abs(S)
    
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    
    return S_db, sr, hop_length

# --- II. Feature Extraction: Peak Detection (Section II.B) ---
def find_spectrogram_peaks(spectrogram_db, structure_size=(3, 3), threshold_db=-80):

    is_local_max = ndi.maximum_filter(spectrogram_db, size=structure_size) == spectrogram_db
    
    meets_threshold = spectrogram_db > threshold_db
    
    peaks_mask = is_local_max & meets_threshold
    
    peak_indices = np.where(peaks_mask)
    
    peaks = list(zip(peak_indices[0], peak_indices[1]))
    
    peaks_m_omega = [(t, f) for f, t in peaks]

    return sorted(peaks_m_omega, key=lambda x: x[0]) # Sort by time index 'm'

# --- III. Hash Generation (Section II.C) ---
def generate_audio_hashes(peaks, fan_value=10):

    hashes = []
    
    for i in range(len(peaks)):
        anchor_t, anchor_f = peaks[i] # P_anchor = P(m, omega) [cite: 81]
        
        for j in range(1, fan_value + 1):
            if i + j < len(peaks):
                target_t, target_f = peaks[i + j] # P_target
                
                delta_t = target_t - anchor_t
                
                if delta_t > 0 and delta_t < 256: 
                    # Create a composite hash (a unique identifier)
                    h = (anchor_f << 16) | (target_f << 8) | delta_t
                    
                    # Store (hash_value, anchor_time)
                    hashes.append((h, anchor_t))
                    
    return hashes

# --- IV. Main Execution and Example ---
def create_audio_fingerprint(audio_path):
    
    # 1. Generate Spectrogram
    spectrogram_db, sr, hop_length = generate_spectrogram(audio_path)
    if spectrogram_db is None:
        return []

    print(f"Spectrogram created: {spectrogram_db.shape[1]} time frames, {spectrogram_db.shape[0]} frequency bins.")

    # 2. Extract Peaks (Features)
    peaks = find_spectrogram_peaks(spectrogram_db)
    print(f"Extracted {len(peaks)} salient peaks.")

    # 3. Generate Hashes
    fingerprint_hashes = generate_audio_hashes(peaks)
    print(f"Generated {len(fingerprint_hashes)} hashes.")
    
    return fingerprint_hashes


def cluster_time_stamps(offset_diff_counts, n_clusters=1):
    # 1. Expand the data: Create a list where each offset difference (MT) 
    # is repeated by its match count. This gives the clustering algorithm 
    # the correct weight.
    raw_data = []
    for offset_diff, count in offset_diff_counts.items():
        raw_data.extend([offset_diff] * count)

    if not raw_data:
        return 0

    # Reshape to (n_samples, 1) for scikit-learn
    data = np.array(raw_data).reshape(-1, 1)

    # 2. Standardization (Standardized Time, ST)
    # The clustering is performed on ST = (MT - MMT) / SDMT
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    # 3. Apply MiniBatchKMeans
    # We use a tunable number of clusters. The correct matches should fall 
    # into the largest cluster (representing the true dominant time shift).
    
    # Ensure n_clusters is not more than the number of unique samples
    n_clusters = min(n_clusters, len(np.unique(standardized_data)))
    if n_clusters < 1: n_clusters = 1 # Must have at least one cluster

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init='auto', batch_size=256)
    kmeans.fit(standardized_data)
    
    # 4. Find the largest cluster (the true dominant offset)
    labels = kmeans.labels_
    label_counts = np.bincount(labels)
    best_count = np.max(label_counts)

    return best_count