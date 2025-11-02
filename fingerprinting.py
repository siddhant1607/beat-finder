# CORRECTED fingerprinting.py with fixes for hash collisions and sample rate consistency

import numpy as np
import scipy.signal
import librosa
import scipy.ndimage as ndi
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# ============================================================================
# FIX #1: STANDARDIZED PARAMETERS (Must be consistent everywhere!)
# ============================================================================
STANDARD_SR = 22050  # Use ONE sample rate for EVERYTHING
STANDARD_N_FFT = 4096
STANDARD_HOP_LENGTH = 512  # Changed from 1024 for finer time resolution
PEAK_NEIGHBORHOOD_SIZE = (10, 10)  # Reduced from (20,20) for better sensitivity
PEAK_THRESHOLD_DB = -30  # Raised from -40 (less aggressive)
FAN_VALUE = 15  # Increased from 10 for better coverage

# ============================================================================
# FIX #2: BETTER HASH GENERATION (Reduce collisions)
# ============================================================================
def generate_audio_hashes_improved(peaks, fan_value=FAN_VALUE):
    """
    Generate hashes using a better constellation map approach.
    
    Instead of simple bit-shifting with hash collisions, we:
    1. Use tuples directly (Python handles them better)
    2. Include multiple anchors per peak
    3. Add backward references for robustness
    """
    hashes = []
    
    for i in range(len(peaks)):
        anchor_t, anchor_f = peaks[i]
        
        # Forward references (next peaks)
        for j in range(1, min(fan_value + 1, len(peaks) - i)):
            target_t, target_f = peaks[i + j]
            delta_t = target_t - anchor_t
            
            # Only keep reasonable time differences
            if delta_t > 0 and delta_t < 512:
                # Use tuple for hash (no collision with other (f1, f2, dt) combinations!)
                # The tuple itself becomes the hash key
                h = (int(anchor_f), int(target_f), int(delta_t))
                hashes.append((h, anchor_t))
        
        # Optional: Include backward references for robustness
        # (peaks that can match this peak at earlier times)
        if i > 0:
            for j in range(1, min(5, i + 1)):  # Look back at most 5 peaks
                target_t, target_f = peaks[i - j]
                delta_t = anchor_t - target_t
                
                if delta_t > 0 and delta_t < 256:
                    h = (int(target_f), int(anchor_f), int(delta_t))
                    hashes.append((h, anchor_t))
    
    return hashes


# ============================================================================
# FIX #3: CONSISTENT SPECTROGRAM GENERATION
# ============================================================================
def generate_spectrogram(audio_path, sr=STANDARD_SR, n_fft=STANDARD_N_FFT, 
                        hop_length=STANDARD_HOP_LENGTH):
    """
    Generate spectrogram with STANDARD parameters.
    CRITICAL: Always resample to STANDARD_SR to ensure consistency!
    """
    try:
        # Load AND resample to standard rate
        x, _ = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

    # Compute STFT with Hamming window (as per paper)
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, 
                     window=scipy.signal.windows.hamming)
    
    # Get magnitude and convert to dB (log scale)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    
    # Normalize to reduce false peaks from loud vs quiet songs
    S_db_normalized = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8) * 80 - 40
    
    return S_db_normalized, sr, hop_length


# ============================================================================
# FIX #4: IMPROVED PEAK DETECTION
# ============================================================================
def find_spectrogram_peaks(spectrogram_db, structure_size=PEAK_NEIGHBORHOOD_SIZE, 
                           threshold_db=PEAK_THRESHOLD_DB):
    """
    Find peaks with adaptive thresholding.
    """
    # Local maxima detection using morphological operations
    is_local_max = ndi.maximum_filter(spectrogram_db, size=structure_size) == spectrogram_db
    
    # Apply threshold
    meets_threshold = spectrogram_db > threshold_db
    
    # Combine conditions
    peaks_mask = is_local_max & meets_threshold
    
    # Get indices
    peak_indices = np.where(peaks_mask)
    peaks = list(zip(peak_indices[0], peak_indices[1]))
    
    # Convert to (time, frequency) format for easier pairing
    peaks_m_omega = [(t, f) for f, t in peaks]
    
    return sorted(peaks_m_omega, key=lambda x: x[0])


# ============================================================================
# FIX #5: IMPROVED CLUSTERING WITH BETTER PARAMETERS
# ============================================================================
def cluster_time_stamps(offset_diff_counts, n_clusters=3, min_cluster_size=5):
    """
    Cluster time offsets to find the dominant alignment.
    
    Args:
        offset_diff_counts: Counter of time differences
        n_clusters: Number of clusters to try (increased from 1!)
        min_cluster_size: Minimum samples in a cluster to be valid
    
    Returns:
        best_count: Size of largest valid cluster
        best_offset: The offset corresponding to that cluster
    """
    if not offset_diff_counts:
        return 0, 0
    
    # Create weighted data
    raw_data = []
    offset_map = {}  # Map index back to offset value
    
    idx = 0
    for offset_diff, count in offset_diff_counts.items():
        offset_map[idx] = offset_diff
        raw_data.extend([idx] * count)
        idx += 1
    
    if not raw_data:
        return 0, 0
    
    data = np.array(raw_data).reshape(-1, 1)
    
    # Standardize
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    
    # Cluster
    n_clusters = min(n_clusters, len(np.unique(data)))
    if n_clusters < 1:
        n_clusters = 1
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, 
                            n_init='auto', batch_size=256)
    kmeans.fit(standardized_data)
    
    # Find largest cluster
    labels = kmeans.labels_
    label_counts = np.bincount(labels)
    best_label = np.argmax(label_counts)
    best_count = label_counts[best_label]
    
    # Get the corresponding offset
    cluster_mask = labels == best_label
    cluster_indices = np.where(cluster_mask)[0]
    # Most common original offset in this cluster
    original_offsets = [data[i, 0] for i in cluster_indices]
    best_offset = int(np.median(original_offsets))
    
    return best_count, best_offset


# ============================================================================
# FIX #6: MAIN FINGERPRINTING FUNCTION
# ============================================================================
def create_audio_fingerprint(audio_path):
    """
    Create audio fingerprint with all fixes applied.
    """
    # 1. Generate Spectrogram with standard parameters
    result = generate_spectrogram(audio_path, sr=STANDARD_SR, 
                                 n_fft=STANDARD_N_FFT, 
                                 hop_length=STANDARD_HOP_LENGTH)
    if result is None:
        return []
    
    spectrogram_db, sr, hop_length = result
    
    print(f"Spectrogram: {spectrogram_db.shape[1]} frames, {spectrogram_db.shape[0]} bins")
    
    # 2. Extract Peaks
    peaks = find_spectrogram_peaks(spectrogram_db)
    print(f"Extracted {len(peaks)} peaks")
    
    # 3. Generate Hashes (with improvements)
    fingerprint_hashes = generate_audio_hashes_improved(peaks, fan_value=FAN_VALUE)
    print(f"Generated {len(fingerprint_hashes)} hashes")
    
    return fingerprint_hashes


# ============================================================================
# DEBUG FUNCTION: Analyze fingerprint quality
# ============================================================================
def analyze_fingerprint_quality(hashes):
    """
    Diagnostic function to understand your fingerprints.
    """
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
    
    # Analyze time distribution
    times = [t for _, t in hashes]
    print(f"Time range: {min(times)} to {max(times)}")
    print(f"Average time between hashes: {(max(times) - min(times)) / len(times):.2f}")
