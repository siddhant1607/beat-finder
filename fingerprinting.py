import librosa
import numpy as np
import scipy.signal

# --- Signal Processing Utilities ---
def lowpass_filter(y, sr, cutoff=5000.0):
    nyq = sr / 2
    b, a = scipy.signal.butter(1, cutoff / nyq, btype='low')
    return scipy.signal.lfilter(b, a, y)

def process_segment(y, sr, n_fft=1024, hop_length=None):
    y = lowpass_filter(y, sr, cutoff=5000)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hamming')
    S_mag = np.abs(S)
    return S_mag

def extract_peaks_bandwise(S_mag, bands=[(0,10),(10,20),(20,40),(40,80),(80,160),(160,512)]):
    num_frames = S_mag.shape[1]
    peaks = []
    for t in range(num_frames):
        mags = S_mag[:, t]
        for band in bands:
            band_mags = mags[band[0]:band[1]]
            if len(band_mags) == 0:
                continue
            max_idx = np.argmax(band_mags)
            max_mag = band_mags[max_idx]
            avg_mag = np.mean(band_mags)
            if max_mag > avg_mag:
                f_idx = band[0] + max_idx
                peaks.append((t, f_idx))
    return peaks

def create_address(anchor_f, target_f, delta_t):
    return (anchor_f << 23) | (target_f << 14) | delta_t

def generate_pair_hashes(peaks, fan_value=5):
    hashes = []
    for i, (t1, f1) in enumerate(peaks):
        for j in range(1, fan_value + 1):
            if i + j < len(peaks):
                t2, f2 = peaks[i + j]
                delta_t = t2 - t1
                if 0 <= delta_t < (1 << 14):
                    h = create_address(f1, f2, delta_t)
                    hashes.append((h, t1))
    return hashes
