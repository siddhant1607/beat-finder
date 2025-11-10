import numpy as np
import scipy.signal
import librosa

def lowpass_filter(y, sr, cutoff=5000.0):
    nyq = sr / 2
    b, a = scipy.signal.butter(1, cutoff / nyq, btype='low')
    return scipy.signal.lfilter(b, a, y)

def process_segment(y, sr, n_fft=1024, hop_length=None):
    y = lowpass_filter(y, sr, cutoff=5000)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hamming')
    S_mag = np.abs(S)
    return S_mag

def extract_peaks_bandwise(S_mag, sr, bands=[(0,10),(10,20),(20,40),(40,80),(80,160),(160,512)]):
    num_frames = S_mag.shape[1]
    num_bins = S_mag.shape[0]
    peaks = []
    # Compute chroma for the whole segment
    chroma = librosa.feature.chroma_stft(S=S_mag, sr=sr, n_chroma=12)
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
                # Calculate which chroma bin this frequency bin maps to
                chroma_bin = np.argmax(chroma[:, t])
                peaks.append((t, chroma_bin))
    return peaks

def create_address(anchor_c, target_c, delta_t):
    # Now use chroma bin instead of frequency bin
    return (anchor_c << 16) | (target_c << 8) | delta_t

def generate_pair_hashes(peaks, fan_value=5):
    hashes = []
    delta_ts = [peaks[i+1][0] - peaks[i][0] for i in range(len(peaks)-1)]
    # Estimate mean delta_t (for tempo normalization), or set empirically
    mean_delta_t = np.mean(delta_ts) if len(delta_ts) > 0 else 1.0
    for i, (t1, c1) in enumerate(peaks):
        for j in range(1, fan_value + 1):
            if i + j < len(peaks):
                t2, c2 = peaks[i + j]
                delta_t = t2 - t1
                norm_delta_t = int((delta_t / mean_delta_t) * 100)
                if 0 <= norm_delta_t < 256:  # fit in 8 bits
                    h = create_address(c1, c2, norm_delta_t)
                    hashes.append((h, t1))
    return hashes
