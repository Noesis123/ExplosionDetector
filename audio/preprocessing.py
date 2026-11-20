"""Unified Audio Preprocessing Pipeline for Acoustic Explosion Detection.

Implemented purely with PyTorch, NumPy, and SciPy (zero Librosa/Numba dependencies).
Shared by both recorded-audio file processing and live-microphone streams.
Provides:
- Channel conversion to mono float32 with peak clipping protection.
- High-fidelity anti-aliasing polyphase sinc resampling (scipy.signal.resample_poly).
- Sliding window segmentation with configurable duration, hop, and transient tail preservation.
- Full 2D Log-Mel spectrogram feature extraction (torch.stft Hann STFT + Slaney Mel filter bank + dB scale).
- Per-spectrogram zero-mean unit-variance normalization.
- Robust input validation, numerical guards against NaN/Inf, and detailed error messages.
"""

from typing import List, Optional, Tuple, Union
import math
import numpy as np
import scipy.signal as signal
import torch

_SLANEY_F_SP = 200.0 / 3.0
_SLANEY_MIN_LOG_HZ = 1000.0
_SLANEY_MIN_LOG_MEL = _SLANEY_MIN_LOG_HZ / _SLANEY_F_SP
_SLANEY_LOGSTEP = math.log(6.4) / 27.0


def _hz_to_mel_slaney(frequencies: np.ndarray) -> np.ndarray:
    """Convert frequencies in Hz to Slaney Mel scale (matches librosa htk=False)."""
    freqs = np.asanyarray(frequencies, dtype=np.float64)
    mels = freqs / _SLANEY_F_SP
    log_mask = freqs >= _SLANEY_MIN_LOG_HZ
    if np.any(log_mask):
        mels = np.where(
            log_mask,
            _SLANEY_MIN_LOG_MEL + np.log(np.maximum(freqs, _SLANEY_MIN_LOG_HZ) / _SLANEY_MIN_LOG_HZ) / _SLANEY_LOGSTEP,
            mels,
        )
    return mels


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    """Convert Slaney Mel frequencies back to Hz (matches librosa htk=False)."""
    mel_arr = np.asanyarray(mels, dtype=np.float64)
    freqs = mel_arr * _SLANEY_F_SP
    log_mask = mel_arr >= _SLANEY_MIN_LOG_MEL
    if np.any(log_mask):
        freqs = np.where(
            log_mask,
            _SLANEY_MIN_LOG_HZ * np.exp(_SLANEY_LOGSTEP * (mel_arr - _SLANEY_MIN_LOG_MEL)),
            freqs,
        )
    return freqs


def build_slaney_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Build area-normalized Slaney Mel filter bank matching librosa.filters.mel(..., htk=False, norm='slaney')."""
    weights = np.zeros((int(n_mels), 1 + int(n_fft) // 2), dtype=np.float64)
    fftfreqs = np.fft.rfftfreq(n=int(n_fft), d=1.0 / float(sample_rate))
    min_mel = float(_hz_to_mel_slaney(np.array([f_min]))[0])
    max_mel = float(_hz_to_mel_slaney(np.array([f_max]))[0])
    mel_f = _mel_to_hz_slaney(np.linspace(min_mel, max_mel, int(n_mels) + 2))

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(int(n_mels)):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_f[2 : int(n_mels) + 2] - mel_f[: int(n_mels)])
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32)


def power_to_db(power_spec: np.ndarray, amin: float = 1e-10, top_db: float = 80.0) -> np.ndarray:
    """Convert power spectrogram to decibels relative to peak (matches librosa.power_to_db(ref=np.max, top_db=80.0))."""
    mag = np.asarray(power_spec, dtype=np.float32)
    ref_val = max(float(amin), float(np.max(mag)))
    log_spec = 10.0 * np.log10(np.maximum(float(amin), mag)) - 10.0 * math.log10(ref_val)
    if top_db is not None:
        log_spec = np.maximum(log_spec, float(np.max(log_spec)) - float(top_db))
    return log_spec.astype(np.float32)


class AudioPreprocessor:
    """Configurable PyTorch/SciPy/NumPy audio preprocessor for explosion acoustic classification.

    Ensures numerical compatibility with legacy baseline checkpoints as well as
    consistency between offline dataset preparation, batch evaluation, and live streaming.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        window_duration_sec: float = 1.0,
        hop_duration_sec: float = 0.25,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        f_min: float = 20.0,
        f_max: Optional[float] = None,
        normalize_spectrogram: bool = True,
    ):
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if window_duration_sec <= 0:
            raise ValueError(f"window_duration_sec must be positive, got {window_duration_sec}")
        if hop_duration_sec <= 0:
            raise ValueError(f"hop_duration_sec must be positive, got {hop_duration_sec}")
        if n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {n_mels}")
        if n_fft <= 0 or (n_fft & (n_fft - 1) != 0):
            raise ValueError(f"n_fft must be a positive power of 2, got {n_fft}")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")

        self.sample_rate = int(sample_rate)
        self.window_duration_sec = float(window_duration_sec)
        self.hop_duration_sec = float(hop_duration_sec)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.f_min = float(f_min)
        self.f_max = float(f_max) if f_max is not None else float(self.sample_rate / 2.0)
        self.normalize_spectrogram = bool(normalize_spectrogram)

        self.window_samples = int(round(self.window_duration_sec * self.sample_rate))
        self.hop_samples = int(round(self.hop_duration_sec * self.sample_rate))

        # Centered STFT (pad n_fft // 2 on both sides) yields 1 + window_samples // hop_length frames (44)
        self.expected_time_frames = 1 + int(self.window_samples // self.hop_length)

        # Precompute Slaney Mel filter bank and Hann window tensor
        self.mel_basis = build_slaney_mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )
        self._mel_basis_tensor = torch.from_numpy(self.mel_basis)
        self._hann_window = torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32)

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert any audio array to a 1D float32 mono signal."""
        if audio is None:
            raise ValueError("Input audio array cannot be None")
        
        arr = np.asarray(audio, dtype=np.float32)
        
        if arr.size == 0:
            raise ValueError("Input audio array is empty (0 samples)")
            
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input audio contains NaN or Inf values")

        # Handle various channel layouts
        if arr.ndim == 1:
            mono = arr
        elif arr.ndim == 2:
            # (channels, samples) or (samples, channels)
            if arr.shape[0] in (1, 2, 4, 6, 8) and arr.shape[1] > arr.shape[0]:
                mono = np.mean(arr, axis=0)
            elif arr.shape[1] in (1, 2, 4, 6, 8) and arr.shape[0] >= arr.shape[1]:
                mono = np.mean(arr, axis=1)
            else:
                mono = np.mean(arr, axis=-1)
        else:
            raise ValueError(f"Unsupported audio array dimension: {arr.ndim} (expected 1D or 2D)")

        # Normalization guard against clipping / float range saturation
        peak = float(np.max(np.abs(mono))) if len(mono) > 0 else 0.0
        if peak > 1.0:
            mono = mono / peak

        return mono.astype(np.float32)

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to target sample rate using polyphase anti-aliasing sinc filter."""
        if orig_sr <= 0:
            raise ValueError(f"orig_sr must be positive, got {orig_sr}")
            
        if orig_sr == self.sample_rate:
            return audio.astype(np.float32)

        # Use polyphase filtering for fast, artifact-free anti-aliasing
        gcd = math.gcd(int(orig_sr), int(self.sample_rate))
        up = int(self.sample_rate // gcd)
        down = int(orig_sr // gcd)

        # Safety check on polyphase factor to avoid excessive memory on unusual sample rates
        if up > 1000 or down > 1000:
            target_samples = max(1, int(round(len(audio) * (self.sample_rate / float(orig_sr)))))
            resampled = signal.resample(audio, target_samples)
        else:
            resampled = signal.resample_poly(audio, up, down)

        return resampled.astype(np.float32)

    def pad_or_trim(self, audio: np.ndarray, target_length: Optional[int] = None) -> np.ndarray:
        """Pad audio with zeros or trim to exact target length."""
        if target_length is None:
            target_length = self.window_samples

        cur_len = len(audio)
        if cur_len == target_length:
            return audio
        elif cur_len > target_length:
            return audio[:target_length]
        else:
            padded = np.zeros(target_length, dtype=np.float32)
            padded[:cur_len] = audio
            return padded

    def create_sliding_windows(
        self,
        audio: np.ndarray,
        preserve_tail: bool = True
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Slice 1D audio into 2D sliding windows of shape (num_windows, window_samples).
        
        Args:
            audio: 1D float32 audio array at target sample rate.
            preserve_tail: If True, pads remaining samples at the end to ensure short
                           transients occurring near the end are not dropped.
                           
        Returns:
            Tuple of:
            - windows: 2D array of shape (num_windows, window_samples)
            - timestamps: List of (start_seconds, end_seconds) for each window.
        """
        total_samples = len(audio)
        
        if total_samples == 0:
            raise ValueError("Cannot window empty audio array")

        # If audio is shorter than a single window
        if total_samples < self.window_samples:
            if preserve_tail:
                padded = self.pad_or_trim(audio, self.window_samples)
                windows = padded[np.newaxis, :]
                timestamps = [(0.0, total_samples / self.sample_rate)]
                return windows, timestamps
            else:
                return np.empty((0, self.window_samples), dtype=np.float32), []

        # Standard sliding windows
        window_list = []
        timestamps = []
        
        start_idx = 0
        while start_idx + self.window_samples <= total_samples:
            end_idx = start_idx + self.window_samples
            window_list.append(audio[start_idx:end_idx])
            timestamps.append((start_idx / self.sample_rate, end_idx / self.sample_rate))
            start_idx += self.hop_samples

        # Handle remaining trailing transient if requested
        if preserve_tail and (start_idx < total_samples):
            trailing_audio = audio[start_idx:]
            # Only include if tail has meaningful content (>= 15% of hop size)
            if len(trailing_audio) >= int(self.hop_samples * 0.15):
                padded_tail = self.pad_or_trim(trailing_audio, self.window_samples)
                window_list.append(padded_tail)
                timestamps.append((start_idx / self.sample_rate, total_samples / self.sample_rate))

        windows = np.stack(window_list, axis=0).astype(np.float32)
        return windows, timestamps

    def compute_log_mel_spectrogram(self, audio_window: np.ndarray) -> np.ndarray:
        """Compute normalized Log-Mel spectrogram from a single 1.0s window.

        Output shape: (n_mels, expected_time_frames), e.g., (64, 44).
        """
        # Ensure exact window length
        fixed_audio = self.pad_or_trim(audio_window, self.window_samples)

        # Silence Guard: If audio window is confirmed silence (< 1e-5 peak, < -100 dBFS),
        # return a calibrated acoustic floor tensor that prevents all-zero neural bias saturation.
        if float(np.max(np.abs(fixed_audio))) < 1e-5:
            return np.full((self.n_mels, self.expected_time_frames), -2.0, dtype=np.float32)

        # PyTorch STFT Power Spectrogram (centered Hann window, constant zero-padding)
        wav_tensor = torch.from_numpy(np.ascontiguousarray(fixed_audio, dtype=np.float32))
        stft = torch.stft(
            wav_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self._hann_window,
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        power_spec = stft.real.pow(2) + stft.imag.pow(2)

        # Project power spectrum onto Slaney Mel filter bank and convert to top-80dB scale
        mel_spec = torch.matmul(self._mel_basis_tensor, power_spec).numpy()
        log_mel = power_to_db(mel_spec, amin=1e-10, top_db=80.0)

        # Guard exact time frame dimension
        if log_mel.shape[1] < self.expected_time_frames:
            pad_width = self.expected_time_frames - log_mel.shape[1]
            log_mel = np.pad(
                log_mel,
                ((0, 0), (0, pad_width)),
                mode="constant",
                constant_values=float(log_mel.min()),
            )
        elif log_mel.shape[1] > self.expected_time_frames:
            log_mel = log_mel[:, : self.expected_time_frames]

        # Per-spectrogram zero-mean unit-variance normalization
        if self.normalize_spectrogram:
            mean = float(np.mean(log_mel))
            std = float(np.std(log_mel)) + 1e-6
            normalized_mel = (log_mel - mean) / std
        else:
            normalized_mel = log_mel

        return normalized_mel.astype(np.float32)

    def process_file_audio(
        self,
        audio: np.ndarray,
        orig_sr: int
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """End-to-end pipeline for recorded audio: mono -> resample -> window -> 3D Mel stack.
        
        Returns:
            Tuple of:
            - specs: 3D array of shape (num_windows, n_mels, expected_time_frames)
            - timestamps: List of (start_sec, end_sec) for each window.
        """
        mono = self.to_mono(audio)
        resampled = self.resample(mono, orig_sr)
        windows, timestamps = self.create_sliding_windows(resampled, preserve_tail=True)
        
        if len(windows) == 0:
            return np.empty((0, self.n_mels, self.expected_time_frames), dtype=np.float32), []

        specs = np.zeros(
            (len(windows), self.n_mels, self.expected_time_frames),
            dtype=np.float32
        )
        for i in range(len(windows)):
            specs[i] = self.compute_log_mel_spectrogram(windows[i])

        return specs, timestamps

    def process_live_buffer(
        self,
        buffer_audio: np.ndarray,
        orig_sr: int
    ) -> np.ndarray:
        """Process a single real-time microphone buffer window into a single Log-Mel spectrogram.
        
        Returns:
            2D array of shape (n_mels, expected_time_frames), e.g. (64, 44).
        """
        mono = self.to_mono(buffer_audio)
        resampled = self.resample(mono, orig_sr)
        # Take the most recent window_samples
        if len(resampled) >= self.window_samples:
            window = resampled[-self.window_samples:]
        else:
            window = self.pad_or_trim(resampled, self.window_samples)
            
        return self.compute_log_mel_spectrogram(window)

