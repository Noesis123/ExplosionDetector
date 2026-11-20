"""Robust Audio File Loader for Acoustic Explosion Detection.

Handles WAV, MP3, FLAC, OGG, and OGA audio files via SoundFile (libsndfile)
with stdlib wave fallback for uncompressed PCM WAV, strict validation,
and descriptive error diagnostics without Librosa/Numba dependencies.
"""

from pathlib import Path
from typing import Dict, Tuple, Union
import wave
import numpy as np
import soundfile as sf

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".mp4")


class AudioLoadError(Exception):
    """Exception raised when an audio file cannot be loaded or validated."""
    pass


def _load_av(path: Path) -> Tuple[np.ndarray, int]:
    """Decode audio via PyAV (ffmpeg wrapper) for M4A, AAC, and container formats."""
    try:
        import av
    except ImportError as e:
        raise AudioLoadError(f"PyAV is required to decode {path.name}: {e}") from e

    try:
        container = av.open(str(path))
        if not container.streams.audio:
            container.close()
            raise AudioLoadError(f"No audio stream found in {path.name}")
        stream = container.streams.audio[0]
        frames = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray())
        container.close()
        if not frames:
            raise AudioLoadError(f"No audio frames decoded from {path.name}")
        audio = np.concatenate(frames, axis=-1)
        if audio.ndim == 2:
            audio = audio.T
            if audio.shape[1] == 1:
                audio = audio.squeeze(1)
        return audio.astype(np.float32), int(stream.rate)
    except Exception as e:
        raise AudioLoadError(f"PyAV failed to decode {path.name}: {e}") from e


def _load_pcm_wav_stdlib(path: Path) -> Tuple[np.ndarray, int]:
    """Decode uncompressed PCM WAV files with stdlib wave if soundfile fails."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise AudioLoadError(f"Unsupported PCM WAV sample width ({sampwidth} bytes): {path.name}")
    if n_channels > 1:
        data = data.reshape(-1, n_channels)
    return data, int(sr)


def load_audio_file(
    file_path: Union[str, Path],
    always_float32: bool = True
) -> Tuple[np.ndarray, int, float, int]:
    """Load an audio file from disk into a NumPy array with full metadata.

    Args:
        file_path: Path to the target audio file (.wav, .mp3, .flac, .ogg, .oga).
        always_float32: If True, normalizes and casts audio to np.float32 [-1.0, 1.0].

    Returns:
        Tuple of:
        - audio: np.ndarray of shape (samples,) or (samples, channels)
        - sample_rate: int native sampling rate
        - duration_seconds: float audio duration in seconds
        - channels: int number of audio channels

    Raises:
        AudioLoadError: If file is missing, empty, unsupported, unreadable, or corrupted.
    """
    path = Path(file_path)

    if not path.exists():
        raise AudioLoadError(f"Audio file does not exist: {path}")

    if not path.is_file():
        raise AudioLoadError(f"Specified path is not a file: {path}")

    if path.stat().st_size == 0:
        raise AudioLoadError(f"Audio file is completely empty (0 bytes): {path}")

    read_dtype = "float32" if always_float32 else "float64"
    try:
        audio, sr = sf.read(str(path), dtype=read_dtype, always_2d=False)
    except Exception as sf_err:
        if path.suffix.lower() == ".wav":
            try:
                audio, sr = _load_pcm_wav_stdlib(path)
            except Exception as wav_err:
                raise AudioLoadError(
                    f"Failed to decode WAV audio file {path.name}. "
                    f"soundfile error: '{sf_err}', stdlib wave error: '{wav_err}'"
                ) from wav_err
        elif path.suffix.lower() in (".m4a", ".aac", ".mp4"):
            try:
                audio, sr = _load_av(path)
            except Exception as av_err:
                raise AudioLoadError(
                    f"Failed to decode {path.name} via PyAV: {av_err}"
                ) from av_err
        else:
            try:
                audio, sr = _load_av(path)
            except Exception:
                raise AudioLoadError(
                    f"Failed to decode audio file {path.name} (supported via SoundFile/libsndfile: "
                    f"{', '.join(SUPPORTED_AUDIO_EXTENSIONS)}). soundfile error: '{sf_err}'"
                ) from sf_err

    if audio is None or len(audio) == 0:
        raise AudioLoadError(f"Decoded audio buffer is empty (0 samples): {path}")

    if not np.all(np.isfinite(audio)):
        raise AudioLoadError(f"Audio file contains invalid NaN or Inf values: {path}")

    channels = 1 if audio.ndim == 1 else audio.shape[1]
    duration_seconds = len(audio) / float(sr)

    # Normalize amplitude to [-1.0, 1.0] if float32 requested
    if always_float32 and audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    return audio, int(sr), float(duration_seconds), int(channels)

def validate_audio_file(file_path: Union[str, Path]) -> Dict[str, Union[bool, str, int, float]]:
    """Inspect and validate an audio file without throwing exceptions.
    
    Returns a dictionary report:
    {'valid': bool, 'error': Optional[str], 'sample_rate': int, 'duration': float, 'channels': int}
    """
    try:
        audio, sr, dur, ch = load_audio_file(file_path)
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))
        return {
            "valid": True,
            "error": None,
            "sample_rate": sr,
            "duration": dur,
            "channels": ch,
            "peak": round(peak, 4),
            "rms": round(rms, 6),
            "is_silent": bool(rms < 0.0001)
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "sample_rate": None,
            "duration": None,
            "channels": None,
            "peak": None,
            "rms": None,
            "is_silent": None
        }

