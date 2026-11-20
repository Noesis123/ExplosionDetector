"""Pure-NumPy/SciPy acoustic similarity and duplicate detection utilities.

Provides waveform cross-correlation and cross-split near-duplicate auditing
without requiring PyTorch or Numba at import or execution time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
from scipy import signal


@dataclass(frozen=True)
class WaveformSimilarityResult:
    """Detailed waveform and metadata comparison between two audio files."""

    file_a: str
    file_b: str
    sample_rate_a: int
    sample_rate_b: int
    channels_a: int
    channels_b: int
    duration_a: float
    duration_b: float
    sha256_a: str
    sha256_b: str
    peak_normalized_xcorr: float
    lag_samples: int
    lag_seconds: float
    aligned_pearson_r: float
    aligned_snr_db: float
    stereo_pearson_left: float
    stereo_pearson_right: float
    welch_psd_cosine: float


def _load_mono_and_raw(path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    mono = arr.mean(axis=1) if arr.ndim > 1 else arr
    return mono, arr, int(sr)


def compute_waveform_similarity(
    path_a: Union[str, Path],
    path_b: Union[str, Path],
    target_sr: int = 16000,
) -> WaveformSimilarityResult:
    """Compare two recordings at native/resampled rate using FFT cross-correlation."""
    pa = Path(path_a)
    pb = Path(path_b)
    ma, ra, sra = _load_mono_and_raw(pa)
    mb, rb, srb = _load_mono_and_raw(pb)

    cha = 1 if ra.ndim == 1 else int(ra.shape[1])
    chb = 1 if rb.ndim == 1 else int(rb.shape[1])
    dura = len(ma) / float(sra)
    durb = len(mb) / float(srb)

    # Compare at common native rate in float64 precision when identical, else resample to target_sr
    eval_sr = sra if sra == srb else target_sr
    wa = (ma if sra == eval_sr else signal.resample_poly(ma, eval_sr, sra)).astype(np.float64)
    wb = (mb if srb == eval_sr else signal.resample_poly(mb, eval_sr, srb)).astype(np.float64)

    norm_a = float(np.linalg.norm(wa) + 1e-12)
    norm_b = float(np.linalg.norm(wb) + 1e-12)
    corr = signal.correlate(wa / norm_a, wb / norm_b, mode="full", method="fft")
    idx = int(np.argmax(np.abs(corr)))
    peak_xcorr = float(min(1.0, np.abs(corr[idx])))
    lag = idx - (len(wb) - 1)
    lag_sec = lag / float(eval_sr)

    if lag >= 0:
        overlap = min(len(wa) - lag, len(wb))
        aa = wa[lag : lag + overlap]
        bb = wb[:overlap]
    else:
        overlap = min(len(wb) + lag, len(wa))
        bb = wb[-lag : -lag + overlap]
        aa = wa[:overlap]

    if len(aa) >= 2:
        aligned_r = float(np.corrcoef(aa, bb)[0, 1])
        sig_rms = float(np.sqrt(np.mean(aa ** 2)))
        diff_rms = float(np.sqrt(np.mean((aa - bb) ** 2)))
        snr_db = float(20.0 * np.log10((sig_rms + 1e-12) / (diff_rms + 1e-12)))
    else:
        aligned_r = 0.0
        snr_db = 0.0

    ch_l = aligned_r
    ch_r = aligned_r
    if sra == srb and ra.ndim > 1 and rb.ndim > 1 and ra.shape[1] >= 2 and rb.shape[1] >= 2 and overlap >= 2:
        if lag >= 0:
            al = ra[lag : lag + overlap, 0]
            bl = rb[:overlap, 0]
            ar = ra[lag : lag + overlap, 1]
            br = rb[:overlap, 1]
        else:
            bl = rb[-lag : -lag + overlap, 0]
            al = ra[:overlap, 0]
            br = rb[-lag : -lag + overlap, 1]
            ar = ra[:overlap, 1]
        ch_l = float(np.corrcoef(al, bl)[0, 1])
        ch_r = float(np.corrcoef(ar, br)[0, 1])

    _, pxx_a = signal.welch(wa, fs=eval_sr, nperseg=max(min(len(wa), 4096), 64))
    _, pxx_b = signal.welch(wb, fs=eval_sr, nperseg=max(min(len(wb), 4096), 64))
    bands_a = np.array([np.mean(b) for b in np.array_split(np.log1p(pxx_a / (np.sum(pxx_a) + 1e-12)), 32)])
    bands_b = np.array([np.mean(b) for b in np.array_split(np.log1p(pxx_b / (np.sum(pxx_b) + 1e-12)), 32)])
    psd_cos = float(
        np.dot(bands_a, bands_b) / ((np.linalg.norm(bands_a) * np.linalg.norm(bands_b)) + 1e-12)
    )

    return WaveformSimilarityResult(
        file_a=pa.name,
        file_b=pb.name,
        sample_rate_a=sra,
        sample_rate_b=srb,
        channels_a=cha,
        channels_b=chb,
        duration_a=dura,
        duration_b=durb,
        sha256_a=hashlib.sha256(pa.read_bytes()).hexdigest(),
        sha256_b=hashlib.sha256(pb.read_bytes()).hexdigest(),
        peak_normalized_xcorr=peak_xcorr,
        lag_samples=lag,
        lag_seconds=lag_sec,
        aligned_pearson_r=aligned_r,
        aligned_snr_db=snr_db,
        stereo_pearson_left=ch_l,
        stereo_pearson_right=ch_r,
        welch_psd_cosine=psd_cos,
    )


def find_cross_split_acoustic_duplicates(
    records: Sequence[Mapping[str, str]],
    path_key: str = "resolved_file_path",
    coarse_sr: int = 1000,
    coarse_threshold: float = 0.45,
    xcorr_threshold: float = 0.85,
) -> List[Dict[str, object]]:
    """Scan resolved manifest records for acoustic duplicates across different splits."""
    low_waves: List[np.ndarray] = []
    for rec in records:
        mono, _, sr = _load_mono_and_raw(Path(rec[path_key]))
        dur = len(mono) / float(sr)
        target_len = max(int(round(dur * coarse_sr)), 64)
        w = signal.resample(mono, target_len).astype(np.float32)
        w /= float(np.linalg.norm(w) + 1e-12)
        low_waves.append(w)

    findings: List[Dict[str, object]] = []
    n = len(records)
    for i in range(n):
        ri = records[i]
        wi = low_waves[i]
        for j in range(i + 1, n):
            rj = records[j]
            if ri.get("split") == rj.get("split"):
                continue
            xc_coarse = float(np.max(np.abs(signal.correlate(wi, low_waves[j], mode="full", method="fft"))))
            if xc_coarse < coarse_threshold:
                continue
            sim = compute_waveform_similarity(ri[path_key], rj[path_key])
            if sim.peak_normalized_xcorr >= xcorr_threshold:
                findings.append({
                    "file_a": sim.file_a,
                    "file_b": sim.file_b,
                    "split_a": ri.get("split", ""),
                    "split_b": rj.get("split", ""),
                    "group_a": ri.get("group_id", ""),
                    "group_b": rj.get("group_id", ""),
                    "peak_normalized_xcorr": sim.peak_normalized_xcorr,
                    "aligned_pearson_r": sim.aligned_pearson_r,
                    "aligned_snr_db": sim.aligned_snr_db,
                    "lag_seconds": sim.lag_seconds,
                })
    findings.sort(key=lambda item: float(item["peak_normalized_xcorr"]), reverse=True)
    return findings

