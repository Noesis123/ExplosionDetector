"""Stateful Streaming Resampler and Robust Live Audio Buffer for Real-Time Detection.

Provides:
1. Continuous anti-aliased streaming resampling from arbitrary hardware sample rates (48kHz, 44.1kHz, etc.) to 22,050 Hz.
2. Bounded circular buffer for sliding-window inference with overrun protection.
"""

from typing import List, Tuple
import numpy as np
import scipy.signal as sps


class StatefulStreamingResampler:
    """Continuous stateful streaming resampler with anti-aliasing filtering.
    Converts live audio streams from native_sr to target_sr across arbitrary
    chunk boundaries without boundary clicks or accumulated phase drift.
    """

    def __init__(self, in_sr: int, out_sr: int = 22050):
        self.in_sr = in_sr
        self.out_sr = out_sr
        self.ratio = in_sr / out_sr
        self.passthrough = (in_sr == out_sr)

        # Anti-aliasing FIR filter if downsampling
        if in_sr > out_sr:
            f_cutoff = (out_sr * 0.45) / (in_sr * 0.5)
            self.b = sps.firwin(63, f_cutoff)
            self.a = 1.0
            self.zi = sps.lfilter_zi(self.b, self.a) * 0.0
        else:
            self.b = None
            self.a = None
            self.zi = None

        self.buf = np.zeros(0, dtype=np.float32)
        self.idx = 0.0

    def reset(self):
        if self.b is not None:
            self.zi = sps.lfilter_zi(self.b, self.a) * 0.0
        self.buf = np.zeros(0, dtype=np.float32)
        self.idx = 0.0

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        if self.passthrough:
            return chunk.astype(np.float32)

        if len(chunk) == 0:
            return np.zeros(0, dtype=np.float32)

        # Anti-alias filter in input domain
        if self.b is not None:
            chunk, self.zi = sps.lfilter(self.b, self.a, chunk, zi=self.zi)

        full = np.concatenate((self.buf, chunk))
        out = []
        n_full = len(full)

        while self.idx + 2 < n_full:
            i0 = int(self.idx)
            t = self.idx - i0
            # 4 points for Hermite / Catmull-Rom cubic interpolation: p0, p1, p2, p3
            p0 = full[max(0, i0 - 1)]
            p1 = full[i0]
            p2 = full[min(n_full - 1, i0 + 1)]
            p3 = full[min(n_full - 1, i0 + 2)]

            val = p1 + 0.5 * t * (
                p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * p1 - p0 - 3.0 * p2 + p3))
            )
            out.append(val)
            self.idx += self.ratio

        consumed = max(0, int(self.idx) - 2)
        self.buf = full[consumed:]
        self.idx -= consumed
        return np.array(out, dtype=np.float32)

    def flush(self) -> np.ndarray:
        if self.passthrough or len(self.buf) == 0:
            return np.zeros(0, dtype=np.float32)
        out = []
        n_full = len(self.buf)
        while self.idx < n_full:
            i0 = int(self.idx)
            t = self.idx - i0
            p0 = self.buf[max(0, i0 - 1)]
            p1 = self.buf[min(n_full - 1, i0)]
            p2 = self.buf[min(n_full - 1, i0 + 1)]
            p3 = self.buf[min(n_full - 1, i0 + 2)]
            val = p1 + 0.5 * t * (
                p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * p1 - p0 - 3.0 * p2 + p3))
            )
            out.append(val)
            self.idx += self.ratio
        self.reset()
        return np.array(out, dtype=np.float32)


class RobustLiveAudioBuffer:
    """Bounded, thread-safe circular sample buffer.
    Accumulates 22,050 Hz samples and emits 1.0s windows every 250 ms hop (5,512 samples).
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        window_duration_sec: float = 1.0,
        hop_duration_sec: float = 0.25,
        max_buffer_sec: float = 8.0,
    ):
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_duration_sec)
        self.hop_samples = int(sample_rate * hop_duration_sec)
        self.max_buffer_samples = int(sample_rate * max_buffer_sec)

        self.sample_buffer = np.zeros(0, dtype=np.float32)
        self.total_samples_received = 0
        self.total_windows_produced = 0
        self.dropped_frames = 0
        self.buffer_overruns = 0

    def reset(self):
        self.sample_buffer = np.zeros(0, dtype=np.float32)
        self.total_samples_received = 0
        self.total_windows_produced = 0
        self.dropped_frames = 0
        self.buffer_overruns = 0

    def push_samples(self, samples: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        if len(samples) == 0:
            return []

        # Check buffer overflow
        if len(self.sample_buffer) + len(samples) > self.max_buffer_samples:
            overflow = (len(self.sample_buffer) + len(samples)) - self.max_buffer_samples
            self.sample_buffer = self.sample_buffer[overflow:]
            self.dropped_frames += overflow
            self.buffer_overruns += 1

        self.sample_buffer = np.concatenate((self.sample_buffer, samples))
        self.total_samples_received += len(samples)

        windows = []
        while len(self.sample_buffer) >= self.window_samples:
            win = self.sample_buffer[: self.window_samples].copy()
            win_time_sec = (self.total_samples_received - len(self.sample_buffer) + self.window_samples) / float(self.sample_rate)
            windows.append((round(win_time_sec, 4), win))
            self.total_windows_produced += 1
            self.sample_buffer = self.sample_buffer[self.hop_samples :]

        return windows
