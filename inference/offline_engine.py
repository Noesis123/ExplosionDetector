"""Offline Recorded-Audio File and Folder Inference Engine.

Processes single files or batch directories (with recursive subfolder support),
running the shared preprocessing pipeline, generating per-window timestamps and scores,
and producing aggregated recording-level results.

Supports CSV and JSON export without modifying source files or dataset manifests.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import csv
import json
import time
import numpy as np

from audio.file_loader import load_audio_file, AudioLoadError
from audio.preprocessing import AudioPreprocessor
from inference.predictor_interface import BasePredictor, MockTestPredictor

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".oga"}

@dataclass
class WindowResult:
    """Individual window detection prediction and timestamp."""
    window_index: int
    start_time_s: float
    end_time_s: float
    explosion_score: float
    window_prediction: str

@dataclass
class FileInferenceResult:
    """Complete inference output for a single audio file."""
    file_path: str
    file_name: str
    file_size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    num_windows: int
    max_explosion_score: float
    mean_explosion_score: float
    positive_windows_count: int
    aggregated_prediction: str
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None
    is_mock_predictor: bool = False
    predictor_name: str = "Unknown"
    windows: List[WindowResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class BatchInferenceResult:
    """Aggregated results across a directory or batch of files."""
    input_path: str
    total_files_discovered: int
    total_files_processed: int
    total_files_failed: int
    total_audio_duration_s: float
    total_windows_processed: int
    total_processing_time_ms: float
    threshold_used: float
    aggregation_mode: str
    predictor_name: str
    is_mock_predictor: bool
    results: List[FileInferenceResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class OfflineInferenceEngine:
    """Offline inference processor for individual audio files and batch folders."""

    def __init__(
        self,
        preprocessor: Optional[AudioPreprocessor] = None,
        predictor: Optional[BasePredictor] = None,
        threshold: float = 0.80,
        aggregation_mode: str = "peak_window",
        min_consecutive_windows: int = 2
    ):
        """
        Args:
            preprocessor: AudioPreprocessor instance (or default baseline if None).
            predictor: BasePredictor instance (defaults to MockTestPredictor for pipeline testing).
            threshold: Confidence threshold for explosion detection (default 0.80).
            aggregation_mode: Strategy to aggregate window scores into recording verdict:
                              - 'peak_window': Positive if max_score >= threshold.
                              - 'consecutive': Positive if >= min_consecutive_windows exceed threshold.
            min_consecutive_windows: Required consecutive positive windows for 'consecutive' mode.
        """
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.predictor = predictor or MockTestPredictor()
        self.threshold = float(threshold)
        self.aggregation_mode = aggregation_mode
        self.min_consecutive_windows = int(min_consecutive_windows)

    def infer_file(self, file_path: Union[str, Path]) -> FileInferenceResult:
        """Run complete preprocessing and inference on an individual audio file."""
        path = Path(file_path)
        t_start = time.perf_counter()

        # 1. Validation checks
        if not path.exists():
            return FileInferenceResult(
                file_path=str(path),
                file_name=path.name,
                file_size_bytes=0,
                duration_seconds=0.0,
                sample_rate=0,
                channels=0,
                num_windows=0,
                max_explosion_score=0.0,
                mean_explosion_score=0.0,
                positive_windows_count=0,
                aggregated_prediction="non_explosion",
                processing_time_ms=0.0,
                success=False,
                error_message=f"File not found: {path}",
                is_mock_predictor=self.predictor.is_mock(),
                predictor_name=self.predictor.name,
            )

        if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            return FileInferenceResult(
                file_path=str(path),
                file_name=path.name,
                file_size_bytes=path.stat().st_size,
                duration_seconds=0.0,
                sample_rate=0,
                channels=0,
                num_windows=0,
                max_explosion_score=0.0,
                mean_explosion_score=0.0,
                positive_windows_count=0,
                aggregated_prediction="non_explosion",
                processing_time_ms=0.0,
                success=False,
                error_message=f"Unsupported audio format: {path.suffix} (supported: {SUPPORTED_AUDIO_EXTENSIONS})",
                is_mock_predictor=self.predictor.is_mock(),
                predictor_name=self.predictor.name,
            )

        file_size = path.stat().st_size

        # 2. Audio loading & decoding
        try:
            audio, sr, dur, ch = load_audio_file(path)
        except Exception as e:
            return FileInferenceResult(
                file_path=str(path),
                file_name=path.name,
                file_size_bytes=file_size,
                duration_seconds=0.0,
                sample_rate=0,
                channels=0,
                num_windows=0,
                max_explosion_score=0.0,
                mean_explosion_score=0.0,
                positive_windows_count=0,
                aggregated_prediction="non_explosion",
                processing_time_ms=(time.perf_counter() - t_start) * 1000,
                success=False,
                error_message=f"Audio load/decode error: {e}",
                is_mock_predictor=self.predictor.is_mock(),
                predictor_name=self.predictor.name,
            )

        # 3. Preprocessing (Mono -> Resample -> Windows -> Log-Mel)
        try:
            specs, timestamps = self.preprocessor.process_file_audio(audio, sr)
        except Exception as e:
            return FileInferenceResult(
                file_path=str(path),
                file_name=path.name,
                file_size_bytes=file_size,
                duration_seconds=dur,
                sample_rate=sr,
                channels=ch,
                num_windows=0,
                max_explosion_score=0.0,
                mean_explosion_score=0.0,
                positive_windows_count=0,
                aggregated_prediction="non_explosion",
                processing_time_ms=(time.perf_counter() - t_start) * 1000,
                success=False,
                error_message=f"Preprocessing error: {e}",
                is_mock_predictor=self.predictor.is_mock(),
                predictor_name=self.predictor.name,
            )

        num_windows = len(specs)
        if num_windows == 0:
            return FileInferenceResult(
                file_path=str(path),
                file_name=path.name,
                file_size_bytes=file_size,
                duration_seconds=dur,
                sample_rate=sr,
                channels=ch,
                num_windows=0,
                max_explosion_score=0.0,
                mean_explosion_score=0.0,
                positive_windows_count=0,
                aggregated_prediction="non_explosion",
                processing_time_ms=(time.perf_counter() - t_start) * 1000,
                success=True,
                error_message="Audio shorter than hop threshold; no windows extracted",
                is_mock_predictor=self.predictor.is_mock(),
                predictor_name=self.predictor.name,
            )

        # 4. Predictor inference
        try:
            probs = self.predictor.predict_batch(specs)
            # Explosion probability is index 1
            exp_scores = probs[:, 1]
        except Exception as e:
            return FileInferenceResult(
                file_path=str(path),
                file_name=path.name,
                file_size_bytes=file_size,
                duration_seconds=dur,
                sample_rate=sr,
                channels=ch,
                num_windows=num_windows,
                max_explosion_score=0.0,
                mean_explosion_score=0.0,
                positive_windows_count=0,
                aggregated_prediction="non_explosion",
                processing_time_ms=(time.perf_counter() - t_start) * 1000,
                success=False,
                error_message=f"Model prediction error: {e}",
                is_mock_predictor=self.predictor.is_mock(),
                predictor_name=self.predictor.name,
            )

        # 5. Build per-window results
        window_results = []
        positive_count = 0
        consecutive_count = 0
        max_consecutive = 0

        for i in range(num_windows):
            score = float(exp_scores[i])
            t0, t1 = timestamps[i]
            is_win_pos = score >= self.threshold
            win_pred = "explosion" if is_win_pos else "non_explosion"

            if is_win_pos:
                positive_count += 1
                consecutive_count += 1
                if consecutive_count > max_consecutive:
                    max_consecutive = consecutive_count
            else:
                consecutive_count = 0

            window_results.append(WindowResult(
                window_index=i,
                start_time_s=round(t0, 3),
                end_time_s=round(t1, 3),
                explosion_score=round(score, 4),
                window_prediction=win_pred
            ))

        max_score = float(np.max(exp_scores)) if num_windows > 0 else 0.0
        mean_score = float(np.mean(exp_scores)) if num_windows > 0 else 0.0

        # 6. Aggregated recording-level decision
        if self.aggregation_mode == "consecutive":
            is_rec_pos = max_consecutive >= self.min_consecutive_windows
        else: # 'peak_window' (default)
            is_rec_pos = max_score >= self.threshold

        agg_pred = "explosion" if is_rec_pos else "non_explosion"
        proc_time_ms = (time.perf_counter() - t_start) * 1000.0

        return FileInferenceResult(
            file_path=str(path.resolve()),
            file_name=path.name,
            file_size_bytes=file_size,
            duration_seconds=round(dur, 3),
            sample_rate=sr,
            channels=ch,
            num_windows=num_windows,
            max_explosion_score=round(max_score, 4),
            mean_explosion_score=round(mean_score, 4),
            positive_windows_count=positive_count,
            aggregated_prediction=agg_pred,
            processing_time_ms=round(proc_time_ms, 2),
            success=True,
            error_message=None,
            is_mock_predictor=self.predictor.is_mock(),
            predictor_name=self.predictor.name,
            windows=window_results
        )

    def infer_folder(
        self,
        folder_path: Union[str, Path],
        recursive: bool = True,
        output_dir: Optional[Union[str, Path]] = None
    ) -> BatchInferenceResult:
        """Run batch inference across all supported audio files in a folder."""
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found or not a directory: {folder}")

        # Discover audio files
        if recursive:
            discovered = [
                f for f in sorted(folder.rglob("*"))
                if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
            ]
        else:
            discovered = [
                f for f in sorted(folder.glob("*"))
                if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
            ]

        t0_total = time.perf_counter()
        results = []
        total_duration = 0.0
        total_windows = 0
        failed_count = 0

        for file_p in discovered:
            res = self.infer_file(file_p)
            results.append(res)
            if res.success:
                total_duration += res.duration_seconds
                total_windows += res.num_windows
            else:
                failed_count += 1

        total_time_ms = (time.perf_counter() - t0_total) * 1000.0

        batch_result = BatchInferenceResult(
            input_path=str(folder.resolve()),
            total_files_discovered=len(discovered),
            total_files_processed=len(results),
            total_files_failed=failed_count,
            total_audio_duration_s=round(total_duration, 2),
            total_windows_processed=total_windows,
            total_processing_time_ms=round(total_time_ms, 2),
            threshold_used=self.threshold,
            aggregation_mode=self.aggregation_mode,
            predictor_name=self.predictor.name,
            is_mock_predictor=self.predictor.is_mock(),
            results=results
        )

        if output_dir is not None:
            self.export_results(batch_result, output_dir)

        return batch_result

    @staticmethod
    def export_results(
        batch_result: BatchInferenceResult,
        output_dir: Union[str, Path]
    ) -> Tuple[Path, Path]:
        """Export batch inference results to CSV and JSON files."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "batch_inference_summary.csv"
        json_path = out_dir / "batch_inference_details.json"

        # 1. Export CSV Summary
        csv_fields = [
            "file_name",
            "file_path",
            "file_size_bytes",
            "duration_seconds",
            "sample_rate",
            "channels",
            "num_windows",
            "max_explosion_score",
            "mean_explosion_score",
            "positive_windows_count",
            "aggregated_prediction",
            "processing_time_ms",
            "success",
            "error_message",
            "is_mock_predictor",
            "predictor_name"
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=csv_fields)
            writer.writeheader()
            for r in batch_result.results:
                writer.writerow({
                    "file_name": r.file_name,
                    "file_path": r.file_path,
                    "file_size_bytes": r.file_size_bytes,
                    "duration_seconds": r.duration_seconds,
                    "sample_rate": r.sample_rate,
                    "channels": r.channels,
                    "num_windows": r.num_windows,
                    "max_explosion_score": r.max_explosion_score,
                    "mean_explosion_score": r.mean_explosion_score,
                    "positive_windows_count": r.positive_windows_count,
                    "aggregated_prediction": r.aggregated_prediction,
                    "processing_time_ms": r.processing_time_ms,
                    "success": r.success,
                    "error_message": r.error_message or "",
                    "is_mock_predictor": r.is_mock_predictor,
                    "predictor_name": r.predictor_name
                })

        # 2. Export Full Detailed JSON (including per-window timelines)
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(batch_result.to_dict(), fp, indent=2)

        return csv_path, json_path

