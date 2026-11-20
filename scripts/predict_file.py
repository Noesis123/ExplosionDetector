"""CLI Script for Offline Recorded-Audio Explosion Detection Inference.

Runs offline inference on an individual audio file or batch folder.
Accepts .wav, .mp3, .flac, and .ogg files.
Exports results to CSV and JSON.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.preprocessing import AudioPreprocessor
from inference.predictor_interface import MockTestPredictor, PyTorchModelPredictor
from inference.offline_engine import OfflineInferenceEngine

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run offline acoustic explosion detection on audio files or directories."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an audio file or directory containing audio recordings."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save batch CSV and JSON results (default: output/inference_<timestamp>)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Confidence threshold for explosion classification (default: 0.80)."
    )
    parser.add_argument(
        "--aggregation-mode",
        type=str,
        choices=["peak_window", "consecutive"],
        default="peak_window",
        help="Strategy to aggregate window scores into recording decision (default: peak_window)."
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to trained PyTorch model checkpoint. If omitted, uses deterministic MockTestPredictor."
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Disable recursive subfolder search when input is a directory."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Initialize predictor
    if args.model_checkpoint:
        try:
            predictor = PyTorchModelPredictor(args.model_checkpoint)
            is_mock = False
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")
            sys.exit(1)
    else:
        predictor = MockTestPredictor(mode="deterministic_hash")
        is_mock = True

    preprocessor = AudioPreprocessor()
    engine = OfflineInferenceEngine(
        preprocessor=preprocessor,
        predictor=predictor,
        threshold=args.threshold,
        aggregation_mode=args.aggregation_mode
    )

    print("=" * 80)
    print("      OFFLINE ACOUSTIC EXPLOSION DETECTION INFERENCE ENGINE")
    print("=" * 80)
    print(f" • Input Path        : {input_path}")
    print(f" • Predictor Engine  : {predictor.name}")
    if is_mock:
        print("   >>> WARNING: RUNNING WITH MOCK TEST PREDICTOR (PIPELINE TEST ONLY) <<<")
        print("   >>> NO TRAINED MODEL EXISTS YET. PREDICTIONS ARE TEST FIXTURES. <<<")
    print(f" • Detection Thresh  : {args.threshold * 100:.1f}%")
    print(f" • Aggregation Mode  : {args.aggregation_mode}")
    print("=" * 80)

    if input_path.is_file():
        res = engine.infer_file(input_path)
        print("\n--- INFERENCE RESULT ---")
        print(f"File Name           : {res.file_name}")
        print(f"Status              : {'SUCCESS' if res.success else 'FAILED'}")
        if not res.success:
            print(f"Error               : {res.error_message}")
            sys.exit(1)

        print(f"Duration            : {res.duration_seconds:.2f}s | Sample Rate: {res.sample_rate} Hz")
        print(f"Windows Processed   : {res.num_windows}")
        print(f"Peak Explosion Score: {res.max_explosion_score * 100:.1f}%")
        print(f"Mean Explosion Score: {res.mean_explosion_score * 100:.1f}%")
        print(f"Positive Windows    : {res.positive_windows_count}/{res.num_windows}")
        print(f"Aggregated Decision : {'EXPLOSION' if res.aggregated_prediction == 'explosion' else 'NON_EXPLOSION'}")
        print(f"Processing Latency  : {res.processing_time_ms:.1f} ms")

        # Show first 5 windows
        print("\nWindow Timeline Preview (first 5 windows):")
        for w in res.windows[:5]:
            print(f"  [{w.start_time_s:5.2f}s - {w.end_time_s:5.2f}s] Score: {w.explosion_score*100:5.1f}% -> {w.window_prediction.upper()}")

        if args.output_dir:
            out_p = Path(args.output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            single_json = out_p / f"{input_path.stem}_inference.json"
            import json
            with open(single_json, "w", encoding="utf-8") as fp:
                json.dump(res.to_dict(), fp, indent=2)
            print(f"\nSaved detailed JSON to {single_json}")

    else:
        # Directory processing
        out_dir = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "reports" / "offline_inference_runs")
        batch = engine.infer_folder(input_path, recursive=(not args.non_recursive), output_dir=out_dir)

        print(f"\nProcessed {batch.total_files_processed} files in {batch.total_processing_time_ms:.1f} ms.")
        print(f"Total audio duration: {batch.total_audio_duration_s:.1f}s ({batch.total_audio_duration_s/60:.2f} min)")
        print(f"Total windows evaluated: {batch.total_windows_processed}")
        print(f"Failed files: {batch.total_files_failed}")

        pos_recs = sum(1 for r in batch.results if r.aggregated_prediction == "explosion")
        print(f"Aggregated Detections: {pos_recs} Explosion, {len(batch.results) - pos_recs} Non-Explosion")
        print(f"\nDetailed CSV & JSON written to: {out_dir}")

if __name__ == "__main__":
    main()

