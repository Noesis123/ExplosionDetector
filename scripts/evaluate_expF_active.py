"""Evaluate Active ExpF + AEV Model on the 92-recording Benchmark.

Self-contained evaluation script for the active single binary CNN:
- Architecture: ExplosionBinaryCNN (130,146 parameters)
- Checkpoint: candidate_expF_accuracy_optimized_best.pt
- Tracker: AcousticVerifiedEventTracker (Acoustic physics verification)
- Evaluation set: 92 historical benchmark recordings
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

TARGET_SR = 22050
WINDOW_SAMPLES = 22050
HOP_SAMPLES = 5512
SQUELCH_DBFS = -34.0
THRESHOLD = 0.70

def main():
    print("=" * 70)
    print("EVALUATING ACTIVE EXPF + AEV EXPLOSION DETECTION SYSTEM")
    print("=" * 70)

    # 1. Load Model
    ckpt_path = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model = ExplosionBinaryCNN(freeze_bn=True)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded Active Model: {ckpt_path.name} (Parameters: {sum(p.numel() for p in model.parameters()):,})")

    # 2. Load Manifest & Benchmark Recordings
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()
    print(f"Loaded Benchmark Set: {len(eval_df)} recordings")

    preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)

    cm = np.zeros((2, 2), dtype=int)
    dispatched_events = 0
    fp_events = 0
    category_fps = {}
    missed_explosions = []
    recovered_positives = []

    t0 = time.time()
    for idx, r in eval_df.iterrows():
        p = PROJECT_ROOT / r["file_path"]
        if not p.exists():
            print(f"Warning: file not found: {p}")
            continue

        lbl = r["label"]
        cat = r.get("category", "unknown")
        is_pos = (lbl == "explosion")
        gt_idx = 1 if is_pos else 0

        sig, sr, dur, ch = load_audio_file(p)
        mono = preprocessor.to_mono(sig)
        if sr != TARGET_SR:
            mono = preprocessor.resample(mono, orig_sr=sr)
        if len(mono) < WINDOW_SAMPLES:
            mono = np.pad(mono, (0, WINDOW_SAMPLES - len(mono)), mode="constant")

        tracker = AcousticVerifiedEventTracker(
            squelch_dbfs=SQUELCH_DBFS,
            conf_threshold=THRESHOLD,
            demo_threshold=0.66,
        )

        n_windows = max(1, (len(mono) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
        windows = [mono[i * HOP_SAMPLES : i * HOP_SAMPLES + WINDOW_SAMPLES] for i in range(n_windows)]

        file_dispatched = 0
        file_alerted = False

        with torch.no_grad():
            for i, w in enumerate(windows):
                win_time = i * (HOP_SAMPLES / TARGET_SR)
                rms = np.sqrt(np.mean(w**2) + 1e-12)
                dbfs = float(20.0 * np.log10(rms + 1e-12))
                peak = float(np.max(np.abs(w)))
                crest = peak / max(rms, 1e-9)

                if dbfs < SQUELCH_DBFS:
                    decision = tracker.process_window_with_acoustics(
                        time_sec=win_time,
                        rms_dbfs=dbfs,
                        p_nt=1.0,
                        p_exp=0.0,
                        p_fw=0.0,
                        peak=peak,
                        lf_ratio=1.0,
                        hf_ratio=0.0,
                        crest=1.0,
                    )
                else:
                    spec = preprocessor.compute_log_mel_spectrogram(w)
                    t_in = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
                    logits = model(t_in)
                    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
                    p_not_exp = float(probs[0])
                    p_exp = float(probs[1])

                    fft_mag = np.abs(np.fft.rfft(w))
                    freqs = np.fft.rfftfreq(len(w), d=1.0 / TARGET_SR)
                    tot_e = np.sum(fft_mag ** 2) + 1e-9
                    lf_ratio = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)] ** 2) / tot_e)
                    hf_ratio = float(np.sum(fft_mag[freqs > 4000.0] ** 2) / tot_e)
                    sub_bass_ratio = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 200.0)] ** 2) / tot_e)

                    # Subframe dynamic range (4 subframes of 250ms each)
                    sub4 = [w[k * 5512 : (k + 1) * 5512] for k in range(4)]
                    sub4_rms = [20.0 * np.log10(max(float(np.sqrt(np.mean(sf ** 2))), 1e-9)) for sf in sub4]
                    subframe_range = max(sub4_rms) - min(sub4_rms)

                    decision = tracker.process_window_with_acoustics(
                        time_sec=win_time,
                        rms_dbfs=dbfs,
                        p_nt=p_not_exp,
                        p_exp=p_exp,
                        p_fw=0.0,
                        peak=peak,
                        lf_ratio=lf_ratio,
                        hf_ratio=hf_ratio,
                        crest=crest,
                        sub_bass_ratio=sub_bass_ratio,
                        subframe_range=subframe_range,
                    )

                if decision.get("action") == "initial_alert":
                    a_cls = decision.get("alert", {}).get("alert_class") if decision.get("alert") else decision.get("alert_class")
                    if a_cls == "explosion":
                        file_dispatched += 1
                        file_alerted = True

        pred_idx = 1 if file_alerted else 0
        cm[gt_idx, pred_idx] += 1
        dispatched_events += file_dispatched

        if not is_pos and file_alerted:
            fp_events += file_dispatched
            category_fps[cat] = category_fps.get(cat, 0) + 1
        elif is_pos and not file_alerted:
            missed_explosions.append(Path(r["file_path"]).name)
        elif is_pos and file_alerted:
            recovered_positives.append(Path(r["file_path"]).name)

    elapsed = time.time() - t0

    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    print("\n" + "=" * 70)
    print("EXPF + AEV BENCHMARK EVALUATION RESULTS (Threshold 0.70):")
    print("=" * 70)
    print(f"  Explosion Recall   : {recall * 100:.2f}% ({tp}/{tp + fn})")
    print(f"  Explosion Precision: {precision * 100:.2f}% ({tp}/{tp + fp})")
    print(f"  F1 Score           : {f1 * 100:.2f}%")
    print(f"  Overall Accuracy   : {accuracy * 100:.2f}% ({(tp + tn)}/{(tp + fp + fn + tn)})")
    print(f"  False Positive Rec : {fp}/{tn + fp} files (Rejection Rate: {tn / (tn + fp) * 100:.2f}%)")
    print(f"  Total FP Events    : {fp_events} dispatched false alerts")
    print(f"  Evaluation Time    : {elapsed:.2f}s ({elapsed / len(eval_df) * 1000:.1f} ms/file)")
    print("-" * 70)
    print("  False Positive Category Breakdown:")
    for cat, cnt in sorted(category_fps.items(), key=lambda x: -x[1]):
        print(f"    - {cat}: {cnt} files")
    print("-" * 70)
    print(f"  Missed Explosions ({len(missed_explosions)}): {missed_explosions}")
    print("=" * 70)

if __name__ == "__main__":
    main()
