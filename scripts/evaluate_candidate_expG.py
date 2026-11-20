import sys
from pathlib import Path
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
SQUELCH_DBFS = -45.0
THRESHOLD = 0.70

def evaluate_checkpoint(ckpt_name):
    ckpt_path = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / ckpt_name
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model = ExplosionBinaryCNN(freeze_bn=True)
    model.load_state_dict(sd)
    model.eval()

    manifest_path = PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
    manifest_df = pd.read_csv(manifest_path)
    eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

    preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)

    cm = np.zeros((2, 2), dtype=int)
    dispatched_events = 0
    fp_events = 0
    category_fps = {}
    missed_explosions = []

    t0 = time.time()
    for idx, r in eval_df.iterrows():
        p = PROJECT_ROOT / r["file_path"]
        if not p.exists():
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
            min_lf_ratio=0.35,
            max_hf_ratio=0.15
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

                if dbfs < SQUELCH_DBFS:
                    decision = tracker.process_window_with_acoustics(
                        time_sec=win_time,
                        rms_dbfs=dbfs,
                        p_nt=1.0,
                        p_exp=0.0,
                        p_fw=0.0,
                        peak=peak,
                        lf_ratio=1.0,
                        hf_ratio=0.0
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
                    lf_ratio = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
                    hf_ratio = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)

                    decision = tracker.process_window_with_acoustics(
                        time_sec=win_time,
                        rms_dbfs=dbfs,
                        p_nt=p_not_exp,
                        p_exp=p_exp,
                        p_fw=0.0,
                        peak=peak,
                        lf_ratio=lf_ratio,
                        hf_ratio=hf_ratio
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

    elapsed = time.time() - t0

    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    return {
        "ckpt": ckpt_name,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "fp_files": fp,
        "fp_events": fp_events,
        "category_fps": category_fps,
        "missed": missed_explosions,
        "latency_ms": elapsed / len(eval_df) * 1000
    }

print("Comparing Original ExpF vs Candidate ExpG vs Candidate ExpH...")
res_f = evaluate_checkpoint("candidate_expF_accuracy_optimized_best.pt")
res_h = evaluate_checkpoint("candidate_expH_recall_anchored_best.pt")

print("\n" + "=" * 70)
print(f"{'Metric':<25} | {'Original ExpF':<18} | {'Candidate ExpH':<18}")
print("=" * 70)
print(f"{'Recall':<25} | {res_f['recall']*100:.2f}% ({res_f['tp']}/{res_f['tp']+res_f['fn']})  | {res_h['recall']*100:.2f}% ({res_h['tp']}/{res_h['tp']+res_h['fn']})")
print(f"{'Precision':<25} | {res_f['precision']*100:.2f}% ({res_f['tp']}/{res_f['tp']+res_f['fp']})  | {res_h['precision']*100:.2f}% ({res_h['tp']}/{res_h['tp']+res_h['fp']})")
print(f"{'F1 Score':<25} | {res_f['f1']*100:.2f}%            | {res_h['f1']*100:.2f}%")
print(f"{'Overall Accuracy':<25} | {res_f['accuracy']*100:.2f}% ({res_f['tp']+res_f['tn']}/92)  | {res_h['accuracy']*100:.2f}% ({res_h['tp']+res_h['tn']}/92)")
print(f"{'False Positive Files':<25} | {res_f['fp_files']}/68             | {res_h['fp_files']}/68")
print(f"{'False Alert Events':<25} | {res_f['fp_events']} events         | {res_h['fp_events']} events")
print(f"{'Missed Explosions':<25} | {len(res_f['missed'])} files            | {len(res_h['missed'])} files")
print(f"{'CPU Latency / file':<25} | {res_f['latency_ms']:.1f} ms           | {res_h['latency_ms']:.1f} ms")
print("=" * 70)
print("Missed in ExpF:", res_f['missed'])
print("Missed in ExpH:", res_h['missed'])
