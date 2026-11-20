import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

CODEX_DIR = Path("D:/NOESIS/Codex_Evaluation_Dataset")
CKPT_DIR = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints"
TARGET_SR = 22050
WINDOW_SAMPLES = 22050
HOP_SAMPLES = 5512
SQUELCH_DBFS = -45.0
THRESHOLD = 0.70

preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)

def eval_codex_manifest(ckpt_file, manifest_file):
    ckpt_path = CKPT_DIR / ckpt_file
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = ExplosionBinaryCNN(freeze_bn=True)
    model.load_state_dict(sd)
    model.eval()

    df = pd.read_csv(manifest_file)
    tp, fp, tn, fn = 0, 0, 0, 0
    fp_by_cat = {}

    with torch.no_grad():
        for idx, r in df.iterrows():
            p = CODEX_DIR / r["audio_path"]
            if not p.exists():
                continue
            cat = r["category"]
            is_pos = (cat == "surface_high_explosive")

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

            n_w = max(1, (len(mono) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
            file_alerted = False

            for i in range(n_w):
                w = mono[i * HOP_SAMPLES : i * HOP_SAMPLES + WINDOW_SAMPLES]
                win_time = i * (HOP_SAMPLES / TARGET_SR)
                rms = np.sqrt(np.mean(w**2) + 1e-12)
                dbfs = float(20.0 * np.log10(rms + 1e-12))
                peak = float(np.max(np.abs(w)))

                if dbfs < SQUELCH_DBFS:
                    d = tracker.process_window_with_acoustics(
                        time_sec=win_time, rms_dbfs=dbfs, p_nt=1.0, p_exp=0.0, p_fw=0.0,
                        peak=peak, lf_ratio=1.0, hf_ratio=0.0
                    )
                else:
                    spec = preprocessor.compute_log_mel_spectrogram(w)
                    t_in = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
                    probs = torch.softmax(model(t_in), dim=-1).squeeze().numpy()
                    p_not_exp = float(probs[0])
                    p_exp = float(probs[1])

                    fft_mag = np.abs(np.fft.rfft(w))
                    freqs = np.fft.rfftfreq(len(w), d=1.0 / TARGET_SR)
                    tot_e = np.sum(fft_mag ** 2) + 1e-9
                    lf_ratio = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
                    hf_ratio = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)

                    d = tracker.process_window_with_acoustics(
                        time_sec=win_time, rms_dbfs=dbfs, p_nt=p_not_exp, p_exp=p_exp, p_fw=0.0,
                        peak=peak, lf_ratio=lf_ratio, hf_ratio=hf_ratio
                    )

                if d.get("action") == "initial_alert":
                    a_cls = d.get("alert", {}).get("alert_class") if d.get("alert") else d.get("alert_class")
                    if a_cls == "explosion":
                        file_alerted = True

            if is_pos:
                if file_alerted:
                    tp += 1
                else:
                    fn += 1
            else:
                if file_alerted:
                    fp += 1
                    fp_by_cat[cat] = fp_by_cat.get(cat, 0) + 1
                else:
                    tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)

    return {
        "ckpt": ckpt_file,
        "manifest": manifest_file.name,
        "total": tp + fn + fp + tn,
        "pos": tp + fn, "neg": fp + tn,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "rec": rec, "prec": prec, "f1": f1, "acc": acc,
        "fp_by_cat": fp_by_cat
    }

print("=" * 80)
print("EVALUATING EXPF vs EXPL ON CODEX RESERVED FROZEN TEST SET 2 (60 FILES)")
print("=" * 80)

for ckpt in ["candidate_expF_accuracy_optimized_best.pt", "candidate_expL_anchored_final.pt"]:
    res = eval_codex_manifest(ckpt, PROJECT_ROOT / "reports" / "codex_reserved_eval_frozen.csv")
    print(f"\n--- {ckpt} ---")
    print(f"Recall: {res['rec']*100:.1f}% ({res['tp']}/{res['pos']})")
    print(f"Precision: {res['prec']*100:.1f}%")
    print(f"F1-Score: {res['f1']:.4f}")
    print(f"Accuracy: {res['acc']*100:.1f}%")
    print(f"False Positives: {res['fp']}/{res['neg']}")
    print(f"FP Breakdown: {res['fp_by_cat']}")

print("\n" + "=" * 80)
print("EVALUATING EXPF vs EXPL ON CODEX TEST SET 1 (86 FILES)")
print("=" * 80)

for ckpt in ["candidate_expF_accuracy_optimized_best.pt", "candidate_expL_anchored_final.pt"]:
    res = eval_codex_manifest(ckpt, PROJECT_ROOT / "reports" / "codex_selected_eval_subset.csv")
    print(f"\n--- {ckpt} ---")
    print(f"Recall: {res['rec']*100:.1f}% ({res['tp']}/{res['pos']})")
    print(f"Precision: {res['prec']*100:.1f}%")
    print(f"F1-Score: {res['f1']:.4f}")
    print(f"Accuracy: {res['acc']*100:.1f}%")
    print(f"False Positives: {res['fp']}/{res['neg']}")
    print(f"FP Breakdown: {res['fp_by_cat']}")
