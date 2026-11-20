import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
import numpy as np
import pandas as pd
from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints"

prep = AudioPreprocessor(sample_rate=22050)

def load_m(ckpt_name):
    p = CKPT_DIR / ckpt_name
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    m = ExplosionBinaryCNN(freeze_bn=True)
    m.load_state_dict(sd)
    m.eval()
    return m

m_expF = load_m("candidate_expF_accuracy_optimized_best.pt")
m_expP = load_m("candidate_expP_recording_balanced_final.pt")

m_df = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv")
hist_df = m_df[m_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

def eval_files(model):
    res = {}
    for idx, r in hist_df.iterrows():
        p = PROJECT_ROOT / r["file_path"]
        if not p.exists():
            continue
        fname = p.name
        is_pos = (r["label"] == "explosion")
        cat = r.get("category", "unknown")

        sig, sr, dur, ch = load_audio_file(p)
        mono = prep.to_mono(sig)
        if sr != 22050:
            mono = prep.resample(mono, orig_sr=sr)
        if len(mono) < 22050:
            mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")
        nw = max(1, (len(mono) - 22050) // 5512 + 1)
        windows = [mono[i * 5512 : i * 5512 + 22050] for i in range(nw)]

        tracker = AcousticVerifiedEventTracker(squelch_dbfs=-45.0, conf_threshold=0.70, min_lf_ratio=0.35, max_hf_ratio=0.15)
        alerted = False
        max_p = 0.0

        for i, w in enumerate(windows):
            win_time = i * (5512 / 22050)
            rms = np.sqrt(np.mean(w**2) + 1e-12)
            dbfs = float(20.0 * np.log10(rms + 1e-12))
            peak = float(np.max(np.abs(w)))

            if dbfs < -45.0:
                dec = tracker.process_window_with_acoustics(win_time, dbfs, 1.0, 0.0, 0.0, peak, 1.0, 0.0)
            else:
                spec = prep.compute_log_mel_spectrogram(w)
                tin = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
                with torch.no_grad():
                    probs = torch.softmax(model(tin), dim=-1).squeeze().numpy()
                p_exp = float(probs[1])
                p_nt = float(probs[0])
                if p_exp > max_p:
                    max_p = p_exp
                fft_mag = np.abs(np.fft.rfft(w))
                freqs = np.fft.rfftfreq(len(w), d=1.0 / 22050)
                tot_e = np.sum(fft_mag ** 2) + 1e-9
                lf = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
                hf = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)
                dec = tracker.process_window_with_acoustics(win_time, dbfs, p_nt, p_exp, 0.0, peak, lf, hf)

            if dec.get("action") == "initial_alert":
                a_cls = dec.get("alert", {}).get("alert_class") if dec.get("alert") else dec.get("alert_class")
                if a_cls == "explosion":
                    alerted = True

        res[fname] = {"is_pos": is_pos, "category": cat, "alerted": alerted, "max_p": max_p}
    return res

print("Evaluating ExpF...")
f_res = eval_files(m_expF)
print("Evaluating ExpP...")
p_res = eval_files(m_expP)

print("\n=== POSITIVES (24 files) COMPARISON ===")
lost_pos = []
for fname, d_f in f_res.items():
    if not d_f["is_pos"]:
        continue
    d_p = p_res[fname]
    stat_f = "DETECTED" if d_f["alerted"] else "MISSED"
    stat_p = "DETECTED" if d_p["alerted"] else "MISSED"
    diff = d_p["max_p"] - d_f["max_p"]
    flag = ""
    if d_f["alerted"] and not d_p["alerted"]:
        flag = ">>> NEWLY MISSED! <<<"
        lost_pos.append((fname, d_f["max_p"], d_p["max_p"]))
    elif not d_f["alerted"] and d_p["alerted"]:
        flag = ">>> RECOVERED! <<<"
    print(f"  {fname:<45}: ExpF={d_f['max_p']:.4f} ({stat_f}) -> ExpP={d_p['max_p']:.4f} ({stat_p}) [diff={diff:+.4f}] {flag}")

print("\n=== NEGATIVES COMPARISON: FALSE POSITIVE CHANGES ===")
suppressed_fps = []
for fname, d_f in f_res.items():
    if d_f["is_pos"]:
        continue
    d_p = p_res[fname]
    diff = d_p["max_p"] - d_f["max_p"]
    if d_f["alerted"] and not d_p["alerted"]:
        suppressed_fps.append((fname, d_f["category"], d_f["max_p"], d_p["max_p"]))
        print(f"  [SUPPRESSED FP] {fname:<40}: {d_f['category']:<18} ExpF={d_f['max_p']:.4f} -> ExpP={d_p['max_p']:.4f} (diff={diff:+.4f})")
    elif not d_f["alerted"] and d_p["alerted"]:
        print(f"  [NEW FP]        {fname:<40}: {d_f['category']:<18} ExpF={d_f['max_p']:.4f} -> ExpP={d_p['max_p']:.4f} (diff={diff:+.4f})")

# Save detailed json report
summary_data = {
    "lost_positives": lost_pos,
    "suppressed_false_positives": suppressed_fps,
    "expF_tp": sum(1 for d in f_res.values() if d["is_pos"] and d["alerted"]),
    "expF_fp": sum(1 for d in f_res.values() if not d["is_pos"] and d["alerted"]),
    "expP_tp": sum(1 for d in p_res.values() if d["is_pos"] and d["alerted"]),
    "expP_fp": sum(1 for d in p_res.values() if not d["is_pos"] and d["alerted"]),
}

with open(PROJECT_ROOT / "reports" / "candidate_expP_file_differences.json", "w") as f:
    json.dump(summary_data, f, indent=2)
print("\nSaved file differences to reports/candidate_expP_file_differences.json")
