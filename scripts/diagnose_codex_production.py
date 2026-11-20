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
SQUELCH_DBFS = -34.0
THRESHOLD = 0.70

manifest_path = PROJECT_ROOT / "data" / "manifests" / "codex_expanded_eval_manifest.csv"
df = pd.read_csv(manifest_path)

ckpt_path = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)

tp, fp, tn, fn = 0, 0, 0, 0
fp_by_cat = {}
missed_exp_reasons = {"squelch": 0, "low_p": 0, "aev_rejection": 0}
fn_details = []
fp_details = []

t0 = time.time()
for idx, r in df.iterrows():
    p = Path(r["absolute_audio_path"])
    if not p.exists():
        continue
    is_pos = (r["eval_binary_label"] == "explosion")
    cat = r["category"]

    sig, sr, dur, ch = load_audio_file(p)
    mono = preprocessor.to_mono(sig)
    if sr != TARGET_SR:
        mono = preprocessor.resample(mono, orig_sr=sr)
    if len(mono) < WINDOW_SAMPLES:
        mono = np.pad(mono, (0, WINDOW_SAMPLES - len(mono)), mode="constant")

    tracker = AcousticVerifiedEventTracker(
        squelch_dbfs=SQUELCH_DBFS,
        conf_threshold=THRESHOLD,
    )

    n_windows = max(1, (len(mono) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
    file_alerted = False
    max_p = 0.0
    max_rms = -99.9
    highest_p_reason = None
    had_aev_suppression = False

    with torch.no_grad():
        for i in range(n_windows):
            w = mono[i * HOP_SAMPLES : i * HOP_SAMPLES + WINDOW_SAMPLES]
            win_time = i * (HOP_SAMPLES / TARGET_SR)
            rms = np.sqrt(np.mean(w**2) + 1e-12)
            dbfs = float(20.0 * np.log10(rms + 1e-12))
            peak = float(np.max(np.abs(w)))
            crest = peak / max(rms, 1e-9)
            if dbfs > max_rms:
                max_rms = dbfs

            if dbfs < SQUELCH_DBFS:
                decision = tracker.process_window_with_acoustics(
                    time_sec=win_time, rms_dbfs=dbfs, p_nt=1.0, p_exp=0.0, p_fw=0.0,
                    peak=peak, lf_ratio=1.0, hf_ratio=0.0, crest=1.0,
                )
            else:
                spec = preprocessor.compute_log_mel_spectrogram(w)
                t_in = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
                probs = torch.softmax(model(t_in), dim=-1).squeeze().numpy()
                p_exp = float(probs[1])
                p_nt = float(probs[0])
                if p_exp > max_p:
                    max_p = p_exp

                fft_mag = np.abs(np.fft.rfft(w))
                freqs = np.fft.rfftfreq(len(w), d=1.0 / TARGET_SR)
                tot_e = np.sum(fft_mag ** 2) + 1e-9
                lf = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)] ** 2) / tot_e)
                hf = float(np.sum(fft_mag[freqs > 4000.0] ** 2) / tot_e)
                sb = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 200.0)] ** 2) / tot_e)
                sub4 = [w[k * 5512 : (k + 1) * 5512] for k in range(4)]
                sub4_rms = [20.0 * np.log10(max(float(np.sqrt(np.mean(sf ** 2))), 1e-9)) for sf in sub4]
                subframe_range = max(sub4_rms) - min(sub4_rms)

                decision = tracker.process_window_with_acoustics(
                    time_sec=win_time, rms_dbfs=dbfs, p_nt=p_nt, p_exp=p_exp, p_fw=0.0,
                    peak=peak, lf_ratio=lf, hf_ratio=hf, crest=crest,
                    sub_bass_ratio=sb, subframe_range=subframe_range
                )

                if decision.get("action") == "suppressed_acoustic_rejection":
                    had_aev_suppression = True
                    highest_p_reason = decision.get("reason")

                if decision.get("action") == "initial_alert":
                    file_alerted = True

    if is_pos:
        if file_alerted:
            tp += 1
        else:
            fn += 1
            if max_rms < SQUELCH_DBFS:
                missed_exp_reasons["squelch"] += 1
                reason = f"Below squelch (Max RMS {max_rms:.1f} < {SQUELCH_DBFS})"
            elif max_p < THRESHOLD:
                missed_exp_reasons["low_p"] += 1
                reason = f"Low CNN confidence (Max p={max_p:.3f} < {THRESHOLD})"
            else:
                missed_exp_reasons["aev_rejection"] += 1
                reason = f"AEV rejected: {highest_p_reason}"
            fn_details.append((p.name, dur, max_rms, max_p, reason))
    else:
        if file_alerted:
            fp += 1
            fp_by_cat[cat] = fp_by_cat.get(cat, 0) + 1
            fp_details.append((cat, p.name, max_rms, max_p))
        else:
            tn += 1

elapsed = time.time() - t0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
acc = (tp + tn) / (tp + tn + fp + fn)

print("\n" + "=" * 70)
print("CODEX 300-FILE INDEPENDENT EVALUATION RESULTS (CURRENT PRODUCTION)")
print("=" * 70)
print(f"Time: {elapsed:.1f}s ({elapsed/len(df)*1000:.1f} ms/file)")
print(f"Recall:    {rec*100:.2f}% ({tp}/{tp+fn})")
print(f"Precision: {prec*100:.2f}% ({tp}/{tp+fp})")
print(f"F1-Score:  {f1*100:.2f}%")
print(f"Accuracy:  {acc*100:.2f}% ({tp+tn}/{len(df)})")
print(f"False Positives: {fp}/200 ({fp/200*100:.1f}%)")
print("\nFalse Positive Breakdown by Category:")
for c, cnt in sorted(fp_by_cat.items(), key=lambda x: -x[1]):
    print(f"  {c:<30}: {cnt}/200")
print(f"\nMissed Explosions Diagnosis (Total FN = {fn}):")
for rsn, cnt in missed_exp_reasons.items():
    print(f"  {rsn:<20}: {cnt}/{tp+fn} ({cnt/(tp+fn)*100:.1f}%)")
print("\nSample Missed Explosions (First 10):")
for f_info in fn_details[:10]:
    print(f"  {f_info[0][:35]:<35} Dur={f_info[1]:.1f}s RMS={f_info[2]:.1f} MaxP={f_info[3]:.3f} -> {f_info[4]}")
print("\nSample False Positives (First 10):")
for fp_info in fp_details[:10]:
    print(f"  [{fp_info[0]}] {fp_info[1][:35]:<35} RMS={fp_info[2]:.1f} MaxP={fp_info[3]:.3f}")
print("=" * 70)
