import numpy as np
import pandas as pd
import torch
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

ckpt = torch.load("explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt", map_location="cpu", weights_only=False)
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()
pre = AudioPreprocessor()

class TestTracker(AcousticVerifiedEventTracker):
    def process_window_with_acoustics(self, time_sec, rms_dbfs, p_nt, p_exp, p_fw=0.0, peak=0.0,
                                      lf_ratio=None, hf_ratio=None, crest=None, sub_bass_ratio=None,
                                      subframe_range=None, onset_surge=None):
        # Compute streaming onset surge from recent window history if not explicitly provided
        if onset_surge is None and len(self.window_buffer) >= 2:
            recent_rms = [w["rms_dbfs"] for w in list(self.window_buffer)[-8:]]
            baseline_rms = min(recent_rms)
            onset_surge = rms_dbfs - baseline_rms

        # Compute crest
        if crest is None and peak > 0.0:
            rms_linear = 10.0 ** (rms_dbfs / 20.0)
            crest = peak / max(rms_linear, 1e-9)

        # Standard trigger: p_exp >= conf_threshold (0.70)
        is_standard_trig = (p_exp >= self.conf_threshold)

        # Stationary noise attack check:
        # A genuine acoustic explosion at quiet/moderate energy (< -28 dBFS) MUST exhibit an explosive
        # transient attack (either intra-window dynamic range >= 3.5 dB or onset surge >= 3.0 dB).
        # Continuous stationary noise (AC airflow, laptop cooling fan, room hum) maintains flat subframe energy.
        has_attack = (
            (onset_surge is not None and onset_surge >= 3.0)
            or (subframe_range is not None and subframe_range >= 3.5)
            or rms_dbfs >= -28.0
        )

        effective_trig = is_standard_trig

        if effective_trig and rms_dbfs >= self.squelch_dbfs:
            if not has_attack:
                self.total_acoustically_rejected += 1
                return {
                    "time_sec": time_sec,
                    "action": "suppressed_acoustic_rejection",
                    "reason": f"Stationary noise rejection (SubRange {subframe_range:.1f} dB < 3.5 & Onset {onset_surge if onset_surge else 0:.1f} < 3.0)",
                    "alert": None,
                    "suppressed": True,
                    "active_event_id": self.active_event["event_id"] if self.active_event else None
                }

        return super().process_window_with_acoustics(
            time_sec=time_sec, rms_dbfs=rms_dbfs, p_nt=p_nt, p_exp=p_exp, p_fw=p_fw, peak=peak,
            lf_ratio=lf_ratio, hf_ratio=hf_ratio, crest=crest, sub_bass_ratio=sub_bass_ratio,
            subframe_range=subframe_range, onset_surge=onset_surge
        )

# Evaluate 92-file Benchmark
manifest_path = ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv"
manifest_df = pd.read_csv(manifest_path)
eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

tp = 0
fp = 0
tn = 0
fn = 0
fp_categories = {}
missed_exp = []

for idx, r in eval_df.iterrows():
    p = ROOT / r["file_path"]
    lbl = r["label"]
    cat = r.get("category", "unknown")
    is_pos = (lbl == "explosion")

    sig, sr, dur, ch = load_audio_file(p)
    mono = pre.to_mono(sig)
    if sr != 22050:
        mono = pre.resample(mono, orig_sr=sr)
    if len(mono) < 22050:
        mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")

    tracker = TestTracker(squelch_dbfs=-38.0, conf_threshold=0.70)
    n_win = max(1, (len(mono) - 22050) // 5512 + 1)
    file_alerted = False

    with torch.no_grad():
        for i in range(n_win):
            w = mono[i*5512 : i*5512 + 22050]
            rms = float(np.sqrt(np.mean(w**2)))
            dbfs = float(20.0 * math.log10(max(rms, 1e-9)))
            peak = float(np.max(np.abs(w)))
            crest = peak / max(rms, 1e-9)

            if dbfs < -38.0:
                d = tracker.process_window_with_acoustics(i*0.25, dbfs, 1.0, 0.0, crest=1.0)
            else:
                spec = pre.compute_log_mel_spectrogram(w)
                inp = torch.from_numpy(spec[np.newaxis, np.newaxis, :, :]).float()
                p_exp = float(torch.softmax(model(inp), dim=-1).squeeze()[1].item())

                fft_mag = np.abs(np.fft.rfft(w))
                freqs = np.fft.rfftfreq(len(w), d=1.0/22050)
                tot_e = np.sum(fft_mag[freqs >= 20.0]**2) + 1e-9
                lf = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)]**2) / tot_e)
                hf = float(np.sum(fft_mag[freqs > 4000.0]**2) / tot_e)
                sub_bass = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 200.0)]**2) / tot_e)

                sub4 = [w[k*5512 : (k+1)*5512] for k in range(4)]
                sub4_rms = [20.0 * math.log10(max(float(np.sqrt(np.mean(sf**2))), 1e-9)) for sf in sub4]
                sub_range = max(sub4_rms) - min(sub4_rms)

                d = tracker.process_window_with_acoustics(
                    i*0.25, dbfs, 1.0 - p_exp, p_exp,
                    peak=peak, lf_ratio=lf, hf_ratio=hf, crest=crest,
                    sub_bass_ratio=sub_bass, subframe_range=sub_range
                )

                if d.get("action") == "initial_alert":
                    file_alerted = True

    if is_pos:
        if file_alerted:
            tp += 1
        else:
            fn += 1
            missed_exp.append(p.name)
    else:
        if file_alerted:
            fp += 1
            fp_categories[cat] = fp_categories.get(cat, 0) + 1
        else:
            tn += 1

print("\n" + "=" * 70)
print(f"BENCHMARK RESULTS (92 Files):")
print(f"  TP: {tp}/24 (Recall: {tp/24*100:.2f}%)")
print(f"  FP: {fp}/68 (Precision: {tp/(tp+fp)*100:.2f}%)")
print(f"  F1: {2*tp/(2*tp+fp+fn)*100:.2f}%")
print(f"  Accuracy: {(tp+tn)/92*100:.2f}%")
print(f"  FP categories: {fp_categories}")
print(f"  Missed explosions: {missed_exp}")
print("=" * 70)
