import sys
from pathlib import Path
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

ckpt = torch.load(PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt", map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

manifest = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv")
bench = manifest[manifest["split"].isin(["val", "holdout", "test", "challenge"])]
preprocessor = AudioPreprocessor(sample_rate=22050)

def evaluate_preset(preset):
    tracker = AcousticVerifiedEventTracker(config_preset=preset)
    tp, fp, tn, fn = 0, 0, 0, 0
    demo_07, demo_08 = False, False

    for _, r in bench.iterrows():
        p = PROJECT_ROOT / r["file_path"]
        is_pos = (r["label"] == "explosion")
        sig, sr, dur, ch = load_audio_file(p)
        mono = preprocessor.to_mono(sig)
        if sr != 22050:
            mono = preprocessor.resample(mono, orig_sr=sr)
        if len(mono) < 22050:
            mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")

        nw = max(1, (len(mono) - 22050) // 5512 + 1)
        windows = [mono[i * 5512 : i * 5512 + 22050] for i in range(nw)]
        tracker.reset()
        alerted = False

        with torch.no_grad():
            for i, w in enumerate(windows):
                win_time = i * (5512 / 22050)
                rms = np.sqrt(np.mean(w**2) + 1e-12)
                dbfs = float(20.0 * np.log10(rms + 1e-12))
                peak_val = float(np.max(np.abs(w)))
                crest_val = peak_val / max(rms, 1e-9)

                if dbfs < tracker.squelch_dbfs:
                    dec = tracker.process_window_with_acoustics(
                        time_sec=win_time, rms_dbfs=dbfs, p_nt=1.0, p_exp=0.0, p_fw=0.0,
                        peak=peak_val, lf_ratio=1.0, hf_ratio=0.0, crest=1.0,
                    )
                else:
                    spec = preprocessor.compute_log_mel_spectrogram(w)
                    t_in = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
                    probs = torch.softmax(model(t_in), dim=-1).squeeze().numpy()
                    p_exp = float(probs[1])
                    p_nt = float(probs[0])

                    fft_mag = np.abs(np.fft.rfft(w))
                    freqs = np.fft.rfftfreq(len(w), d=1.0 / 22050)
                    tot_e = np.sum(fft_mag ** 2) + 1e-12
                    lf = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)] ** 2) / tot_e)
                    hf = float(np.sum(fft_mag[freqs > 4000.0] ** 2) / tot_e)
                    sub20_80 = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 80.0)] ** 2) / tot_e)
                    sub_bass = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 200.0)] ** 2) / tot_e)
                    mid = float(np.sum(fft_mag[(freqs >= 1000.0) & (freqs < 4000.0)] ** 2) / tot_e)

                    sub4 = [w[k * 5512 : (k + 1) * 5512] for k in range(4)]
                    sub4_rms = [20.0 * np.log10(max(float(np.sqrt(np.mean(sf ** 2))), 1e-9)) for sf in sub4]
                    subframe_range = max(sub4_rms) - min(sub4_rms)

                    e_half1 = np.sum(w[:11025] ** 2) + 1e-9
                    e_half2 = np.sum(w[11025:] ** 2) + 1e-9
                    decay_ratio_in_win = float(e_half2 / e_half1)

                    dec = tracker.process_window_with_acoustics(
                        time_sec=win_time, rms_dbfs=dbfs, p_nt=p_nt, p_exp=p_exp, p_fw=0.0,
                        peak=peak_val, lf_ratio=lf, hf_ratio=hf, crest=crest_val,
                        sub_bass_ratio=sub_bass, subframe_range=subframe_range,
                        mid_ratio=mid, sub20_80_ratio=sub20_80,
                        decay_ratio_in_win=decay_ratio_in_win,
                    )

                if dec.get("action") == "initial_alert":
                    alerted = True
                    break

        if is_pos and alerted:
            tp += 1
            if "07022507" in p.name:
                demo_07 = True
            if "07022508" in p.name:
                demo_08 = True
        elif is_pos and not alerted:
            fn += 1
        elif not is_pos and alerted:
            fp += 1
        else:
            tn += 1

    rec = tp / (tp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "preset": preset, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "rec": rec, "prec": prec, "f1": f1,
        "demo_07": demo_07, "demo_08": demo_08
    }

print("Running Verified Preset Metrics Evaluation...")
print("1. Evaluating Config C (Historical Baseline)...")
res_c = evaluate_preset("Config C")
print(f"   Config C Results: TP={res_c['tp']}, FP={res_c['fp']}, TN={res_c['tn']}, FN={res_c['fn']}, Prec={res_c['prec']*100:.2f}%, Rec={res_c['rec']*100:.2f}%, F1={res_c['f1']:.4f}")

print("2. Evaluating Config E (Production Target)...")
res_e = evaluate_preset("Config E")
print(f"   Config E Results: TP={res_e['tp']}, FP={res_e['fp']}, TN={res_e['tn']}, FN={res_e['fn']}, Prec={res_e['prec']*100:.2f}%, Rec={res_e['rec']*100:.2f}%, F1={res_e['f1']:.4f}")
print(f"   Demolitions: 07={res_e['demo_07']}, 08={res_e['demo_08']}")
