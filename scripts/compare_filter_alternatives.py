import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import numpy as np
import scipy.signal as signal
import torch
import pandas as pd

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker
from audio.microphone_stream import PhysicalWaveInCapture

print("=" * 75)
print("COMPARATIVE EVALUATION OF MICROPHONE AUDIO CONDITIONING ALTERNATIVES")
print("=" * 75)

# 1. Load active model
ckpt_path = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

prep = AudioPreprocessor(sample_rate=22050)

# Capture 4 seconds of quiet room audio from hardware mic
samples = []
cap = PhysicalWaveInCapture(device_id=0, sample_rate=22050, chunk_callback=lambda c, sr: samples.append(c))
cap.start()
time.sleep(4.0)
cap.stop()
quiet_raw = np.concatenate(samples)
print(f"Captured {len(quiet_raw)/22050:.2f}s quiet room audio. Raw RMS: {20*np.log10(max(np.sqrt(np.mean(quiet_raw**2)), 1e-9)):.2f} dBFS")

# Benchmark files
manifest_df = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv")
eval_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

vulnerable_files = [
    "data/positives/bbc_demolition_07022508.mp3",
    "data/phase13_approved/positives/fs_367622_fxkid2.mp3",
    "data/phase13_approved/positives/fs_323316_alfrodou.mp3"
]

all_exp_files = eval_df[eval_df["label"] == "explosion"]["file_path"].tolist()
neg_sample_files = eval_df[eval_df["label"] != "explosion"]["file_path"].tolist()[:20]

# Define Filtering Options
def get_filters():
    filters = {}
    
    # Option A: Baseline (Current 90 Hz 2nd-order Butterworth HPF)
    sos_90 = signal.butter(2, 90.0, btype="highpass", fs=22050, output="sos")
    filters["OptA_HPF90 (Baseline)"] = lambda sig: signal.sosfilt(sos_90, sig).astype(np.float32)
    
    # Option B: 70 Hz 2nd-order Butterworth HPF
    sos_70 = signal.butter(2, 70.0, btype="highpass", fs=22050, output="sos")
    filters["OptB_HPF70"] = lambda sig: signal.sosfilt(sos_70, sig).astype(np.float32)
    
    # Option C: DC Blocker (25 Hz HPF) + Dual Notch Filter at Fan Peaks (37.1 Hz & 86.5 Hz, Q=8)
    sos_dc25 = signal.butter(2, 25.0, btype="highpass", fs=22050, output="sos")
    b_n1, a_n1 = signal.iirnotch(37.1, 8.0, fs=22050)
    b_n2, a_n2 = signal.iirnotch(86.5, 8.0, fs=22050)
    def filter_notch(sig):
        s1 = signal.sosfilt(sos_dc25, sig)
        s2 = signal.lfilter(b_n1, a_n1, s1)
        s3 = signal.lfilter(b_n2, a_n2, s2)
        return s3.astype(np.float32)
    filters["OptC_DC25_NotchFan"] = filter_notch

    # Option D: 4th-order Butterworth HPF at 75 Hz (Steeper transition band)
    sos_75_ord4 = signal.butter(4, 75.0, btype="highpass", fs=22050, output="sos")
    filters["OptD_HPF75_Ord4"] = lambda sig: signal.sosfilt(sos_75_ord4, sig).astype(np.float32)

    # Option E: Raw (No filtering)
    filters["OptE_Raw (No Filter)"] = lambda sig: sig.astype(np.float32)

    return filters

filters = get_filters()

def evaluate_pipeline_on_file(audio_mono, filter_fn):
    conditioned = filter_fn(audio_mono)
    tracker = AcousticVerifiedEventTracker(squelch_dbfs=-45.0, conf_threshold=0.70, min_lf_ratio=0.35, max_hf_ratio=0.15)
    n_windows = max(1, (len(conditioned) - 22050) // 5512 + 1)
    
    alerted = False
    max_p = 0.0
    lf_at_max = 0.0
    hf_at_max = 0.0
    rms_at_max = -99.0
    
    with torch.no_grad():
        for i in range(n_windows):
            w = conditioned[i * 5512 : i * 5512 + 22050]
            win_time = i * (5512 / 22050)
            rms = np.sqrt(np.mean(w**2) + 1e-12)
            dbfs = float(20.0 * np.log10(rms + 1e-12))
            peak = float(np.max(np.abs(w)))

            if dbfs < -45.0:
                dec = tracker.process_window_with_acoustics(win_time, dbfs, 1.0, 0.0, 0.0, peak, 1.0, 0.0)
            else:
                spec = prep.compute_log_mel_spectrogram(w)
                t_in = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
                probs = torch.softmax(model(t_in), dim=-1).squeeze().numpy()
                p_exp = float(probs[1])

                fft_mag = np.abs(np.fft.rfft(w))
                freqs = np.fft.rfftfreq(len(w), d=1.0 / 22050)
                tot_e = np.sum(fft_mag ** 2) + 1e-9
                lf = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
                hf = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)

                if p_exp > max_p:
                    max_p = p_exp
                    lf_at_max = lf
                    hf_at_max = hf
                    rms_at_max = dbfs

                dec = tracker.process_window_with_acoustics(win_time, dbfs, float(probs[0]), p_exp, 0.0, peak, lf, hf)

            if dec.get("action") == "initial_alert":
                alerted = True

    return alerted, max_p, lf_at_max, hf_at_max, rms_at_max

print("\n--- RESULTS TABLE ACROSS CONDITIONING OPTIONS ---")

comparison_results = []

for opt_name, filt_fn in filters.items():
    t_start = time.perf_counter()
    
    # 1. Quiet room stats
    q_filt = filt_fn(quiet_raw)
    q_rms = 20 * np.log10(max(np.sqrt(np.mean(q_filt**2)), 1e-9))
    q_wins = [q_filt[i*5512:i*5512+22050] for i in range(max(1, (len(q_filt)-22050)//5512+1))]
    q_open = sum(1 for w in q_wins if (20*np.log10(max(np.sqrt(np.mean(w**2)), 1e-9))) >= -45.0)
    q_open_pct = (q_open / len(q_wins)) * 100 if q_wins else 0.0

    # 2. Vulnerable explosions
    vuln_detected = 0
    vuln_details = []
    for vf in vulnerable_files:
        p = PROJECT_ROOT / vf
        sig, sr, dur, ch = load_audio_file(p)
        mono = prep.resample(prep.to_mono(sig), sr)
        if len(mono) < 22050: mono = np.pad(mono, (0, 22050-len(mono)))
        al, p_exp, lf, hf, rms = evaluate_pipeline_on_file(mono, filt_fn)
        if al: vuln_detected += 1
        vuln_details.append(f"{p.name[:12]}: {'HIT' if al else 'MISS'}(p={p_exp:.3f})")

    # 3. All 24 benchmark explosions
    total_exp_detected = 0
    for ef in all_exp_files:
        p = PROJECT_ROOT / ef
        sig, sr, dur, ch = load_audio_file(p)
        mono = prep.resample(prep.to_mono(sig), sr)
        if len(mono) < 22050: mono = np.pad(mono, (0, 22050-len(mono)))
        al, p_exp, lf, hf, rms = evaluate_pipeline_on_file(mono, filt_fn)
        if al: total_exp_detected += 1

    # 4. Sample false positives
    neg_fps = 0
    for nf in neg_sample_files:
        p = PROJECT_ROOT / nf
        sig, sr, dur, ch = load_audio_file(p)
        mono = prep.resample(prep.to_mono(sig), sr)
        if len(mono) < 22050: mono = np.pad(mono, (0, 22050-len(mono)))
        al, p_exp, lf, hf, rms = evaluate_pipeline_on_file(mono, filt_fn)
        if al: neg_fps += 1

    t_elapsed = time.perf_counter() - t_start

    comparison_results.append({
        "Option": opt_name,
        "Quiet RMS": q_rms,
        "Quiet Squelch Open %": q_open_pct,
        "Vuln Explosions (of 3)": vuln_detected,
        "Total Recall (of 24)": f"{total_exp_detected}/24 ({total_exp_detected/24*100:.1f}%)",
        "Neg FPs (of 20)": f"{neg_fps}/20",
        "Time (s)": round(t_elapsed, 2),
        "Details": ", ".join(vuln_details)
    })

print(f"{'Option':<25} | {'Quiet RMS':<10} | {'Open %':<8} | {'Vuln Hit':<8} | {'Total Recall':<16} | {'Neg FP (of 20)':<14} | {'CPU (s)'}")
print("-" * 105)
for r in comparison_results:
    print(f"{r['Option']:<25} | {r['Quiet RMS']:<10.2f} | {r['Quiet Squelch Open %']:<8.1f} | {r['Vuln Explosions (of 3)']:<8} | {r['Total Recall (of 24)']:<16} | {r['Neg FPs (of 20)']:<14} | {r['Time (s)']}")

print("\nVulnerable File Details:")
for r in comparison_results:
    print(f"  {r['Option']:<25}: {r['Details']}")
