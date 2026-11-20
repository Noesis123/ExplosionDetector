import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from audio.microphone_stream import StreamingHighpassFilter
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN

MODEL_PATH = ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
preprocessor = AudioPreprocessor(sample_rate=22050)
hp_filter = StreamingHighpassFilter(cutoff_hz=35.0, sample_rate=22050)

model = ExplosionBinaryCNN()
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()

def run_simulation(files, squelch=-35.0, conf_threshold=0.70):
    total_alerts = 0
    file_results = []
    
    for fpath in files:
        raw_audio, sr, dur, ch = load_audio_file(fpath)
        mono = preprocessor.to_mono(raw_audio)
        res = preprocessor.resample(mono, orig_sr=sr)
        filt = hp_filter.process(res)
        
        win_len = 22050
        hop_len = int(0.25 * 22050)
        n_w = max(1, (len(filt) - win_len) // hop_len + 1)
        
        file_alerted = False
        max_p = 0.0
        alerts = []
        recent_rms = []
        
        for i in range(n_w):
            w = filt[i*hop_len : i*hop_len + win_len]
            time_sec = i * 0.25
            r_lin = float(np.sqrt(np.mean(w**2)))
            dbfs = 20.0 * np.log10(max(r_lin, 1e-9))
            peak = float(np.max(np.abs(w)))
            recent_rms.append(dbfs)
            if len(recent_rms) > 8:
                recent_rms.pop(0)
                
            if dbfs < squelch:
                continue
                
            spec = preprocessor.compute_log_mel_spectrogram(w)
            probs = torch.softmax(model(torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()), dim=1).squeeze().detach().numpy()
            p_exp = float(probs[1])
            if p_exp > max_p:
                max_p = p_exp
                
            fft_mag = np.abs(np.fft.rfft(w))
            freqs = np.fft.rfftfreq(len(w), d=1.0/22050.0)
            tot_e = np.sum(fft_mag[freqs >= 20.0]**2) + 1e-9
            lf = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)]**2) / tot_e)
            hf = float(np.sum(fft_mag[freqs > 4000.0]**2) / tot_e)
            crest = peak / max(r_lin, 1e-9)
            
            sub_len = len(w) // 4
            sub_rms = [20.0 * np.log10(max(np.sqrt(np.mean(w[s*sub_len:(s+1)*sub_len]**2)), 1e-9)) for s in range(4)]
            sub_rng = max(sub_rms) - min(sub_rms)
            onset_surge = dbfs - min(recent_rms) if len(recent_rms) >= 2 else 0.0
            
            # --- EVALUATE AEV RULES ---
            # 1. Confidence trigger
            has_demo_attack = (onset_surge >= 4.0 or sub_rng >= 5.0)
            is_demo = (p_exp >= 0.62 and lf >= 0.50 and hf <= 0.05 and crest >= 3.5 and has_demo_attack)
            is_standard = (p_exp >= conf_threshold)
            
            if not (is_standard or is_demo):
                continue
                
            # 2. Physics guards
            if lf < 0.40 or hf > 0.08 or crest < 2.5 or crest > 12.0:
                continue
                
            # 3. Adaptive marginal LF
            if p_exp < 0.78 and lf < 0.65:
                continue
                
            # 4. Stationary Continuous Noise Rejection
            if dbfs < -28.0 and sub_rng < 3.5:
                continue
                
            # 5. Stationary Low-Frequency Machinery / Room Ambient Rejection
            # Continuous motor/HVAC hum (LF >= 0.96, HF <= 0.010) without authentic explosive detonation attack
            if lf >= 0.96 and hf <= 0.010:
                has_detonation_attack = (sub_rng >= 7.0 or onset_surge >= 6.0)
                if not has_detonation_attack:
                    continue
                    
            # Passed all verification! Alert fired!
            file_alerted = True
            alerts.append((time_sec, p_exp, dbfs, sub_rng, lf, hf))
            break # one alert per file is enough to detect
            
        file_results.append({
            "filename": fpath.name,
            "alerted": file_alerted,
            "max_p": max_p,
            "alerts": alerts
        })
        if file_alerted:
            total_alerts += 1
            
    return total_alerts, file_results

# 1. Test on Room Ambient Recordings
ambient_files = [
    ROOT / "data/phase24_approved/room_ambient/p24_room_ambient_ac_rec07.m4a",
    ROOT / "data/phase24_approved/room_ambient/p24_room_ambient_ac_rec08.m4a",
    ROOT / "data/phase24_approved/room_ambient/p24_room_ambient_ac_rec09.m4a",
]
print("=" * 75)
print("TEST 1: PHASE 24 ROOM AMBIENT RECORDINGS (Expected: 0 alerts)")
print("=" * 75)
for sq in [-38.0, -35.0, -32.0]:
    n_alerts, res = run_simulation(ambient_files, squelch=sq)
    print(f"Squelch {sq} dBFS: {n_alerts} / {len(ambient_files)} files alerted")
    for r in res:
        print(f"  {r['filename']}: Alerted={r['alerted']}, max_p={r['max_p']:.3f}, Alerts={r['alerts']}")

# 2. Test on Historical Benchmark
manifest = pd.read_csv(ROOT / "data/manifests/dataset_manifest_phase21_expanded.csv")
eval_df = manifest[manifest["split"].isin(["val", "holdout", "test", "challenge"])]
pos_files = [ROOT / r["file_path"] for _, r in eval_df[eval_df["label"] == "explosion"].iterrows() if (ROOT / r["file_path"]).exists()]
neg_files = [ROOT / r["file_path"] for _, r in eval_df[eval_df["label"] != "explosion"].iterrows() if (ROOT / r["file_path"]).exists()]

print("\n" + "=" * 75)
print(f"TEST 2: HISTORICAL BENCHMARK ({len(pos_files)} Positives, {len(neg_files)} Negatives)")
print("=" * 75)
for sq in [-38.0, -35.0, -32.0]:
    tp_cnt, pos_res = run_simulation(pos_files, squelch=sq)
    fp_cnt, neg_res = run_simulation(neg_files, squelch=sq)
    rec = tp_cnt / len(pos_files)
    prec = tp_cnt / (tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    # Check demolition 08
    demo08 = next((r for r in pos_res if "bbc_demolition_07022508" in r["filename"]), None)
    demo07 = next((r for r in pos_res if "bbc_demolition_07022507" in r["filename"]), None)
    
    print(f"\nSquelch {sq} dBFS:")
    print(f"  Recall     : {rec*100:.2f}% ({tp_cnt}/{len(pos_files)})")
    print(f"  False Pos  : {fp_cnt}/{len(neg_files)}")
    print(f"  Precision  : {prec*100:.2f}%")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  Demolition 08 Detected: {demo08['alerted'] if demo08 else 'Missing'}")
    print(f"  Demolition 07 Detected: {demo07['alerted'] if demo07 else 'Missing'}")

# 3. Test on Speech Files
speech_df = manifest[(manifest["category"].str.contains("speech|shout|conversation", case=False, na=False)) & (manifest["split"].isin(["val", "holdout", "test", "challenge"]))]
speech_files = [ROOT / r["file_path"] for _, r in speech_df.iterrows() if (ROOT / r["file_path"]).exists()]
if speech_files:
    print("\n" + "=" * 75)
    print(f"TEST 3: SPEECH / VOCAL FILES ({len(speech_files)} Files)")
    print("=" * 75)
    for sq in [-38.0, -35.0, -32.0]:
        n_sp, sp_res = run_simulation(speech_files, squelch=sq)
        print(f"Squelch {sq} dBFS: {n_sp}/{len(speech_files)} speech files alerted")
