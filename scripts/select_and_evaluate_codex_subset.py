import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch
import json
import time

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

CODEX_DIR = Path("D:/NOESIS/Codex_Evaluation_Dataset")
manifest_path = CODEX_DIR / "manifest.csv"

# Load training manifest to guarantee zero overlap
train_manifest = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv")
train_filenames = set(Path(f).name for f in train_manifest["file_path"].dropna())
train_groups = set(train_manifest["group_id"].dropna())

codex_df = pd.read_csv(manifest_path, low_memory=False)

# Filter for ready_for_validation and not quarantined
valid_codex = codex_df[(codex_df["status"] == "ready_for_validation") & (codex_df["status"] != "quarantined")].copy()

# Ensure file exists on disk and no overlap with training data
existing_valid = []
for idx, r in valid_codex.iterrows():
    p = CODEX_DIR / r["audio_path"]
    if not p.exists():
        continue
    fname = p.name
    if fname in train_filenames:
        continue
    gid = r.get("uploader_group_id") or r.get("source_session_id")
    if gid in train_groups:
        continue
    existing_valid.append(r)

df_candidates = pd.DataFrame(existing_valid)
print(f"Verified candidate pool (0% training overlap, on-disk): {len(df_candidates)} files")

# Deterministic sampling per category
np.random.seed(42)

selected_records = []

# 1. Genuine Explosions: 20 from surface_high_explosive
exp_pool = df_candidates[df_candidates["category"] == "surface_high_explosive"]
selected_exp = exp_pool.sample(n=min(20, len(exp_pool)), random_state=42)
selected_records.append(selected_exp)

# 2. Gunfire: up to 20 (16 available)
gun_pool = df_candidates[df_candidates["category"] == "gunfire"]
selected_gun = gun_pool.sample(n=min(20, len(gun_pool)), random_state=42)
selected_records.append(selected_gun)

# 3. Vehicle Backfires: 10 from engines
eng_pool = df_candidates[df_candidates["category"] == "engines"]
selected_eng = eng_pool.sample(n=min(10, len(eng_pool)), random_state=42)
selected_records.append(selected_eng)

# 4. Thunder: 10 from thunder
thun_pool = df_candidates[df_candidates["category"] == "thunder"]
selected_thun = thun_pool.sample(n=min(10, len(thun_pool)), random_state=42)
selected_records.append(selected_thun)

# 5. Industrial & Mechanical Impacts: 10 from mechanical_impulses & construction_impacts
mech_pool = df_candidates[df_candidates["category"].isin(["mechanical_impulses", "construction_impacts"])]
selected_mech = mech_pool.sample(n=min(10, len(mech_pool)), random_state=42)
selected_records.append(selected_mech)

# 6. Speech, Shouting & Everyday Ambience: 20 (10 speech, 5 environment, 5 household)
speech_pool = df_candidates[df_candidates["category"] == "speech"]
env_pool = df_candidates[df_candidates["category"] == "environment"]
house_pool = df_candidates[df_candidates["category"] == "household"]

selected_speech = speech_pool.sample(n=min(10, len(speech_pool)), random_state=42)
selected_env = env_pool.sample(n=min(5, len(env_pool)), random_state=42)
selected_house = house_pool.sample(n=min(5, len(house_pool)), random_state=42)

selected_records.extend([selected_speech, selected_env, selected_house])

subset_df = pd.concat(selected_records, ignore_index=True)
print(f"\nFinal Selected Codex Evaluation Subset: {len(subset_df)} files")
print(subset_df["category"].value_counts())

# Save selection manifest for reproducibility
subset_save_path = PROJECT_ROOT / "reports" / "codex_selected_eval_subset.csv"
subset_df[["record_id", "category", "proposed_binary_label", "audio_path", "filename", "duration_seconds"]].to_csv(subset_save_path, index=False)
print(f"Saved subset manifest to: {subset_save_path}")

# Load ExpF model
ckpt_path = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

prep = AudioPreprocessor(sample_rate=22050)

print("\n" + "=" * 75)
print("EVALUATING UNCHANGED EXPF PIPELINE ON CODEX EVALUATION SUBSET")
print("=" * 75)

results = []
fp_details = []
fn_details = []

t0 = time.time()
for idx, r in subset_df.iterrows():
    p = CODEX_DIR / r["audio_path"]
    cat = r["category"]
    is_pos = (cat == "surface_high_explosive")
    
    sig, sr, dur, ch = load_audio_file(p)
    mono = prep.to_mono(sig)
    if sr != 22050:
        mono = prep.resample(mono, orig_sr=sr)
    if len(mono) < 22050:
        mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")
        
    tracker = AcousticVerifiedEventTracker(squelch_dbfs=-45.0, conf_threshold=0.70, min_lf_ratio=0.35, max_hf_ratio=0.15)
    n_windows = max(1, (len(mono) - 22050) // 5512 + 1)
    
    file_dispatched = 0
    file_alerted = False
    max_p = 0.0
    trig_info = None
    
    with torch.no_grad():
        for i in range(n_windows):
            w = mono[i * 5512 : i * 5512 + 22050]
            win_time = i * (5512 / 22050)
            rms = np.sqrt(np.mean(w**2) + 1e-12)
            dbfs = float(20.0 * np.log10(rms + 1e-12))
            peak = float(np.max(np.abs(w)))

            if dbfs < -45.0:
                dec = tracker.process_window_with_acoustics(win_time, dbfs, 1.0, 0.0, 0.0, peak, 1.0, 0.0)
            else:
                spec = prep.compute_log_mel_spectrogram(w)
                t_in = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
                logits = model(t_in)
                probs = torch.softmax(logits, dim=-1).squeeze().numpy()
                p_exp = float(probs[1])

                fft_mag = np.abs(np.fft.rfft(w))
                freqs = np.fft.rfftfreq(len(w), d=1.0 / 22050)
                audible_mask = (freqs >= 20.0)
                tot_e = np.sum(fft_mag[audible_mask] ** 2) + 1e-9
                lf = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
                hf = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)

                if p_exp > max_p:
                    max_p = p_exp

                dec = tracker.process_window_with_acoustics(win_time, dbfs, float(probs[0]), p_exp, 0.0, peak, lf, hf)

                if dec.get("action") == "initial_alert" and p_exp >= 0.70:
                    file_dispatched += 1
                    file_alerted = True
                    if trig_info is None:
                        trig_info = {
                            "time": win_time,
                            "p_exp": p_exp,
                            "rms": dbfs,
                            "lf_ratio": lf,
                            "hf_ratio": hf,
                            "action": dec.get("action")
                        }

    results.append({
        "record_id": r["record_id"],
        "filename": p.name,
        "category": cat,
        "is_pos": is_pos,
        "alerted": file_alerted,
        "dispatched_count": file_dispatched,
        "max_p": max_p,
        "trig_info": trig_info
    })
    
    if not is_pos and file_alerted:
        fp_details.append({
            "category": cat,
            "filename": p.name,
            "max_p": max_p,
            "trig": trig_info
        })
    elif is_pos and not file_alerted:
        fn_details.append({
            "filename": p.name,
            "max_p": max_p
        })

elapsed = time.time() - t0

# Metrics computation
tp = sum(1 for r in results if r["is_pos"] and r["alerted"])
fn = sum(1 for r in results if r["is_pos"] and not r["alerted"])
fp = sum(1 for r in results if not r["is_pos"] and r["alerted"])
tn = sum(1 for r in results if not r["is_pos"] and not r["alerted"])

rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
acc = (tp + tn) / len(results)

print("\n" + "=" * 75)
print("EXPF PERFORMANCE ON UNTOUCHED CODEX EVALUATION DATASET:")
print("=" * 75)
print(f"Total Files Evaluated : {len(results)}")
print(f"Explosion Recall      : {rec*100:.2f}% ({tp}/{tp+fn})")
print(f"Explosion Precision   : {prec*100:.2f}% ({tp}/{tp+fp})")
print(f"F1-Score              : {f1*100:.2f}%")
print(f"Overall Accuracy      : {acc*100:.2f}% ({tp+tn}/{len(results)})")
print(f"False-Positive Files  : {fp}/{tn+fp} (Rejection Rate: {tn/(tn+fp)*100:.2f}%)")
print(f"Missed Explosions     : {fn}/{tp+fn}")
print(f"Evaluation Latency    : {elapsed:.2f}s ({elapsed/len(results)*1000:.1f} ms/file)")

print("\nFalse-Positive Breakdown by Category:")
cat_fp_counts = {}
for item in fp_details:
    cat = item["category"]
    cat_fp_counts[cat] = cat_fp_counts.get(cat, 0) + 1

for cat, cnt in sorted(cat_fp_counts.items(), key=lambda x: -x[1]):
    print(f"  - {cat:<22}: {cnt} false alerts")

print("\nDetailed False-Positive Triggers:")
print(f"{'Category':<18} | {'Filename':<35} | {'P(exp)':<7} | {'RMS (dBFS)':<10} | {'LF':<6} | {'HF':<6}")
print("-" * 88)
for item in fp_details:
    tr = item["trig"]
    print(f"{item['category']:<18} | {item['filename'][:35]:<35} | {tr['p_exp']:.4f}  | {tr['rms']:<10.1f} | {tr['lf_ratio']:.3f}  | {tr['hf_ratio']:.3f}")

if fn_details:
    print("\nMissed Explosions:")
    for item in fn_details:
        print(f"  - {item['filename']}: Max P(exp) = {item['max_p']:.4f}")
