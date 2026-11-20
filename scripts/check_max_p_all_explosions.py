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

prep = AudioPreprocessor(sample_rate=22050)
manifest_df = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "dataset_manifest_phase21_expanded.csv")
bench_df = manifest_df[manifest_df["split"].isin(["val", "holdout", "test", "challenge"])].copy()

ckpt_f = torch.load(PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt", map_location="cpu", weights_only=False)
sd_f = ckpt_f["model_state_dict"] if "model_state_dict" in ckpt_f else ckpt_f
model_f = ExplosionBinaryCNN(freeze_bn=True)
model_f.load_state_dict(sd_f)
model_f.eval()

pos_df = bench_df[bench_df["label"] == "explosion"]

print("=" * 80)
print("MAXIMUM P_EXP ACROSS ALL 24 BENCHMARK EXPLOSIONS")
print("=" * 80)

max_p_list = []
for idx, r in pos_df.iterrows():
    p = PROJECT_ROOT / r["file_path"]
    if not p.exists(): continue
    sig, sr, dur, ch = load_audio_file(p)
    mono = prep.to_mono(sig)
    if sr != 22050: mono = prep.resample(mono, orig_sr=sr)
    if len(mono) < 22050: mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")
    
    nw = max(1, (len(mono) - 22050) // 5512 + 1)
    max_p = 0.0
    best_w = None
    for i in range(nw):
        w = mono[i * 5512 : i * 5512 + 22050]
        rms = np.sqrt(np.mean(w**2) + 1e-12)
        dbfs = float(20.0 * np.log10(rms + 1e-12))
        if dbfs < -34.0: continue
        
        spec = prep.compute_log_mel_spectrogram(w)
        tin = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            p_exp = float(torch.softmax(model_f(tin), dim=-1)[0, 1])
        if p_exp > max_p:
            max_p = p_exp
            best_w = (dbfs, p_exp)
    max_p_list.append((max_p, p.name, best_w))

max_p_list.sort(key=lambda x: x[0])
for mp, name, bw in max_p_list:
    print(f"Max p={mp:.4f} | RMS={bw[0] if bw else -99.9:5.1f} dBFS | {name}")
