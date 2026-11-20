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

m = bench_df[bench_df["file_path"].str.contains("fs_184275_qubodup")]
p = PROJECT_ROOT / m.iloc[0]["file_path"]
sig, sr, dur, ch = load_audio_file(p)
mono = prep.to_mono(sig)
if sr != 22050: mono = prep.resample(mono, orig_sr=sr)
if len(mono) < 22050: mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")

nw = max(1, (len(mono) - 22050) // 5512 + 1)
print(f"FILE: {p.name}")
for i in range(nw):
    w = mono[i * 5512 : i * 5512 + 22050]
    rms = np.sqrt(np.mean(w**2) + 1e-12)
    dbfs = float(20.0 * np.log10(rms + 1e-12))
    peak = float(np.max(np.abs(w)))
    crest = peak / max(rms, 1e-9)
    if dbfs < -34.0: continue
    
    spec = prep.compute_log_mel_spectrogram(w)
    tin = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        p_exp = float(torch.softmax(model_f(tin), dim=-1)[0, 1])
    if p_exp < 0.65: continue
    
    fft_mag = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), d=1.0 / 22050)
    tot_e = np.sum(fft_mag ** 2) + 1e-9
    
    lf = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
    hf = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)
    sub20_80 = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 80)] ** 2) / tot_e)
    mid1k_4k = float(np.sum(fft_mag[(freqs >= 1000) & (freqs < 4000)] ** 2) / tot_e)
    
    sub4 = [w[k * 5512 : (k + 1) * 5512] for k in range(4)]
    sub4_rms = [20.0 * np.log10(max(float(np.sqrt(np.mean(sf ** 2))), 1e-9)) for sf in sub4]
    subframe_range = max(sub4_rms) - min(sub4_rms)
    
    e_half1 = np.sum(w[:11025] ** 2) + 1e-9
    e_half2 = np.sum(w[11025:] ** 2) + 1e-9
    decay_ratio_in_win = float(e_half2 / e_half1)
    
    print(f"Win {i:02d}: p={p_exp:.3f} | RMS={dbfs:.1f} | Crest={crest:.2f} | LF={lf:.3f} | HF={hf:.3f} | Mid={mid1k_4k:.3f} | Sub80={sub20_80:.3f} | SubRng={subframe_range:.1f}dB | DecWin={decay_ratio_in_win:.2f}")
