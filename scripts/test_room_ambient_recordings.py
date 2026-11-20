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
ckpt_f = torch.load(PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt", map_location="cpu", weights_only=False)
sd_f = ckpt_f["model_state_dict"] if "model_state_dict" in ckpt_f else ckpt_f
model_f = ExplosionBinaryCNN(freeze_bn=True)
model_f.load_state_dict(sd_f)
model_f.eval()

room_files = [
    PROJECT_ROOT / "data" / "user_room_ambient" / "Recording (7).m4a",
    PROJECT_ROOT / "data" / "user_room_ambient" / "Recording (8).m4a",
    PROJECT_ROOT / "data" / "user_room_ambient" / "Recording (9).m4a",
]

print("=" * 80)
print("TESTING ROOM AMBIENT RECORDINGS THROUGH NEW GATING LOGIC")
print("=" * 80)

def test_file(p):
    if not p.exists():
        # Check brain user_uploaded
        alt = list(Path("C:/Users/Aman Yazeen/.gemini/antigravity/brain/ad076906-e1fe-4046-9a40-92f5f2339702/.user_uploaded").glob("*.m4a"))
        if not alt:
            print(f"File not found: {p}")
            return
        p = alt[0]
        
    sig, sr, dur, ch = load_audio_file(p)
    mono = prep.to_mono(sig)
    if sr != 22050: mono = prep.resample(mono, orig_sr=sr)
    if len(mono) < 22050: mono = np.pad(mono, (0, 22050 - len(mono)), mode="constant")
    
    nw = max(1, (len(mono) - 22050) // 5512 + 1)
    alerts = 0
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
            
        fft_mag = np.abs(np.fft.rfft(w))
        freqs = np.fft.rfftfreq(len(w), d=1.0 / 22050)
        tot_e = np.sum(fft_mag ** 2) + 1e-9
        
        lf = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 1000)] ** 2) / tot_e)
        hf = float(np.sum(fft_mag[freqs > 4000] ** 2) / tot_e)
        sub20_80 = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 80)] ** 2) / tot_e)
        mid1k_4k = float(np.sum(fft_mag[(freqs >= 1000) & (freqs < 4000)] ** 2) / tot_e)
        sub_bass = float(np.sum(fft_mag[(freqs >= 20) & (freqs < 200)] ** 2) / tot_e)
        
        sub4 = [w[k * 5512 : (k + 1) * 5512] for k in range(4)]
        sub4_rms = [20.0 * np.log10(max(float(np.sqrt(np.mean(sf ** 2))), 1e-9)) for sf in sub4]
        subframe_range = max(sub4_rms) - min(sub4_rms)
        
        e_half1 = np.sum(w[:11025] ** 2) + 1e-9
        e_half2 = np.sum(w[11025:] ** 2) + 1e-9
        decay_win = float(e_half2 / e_half1)
        
        # Check triggering
        is_std = (p_exp >= 0.80)
        is_demolition = (
            p_exp >= 0.70
            and sub20_80 >= 0.45
            and lf >= 0.95
            and mid1k_4k <= 0.020
            and hf <= 0.010
            and crest >= 5.0
            and subframe_range >= 8.0
        )
        is_loud_detonation = (
            p_exp >= 0.71
            and dbfs >= -22.0
            and (sub20_80 >= 0.12 or subframe_range >= 12.0)
            and lf >= 0.70
            and mid1k_4k <= 0.21
            and (3.0 <= crest <= 10.0)
            and decay_win >= 0.18
            and (subframe_range >= 7.0 or (hf >= 0.04 and decay_win >= 0.60))
        )
        is_onset_blast = (
            p_exp >= 0.70
            and subframe_range >= 30.0
            and lf >= 0.90
            and crest >= 5.0
        )
        is_fragile = (is_demolition or is_loud_detonation or is_onset_blast)
        if not (is_std or is_fragile): continue
        
        # Filters:
        if dbfs < -28.0 and subframe_range < 3.5: continue
        if lf < 0.40: continue
        if hf > 0.08: continue
        if crest < 2.5 or crest > 12.0: continue
        if p_exp < 0.78 and lf < 0.65: continue
        if sub_bass < 0.045 and hf > 0.02: continue
        if dbfs < -25.0 and lf >= 0.96 and hf <= 0.010 and subframe_range < 6.0: continue
        if mid1k_4k >= 0.28 and sub20_80 <= 0.05: continue
        if dbfs < -30.0 and p_exp < 0.84 and subframe_range < 10.0: continue
        if dbfs < -25.0 and sub20_80 < 0.10: continue
        if dbfs < -20.0 and sub20_80 < 0.08 and subframe_range < 5.0: continue
        if decay_win < 0.08 and hf <= 0.010 and subframe_range < 25.0: continue
        
        alerts += 1
        print(f"  ALERT on Win {i:02d}: p={p_exp:.3f}, RMS={dbfs:.1f}, LF={lf:.3f}, HF={hf:.3f}, Sub80={sub20_80:.3f}, SubRng={subframe_range:.1f}")
        
    print(f"File {p.name}: Alerts = {alerts}")

for rf in room_files:
    test_file(rf)
