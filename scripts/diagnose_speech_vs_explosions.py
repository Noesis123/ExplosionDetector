import glob
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor

pre = AudioPreprocessor(sample_rate=22050)

exp_files = glob.glob('data/positives/*.mp3')[:15]
speech_files = glob.glob('data/phase19_approved/speech_conversation/*.mp3') + glob.glob('data/phase19_approved/mixed_speech_noise/*.mp3')

def analyze_file(p):
    sig, sr, dur, ch = load_audio_file(p)
    mono = pre.to_mono(sig)
    if sr != 22050: mono = pre.resample(mono, orig_sr=sr)
    
    hop = 5512
    win_len = 22050
    n_wins = max(1, (len(mono) - win_len) // hop + 1)
    
    best_w = None
    best_rms = -100.0
    for i in range(n_wins):
        w = mono[i*hop : i*hop + win_len]
        rms = 20.0 * np.log10(max(np.sqrt(np.mean(w**2)), 1e-9))
        if rms > best_rms:
            best_rms = rms
            best_w = w
            
    if best_w is None: return None
    
    rms = np.sqrt(np.mean(best_w**2))
    peak = np.max(np.abs(best_w))
    crest = peak / (rms + 1e-9)
    
    fft_mag = np.abs(np.fft.rfft(best_w))
    freqs = np.fft.rfftfreq(len(best_w), d=1.0/22050)
    tot_e = np.sum(fft_mag**2) + 1e-9
    
    centroid = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-9)
    e_20_150 = np.sum(fft_mag[(freqs >= 20) & (freqs < 150)]**2) / tot_e
    e_150_600 = np.sum(fft_mag[(freqs >= 150) & (freqs < 600)]**2) / tot_e
    e_600_2500 = np.sum(fft_mag[(freqs >= 600) & (freqs < 2500)]**2) / tot_e
    e_gt_2500 = np.sum(fft_mag[freqs >= 2500]**2) / tot_e
    zcr = np.mean(np.abs(np.diff(np.sign(best_w)))) / 2.0
    
    return {
        'rms_dbfs': best_rms,
        'crest': crest,
        'centroid': centroid,
        'e_20_150': e_20_150,
        'e_150_600': e_150_600,
        'e_600_2500': e_600_2500,
        'e_gt_2500': e_gt_2500,
        'zcr': zcr
    }

print('=== EXPLOSIONS ===')
exp_stats = [analyze_file(f) for f in exp_files]
exp_stats = [s for s in exp_stats if s]
print(f'Count: {len(exp_stats)}')
print(f'Crest factor: mean={np.mean([s["crest"] for s in exp_stats]):.2f}, min={min([s["crest"] for s in exp_stats]):.2f}, max={max([s["crest"] for s in exp_stats]):.2f}')
print(f'Spectral Centroid: mean={np.mean([s["centroid"] for s in exp_stats]):.1f} Hz, min={min([s["centroid"] for s in exp_stats]):.1f}, max={max([s["centroid"] for s in exp_stats]):.1f}')
print(f'Energy 20-150 Hz: mean={np.mean([s["e_20_150"] for s in exp_stats])*100:.1f}%, min={min([s["e_20_150"] for s in exp_stats])*100:.1f}%')
print(f'Energy 150-600 Hz: mean={np.mean([s["e_150_600"] for s in exp_stats])*100:.1f}%')
print(f'Energy 600-2500 Hz: mean={np.mean([s["e_600_2500"] for s in exp_stats])*100:.1f}%')
print(f'Energy >2500 Hz: mean={np.mean([s["e_gt_2500"] for s in exp_stats])*100:.1f}%')

print('\n=== SPEECH ===')
sp_stats = [analyze_file(f) for f in speech_files]
sp_stats = [s for s in sp_stats if s]
print(f'Count: {len(sp_stats)}')
print(f'Crest factor: mean={np.mean([s["crest"] for s in sp_stats]):.2f}, min={min([s["crest"] for s in sp_stats]):.2f}, max={max([s["crest"] for s in sp_stats]):.2f}')
print(f'Spectral Centroid: mean={np.mean([s["centroid"] for s in sp_stats]):.1f} Hz, min={min([s["centroid"] for s in sp_stats]):.1f}, max={max([s["centroid"] for s in sp_stats]):.1f}')
print(f'Energy 20-150 Hz: mean={np.mean([s["e_20_150"] for s in sp_stats])*100:.1f}%, max={max([s["e_20_150"] for s in sp_stats])*100:.1f}%')
print(f'Energy 150-600 Hz: mean={np.mean([s["e_150_600"] for s in sp_stats])*100:.1f}%')
print(f'Energy 600-2500 Hz: mean={np.mean([s["e_600_2500"] for s in sp_stats])*100:.1f}%')
print(f'Energy >2500 Hz: mean={np.mean([s["e_gt_2500"] for s in sp_stats])*100:.1f}%')
