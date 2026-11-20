import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.file_loader import load_audio_file
from audio.preprocessing import AudioPreprocessor
from audio.microphone_stream import StreamingHighpassFilter
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN
from inference.tracker_verified import AcousticVerifiedEventTracker

MODEL_PATH = ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
FILES = [
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec07.m4a",
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec08.m4a",
    ROOT / "data" / "phase24_approved" / "room_ambient" / "p24_room_ambient_ac_rec09.m4a",
]

# Load model
device = torch.device("cpu")
model = ExplosionBinaryCNN()
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()

preprocessor = AudioPreprocessor(sample_rate=22050, window_duration_sec=1.0, hop_duration_sec=0.25)

print("=" * 80)
print("DIAGNOSTIC ANALYSIS OF PHASE 24 ROOM AMBIENT RECORDINGS")
print(f"Model: {MODEL_PATH.name} (ExpF)")
print("=" * 80)

for fpath in FILES:
    print(f"\nAnalyzing: {fpath.name}")
    raw_audio, sr, dur, ch = load_audio_file(fpath)
    
    # Preprocess: convert to mono and resample to 22050 Hz
    mono_audio = preprocessor.to_mono(raw_audio)
    resampled_audio = preprocessor.resample(mono_audio, orig_sr=sr)
    
    # 35 Hz highpass filter (production filter)
    hp_filter = StreamingHighpassFilter(cutoff_hz=35.0, sample_rate=22050)
    filtered_audio = hp_filter.process(resampled_audio)
    
    # Windowing: 1.0s window, 0.25s hop
    win_len = 22050
    hop_len = int(0.25 * 22050)
    
    tracker = AcousticVerifiedEventTracker(
        squelch_dbfs=-38.0,
        conf_threshold=0.70,
        min_lf_ratio=0.40,
        max_hf_ratio=0.08,
        demo_threshold=0.62,
    )
    
    num_windows = (len(filtered_audio) - win_len) // hop_len + 1
    alerts = []
    
    print(f"  Duration: {dur:.2f}s | Sample rate: {sr} -> 22050 Hz | Windows: {num_windows}")
    print(f"  {'Win':<4} {'Time':<6} {'RawRMS':<8} {'FiltRMS':<8} {'p_exp':<7} {'LF':<6} {'HF':<6} {'SubRng':<7} {'Crest':<6} {'Action':<15} {'Reason'}")
    print("  " + "-" * 95)
    
    for i in range(num_windows):
        start_idx = i * hop_len
        end_idx = start_idx + win_len
        window = filtered_audio[start_idx:end_idx]
        raw_win = resampled_audio[start_idx:end_idx]
        time_sec = start_idx / 22050.0
        
        rms_lin = float(np.sqrt(np.mean(window ** 2)))
        raw_rms = 20.0 * np.log10(max(float(np.sqrt(np.mean(raw_win ** 2))), 1e-9))
        filt_rms = 20.0 * np.log10(max(rms_lin, 1e-9))
        peak = float(np.max(np.abs(window)))
        
        # Log-mel spectrogram
        spec = preprocessor.compute_log_mel_spectrogram(window)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
        
        with torch.no_grad():
            logits = model(spec_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            p_nt, p_exp = probs[0], probs[1]
            
        # Acoustic metrics matching microphone_stream.py
        fft_mag = np.abs(np.fft.rfft(window))
        freqs = np.fft.rfftfreq(len(window), d=1.0 / 22050.0)
        audible_mask = (freqs >= 20.0)
        tot_e = np.sum(fft_mag[audible_mask] ** 2) + 1e-9
        lf_ratio = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)] ** 2) / tot_e)
        hf_ratio = float(np.sum(fft_mag[freqs > 4000.0] ** 2) / tot_e)
        sub_bass_ratio = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 200.0)] ** 2) / tot_e)
        crest = peak / max(rms_lin, 1e-9)
        
        sub_len = len(window) // 4
        sub_rms = [float(np.sqrt(np.mean(window[s*sub_len:(s+1)*sub_len]**2))) for s in range(4)]
        sub_db = [20.0 * np.log10(max(r, 1e-9)) for r in sub_rms]
        subframe_range = max(sub_db) - min(sub_db)
        
        res = tracker.process_window_with_acoustics(
            time_sec=time_sec,
            rms_dbfs=filt_rms,
            p_nt=float(p_nt),
            p_exp=float(p_exp),
            peak=peak,
            lf_ratio=lf_ratio,
            hf_ratio=hf_ratio,
            crest=crest,
            sub_bass_ratio=sub_bass_ratio,
            subframe_range=subframe_range,
        )
        
        action = res.get("action", "none")
        reason = res.get("reason", "")
        if res.get("alert"):
            alerts.append((time_sec, p_exp, filt_rms, reason))
            
        is_highlight = p_exp >= 0.50 or res.get("alert") is not None
        mark = "!" if is_highlight else " "
        
        print(f" {mark}{i:<3} {time_sec:5.2f}s {raw_rms:7.1f} {filt_rms:7.1f} {p_exp:6.3f} {lf_ratio:5.3f} {hf_ratio:5.3f} {subframe_range:6.2f} {crest:5.2f} {action:<15} {reason[:35]}")

    print(f"\n  Total Alerts Dispatched for {fpath.name}: {len(alerts)}")
    for a in alerts:
        print(f"    ALERT at t={a[0]:.2f}s: p_exp={a[1]:.3f}, RMS={a[2]:.1f} dBFS, Reason: {a[3]}")
