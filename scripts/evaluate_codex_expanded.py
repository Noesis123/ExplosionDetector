import sys
from pathlib import Path
import time
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

TARGET_SR = 22050
WINDOW_SAMPLES = 22050
HOP_SAMPLES = 5512

def evaluate_codex_manifest(config_name, squelch_dbfs, conf_threshold, demo_thresh, min_subbass=None, max_crest=12.0):
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "codex_expanded_eval_manifest.csv"
    df = pd.read_csv(manifest_path)
    
    ckpt_path = PROJECT_ROOT / "explosion_detection" / "models" / "checkpoints" / "candidate_expF_accuracy_optimized_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = ExplosionBinaryCNN(freeze_bn=True)
    model.load_state_dict(sd)
    model.eval()
    
    preprocessor = AudioPreprocessor(sample_rate=TARGET_SR)
    
    tp, fp, tn, fn = 0, 0, 0, 0
    fp_categories = {}
    lost_explosions = []
    
    t0 = time.time()
    for idx, r in df.iterrows():
        p = Path(r["absolute_audio_path"])
        if not p.exists():
            print(f"Warning: not found {p}")
            continue
        is_pos = (r["eval_binary_label"] == "explosion")
        cat = r["category"]
        
        sig, sr, dur, ch = load_audio_file(p)
        mono = preprocessor.to_mono(sig)
        if sr != TARGET_SR:
            mono = preprocessor.resample(mono, orig_sr=sr)
        if len(mono) < WINDOW_SAMPLES:
            mono = np.pad(mono, (0, WINDOW_SAMPLES - len(mono)), mode="constant")
            
        tracker = AcousticVerifiedEventTracker(
            squelch_dbfs=squelch_dbfs,
            conf_threshold=conf_threshold,
            demo_threshold=demo_thresh,
            max_crest_factor=max_crest
        )
        
        n_windows = max(1, (len(mono) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
        alerted = False
        
        with torch.no_grad():
            for i in range(n_windows):
                w = mono[i * HOP_SAMPLES : i * HOP_SAMPLES + WINDOW_SAMPLES]
                win_time = i * (HOP_SAMPLES / TARGET_SR)
                rms = np.sqrt(np.mean(w**2) + 1e-12)
                dbfs = float(20.0 * np.log10(rms + 1e-12))
                peak = float(np.max(np.abs(w)))
                crest = peak / max(rms, 1e-9)
                
                if dbfs < squelch_dbfs:
                    decision = tracker.process_window_with_acoustics(
                        time_sec=win_time,
                        rms_dbfs=dbfs,
                        p_nt=1.0,
                        p_exp=0.0,
                        p_fw=0.0,
                        peak=peak,
                        lf_ratio=1.0,
                        hf_ratio=0.0,
                        crest=1.0,
                    )
                else:
                    spec = preprocessor.compute_log_mel_spectrogram(w)
                    t_in = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
                    probs = torch.softmax(model(t_in), dim=-1).squeeze().numpy()
                    p_exp = float(probs[1])
                    p_nt = float(probs[0])
                    
                    fft_mag = np.abs(np.fft.rfft(w))
                    freqs = np.fft.rfftfreq(len(w), d=1.0 / TARGET_SR)
                    tot_e = np.sum(fft_mag ** 2) + 1e-9
                    lf_ratio = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 1000.0)] ** 2) / tot_e)
                    hf_ratio = float(np.sum(fft_mag[freqs > 4000.0] ** 2) / tot_e)
                    sub_bass = float(np.sum(fft_mag[(freqs >= 20.0) & (freqs < 200.0)] ** 2) / tot_e)
                    
                    if min_subbass is not None and sub_bass < min_subbass:
                        # Skip trigger window if subbass insufficient
                        p_exp = 0.0
                        
                    decision = tracker.process_window_with_acoustics(
                        time_sec=win_time,
                        rms_dbfs=dbfs,
                        p_nt=p_nt,
                        p_exp=p_exp,
                        p_fw=0.0,
                        peak=peak,
                        lf_ratio=lf_ratio,
                        hf_ratio=hf_ratio,
                        crest=crest,
                    )
                    
                if decision.get("action") == "initial_alert":
                    alerted = True
                    break
                    
        if is_pos:
            if alerted:
                tp += 1
            else:
                fn += 1
                lost_explosions.append(p.name)
        else:
            if alerted:
                fp += 1
                fp_categories[cat] = fp_categories.get(cat, 0) + 1
            else:
                tn += 1
                
    elapsed = time.time() - t0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"\n=======================================================")
    print(f"CONFIGURATION: {config_name}")
    print(f"=======================================================")
    print(f"Evaluated: {len(df)} recordings in {elapsed:.1f}s ({elapsed/len(df)*1000:.1f} ms/file)")
    print(f"Explosion Recall:    {rec*100:.2f}% ({tp}/{tp+fn})")
    print(f"Explosion Precision: {prec*100:.2f}% ({tp}/{tp+fp})")
    print(f"F1-Score:            {f1*100:.2f}%")
    print(f"Accuracy:            {acc*100:.2f}% ({tp+tn}/{len(df)})")
    print(f"False Positives:     {fp}/200 ({fp/200*100:.1f}%)")
    print(f"False Positive breakdown:")
    for c, cnt in sorted(fp_categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {c:<28}: {cnt}")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "rec": rec, "prec": prec, "f1": f1, "acc": acc, "fp_categories": fp_categories}

if __name__ == "__main__":
    print("RUNNING COMPARATIVE EVALUATION ON 300-FILE EXPANDED CODEX DATASET")
    # 1. Active Baseline ExpF + AEV Config C
    res_c = evaluate_codex_manifest(
        "ExpF + AEV Config C (Baseline: squelch=-45.0, thresh=0.70)",
        squelch_dbfs=-45.0,
        conf_threshold=0.70,
        demo_thresh=1.0,
        min_subbass=None,
        max_crest=12.0
    )
    
    # 2. ExpF + Config D (Previous with 0.66 demo shortcut)
    res_d_prev = evaluate_codex_manifest(
        "ExpF + AEV Config D (Previous: squelch=-42.0, thresh=0.70, demo_thresh=0.66)",
        squelch_dbfs=-42.0,
        conf_threshold=0.70,
        demo_thresh=0.66,
        min_subbass=None,
        max_crest=12.0
    )
    
    # 3. ExpF + Config D Clean / Enhanced (squelch=-42.0, thresh=0.70, no demo shortcut, subbass>=0.045, crest<=9.5)
    res_d_clean = evaluate_codex_manifest(
        "ExpF + AEV Config D Enhanced (Production: squelch=-42.0, thresh=0.70, SubBass>=0.045, Crest<=9.5)",
        squelch_dbfs=-42.0,
        conf_threshold=0.70,
        demo_thresh=1.0,
        min_subbass=0.045,
        max_crest=9.5
    )
