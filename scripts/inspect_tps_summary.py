import pickle

with open("experiments/cached_92_file_windows.pkl", "rb") as f:
    file_windows = pickle.load(f)

print("Peak and minimum stats across all 22 True Positives:")
for fname, d in file_windows.items():
    if d["is_pos"]:
        wins = d["windows"]
        # Find all windows that trigger baseline
        trigs = [w for w in wins if w["p_exp"] >= 0.70 and w["rms_dbfs"] >= -42.0]
        if trigs:
            max_p_win = max(trigs, key=lambda w: w["p_exp"])
            max_rms_win = max(trigs, key=lambda w: w["rms_dbfs"])
            print(f"[TP] {fname:<32} | trigs={len(trigs):<2} | max_p={max_p_win['p_exp']:.4f} | max_rms={max_rms_win['rms_dbfs']:.1f} | LF={max_p_win['lf']:.3f} | HF={max_p_win['hf']:.4f} | SubBass={max_p_win['sub_bass']:.3f} | Crest={max_p_win['crest']:.2f}")
        else:
            print(f"[MISSED] {fname:<32}")
