import pickle
import numpy as np

with open("experiments/cached_92_file_windows.pkl", "rb") as f:
    file_windows = pickle.load(f)

thunder_files = ['fs_36044_inchadney.mp3', 'fs_195529_klankbeeld.mp3']
for tf in thunder_files:
    cat = file_windows[tf]["category"]
    wins = file_windows[tf]["windows"]
    trig_wins = [w for w in wins if w["p_exp"] >= 0.70 and w["rms_dbfs"] >= -42.0]
    print(f"\nThunder {tf} ({cat}): total trigs={len(trig_wins)}")
    for i, w in enumerate(trig_wins[:6]):
        print(f"  win {i}: t={w['time_sec']}s, p={w['p_exp']:.3f}, rms={w['rms_dbfs']:.1f}, LF={w['lf']:.3f}, SubBass={w['sub_bass']:.3f}, Crest={w['crest']:.2f}, Rise={w['rise_ms']:.1f}ms")
