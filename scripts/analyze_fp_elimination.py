import pickle
import numpy as np

with open("experiments/cached_92_file_windows.pkl", "rb") as f:
    file_windows = pickle.load(f)

# Collect all trigger windows of 22 TPs
tp_windows = []
for fname, d in file_windows.items():
    if d["is_pos"]:
        for w in d["windows"]:
            if w["p_exp"] >= 0.70 and w["rms_dbfs"] >= -42.0:
                tp_windows.append((fname, w))

fps = [
    'bbc_gunfire---_07034194.mp3',
    'bbc_gunfire---_07034201.mp3',
    'fs_410347_inspectorj.mp3',
    'fs_36044_inchadney.mp3',
    'fs_195529_klankbeeld.mp3',
    'fs_161341_qubodup.mp3',
    'fs_171475_2887679652.mp3',
    'fs_184275_qubodup.mp3',
    'fs_213925_Diboz.mp3',
    'fs_105351_CeebFrack.mp3',
    'fs_350672_Dominik_W.mp3'
]

print("=== Comparing TPs and FPs across features ===")
tp_min_lf = min(w["lf"] for fn, w in tp_windows)
tp_max_hf = max(w["hf"] for fn, w in tp_windows)
tp_min_crest = min(w["crest"] for fn, w in tp_windows)
tp_max_crest = max(w["crest"] for fn, w in tp_windows)
tp_min_subbass = min(w["sub_bass"] for fn, w in tp_windows)
tp_min_rise = min(w["rise_ms"] for fn, w in tp_windows)
tp_max_rise = max(w["rise_ms"] for fn, w in tp_windows)

print(f"TP ranges across ALL trigger windows:")
print(f"  LF: [{tp_min_lf:.3f}, {max(w['lf'] for fn, w in tp_windows):.3f}]")
print(f"  HF: [{min(w['hf'] for fn, w in tp_windows):.4f}, {tp_max_hf:.4f}]")
print(f"  Crest: [{tp_min_crest:.2f}, {tp_max_crest:.2f}]")
print(f"  SubBass: [{tp_min_subbass:.3f}, {max(w['sub_bass'] for fn, w in tp_windows):.3f}]")
print(f"  Rise_ms: [{tp_min_rise:.2f}, {tp_max_rise:.2f}]")

# Now check for each FP, what rules could eliminate its trigger windows without violating TP constraints
print("\n=== Per-FP Trigger Window Analysis ===")
for fname in fps:
    cat = file_windows[fname]["category"]
    wins = [w for w in file_windows[fname]["windows"] if w["p_exp"] >= 0.70 and w["rms_dbfs"] >= -42.0]
    print(f"\n{fname} ({cat}) - {len(wins)} windows:")
    for w in wins[:5]:
        print(f"  t={w['time_sec']}s: p={w['p_exp']:.3f}, rms={w['rms_dbfs']:.1f}, LF={w['lf']:.3f}, HF={w['hf']:.4f}, Crest={w['crest']:.2f}, SubBass={w['sub_bass']:.3f}, Rise={w['rise_ms']:.1f}ms")
