import pickle

with open("experiments/cached_92_file_windows.pkl", "rb") as f:
    file_windows = pickle.load(f)

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

print("Maximum confidence and feature stats for each FP:")
for fname in fps:
    cat = file_windows[fname]["category"]
    wins = file_windows[fname]["windows"]
    # Sort by p_exp descending
    qualifying = [w for w in wins if w["rms_dbfs"] >= -42.0 and w["lf"] >= 0.40 and w["hf"] <= 0.08]
    if qualifying:
        qualifying.sort(key=lambda w: w["p_exp"], reverse=True)
        top = qualifying[0]
        # count how many windows have p_exp >= 0.70
        count_70 = sum(1 for w in qualifying if w["p_exp"] >= 0.70)
        print(f"{fname:<32} ({cat:<17}) | max_p={top['p_exp']:.4f} | rms={top['rms_dbfs']:.1f} | LF={top['lf']:.3f} | HF={top['hf']:.4f} | SubBass={top['sub_bass']:.3f} | Crest={top['crest']:.2f} | count_ge_70={count_70}")
