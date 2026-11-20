import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.test_physical_verification_rules import file_windows

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
    'fs_350672_Dominik_W.mp3',
    'p19_shouting_screaming_3_halgrimm_44e0bfe0.mp3'
]

print("Detail of trigger windows for 12 FPs:")
for fname in fps:
    cat = file_windows[fname]["category"]
    wins = file_windows[fname]["windows"]
    trigs = [w for w in wins if w["p_exp"] >= 0.70 and w["rms_dbfs"] >= -42.0]
    print(f"=== {fname} ({cat}) has {len(trigs)} trigs ===")
    for tw in trigs[:3]:
        print(f"   t={tw['time_sec']}s, p={tw['p_exp']:.4f}, rms={tw['rms_dbfs']:.1f}, LF={tw['lf']:.3f}, HF={tw['hf']:.4f}, SubBass={tw['sub_bass']:.3f}, Crest={tw['crest']:.2f}, Rise={tw['rise_ms']:.1f}ms")
