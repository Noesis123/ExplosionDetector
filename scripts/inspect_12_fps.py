import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.test_physical_verification_rules import file_windows

def rule_clean(w):
    if w["rms_dbfs"] < -42.0:
        return False
    p_exp = w["p_exp"]
    if p_exp < 0.70:
        return False
    if w["lf"] < 0.40 or w["hf"] > 0.08:
        return False
    if w["crest"] < 2.5 or w["crest"] > 12.0:
        return False
    if w["rms_dbfs"] < -39.0 and p_exp < 0.75:
        return False
    if p_exp < 0.78 and w["lf"] < 0.65:
        return False
    return True

print("Remaining 12 FP files under clean 0.70 threshold:")
for fname, d in file_windows.items():
    if not d["is_pos"]:
        trig_wins = [w for w in d["windows"] if rule_clean(w)]
        if trig_wins:
            w0 = trig_wins[0]
            cat = d["category"]
            print(f"  {fname:<35} | {cat:^16} | p={w0['p_exp']:.4f} | LF={w0['lf']:.3f} | HF={w0['hf']:.3f} | Crest={w0['crest']:.2f} | Rise={w0['rise_ms']:.1f}ms | SubBass={w0['sub_bass']:.3f} | dBFS={w0['rms_dbfs']:.1f}")

print("\nAll 22 True Positives under clean 0.70 threshold:")
for fname, d in file_windows.items():
    if d["is_pos"]:
        trig_wins = [w for w in d["windows"] if rule_clean(w)]
        if trig_wins:
            w0 = trig_wins[0]
            print(f"  [TP] {fname:<30} | p={w0['p_exp']:.4f} | LF={w0['lf']:.3f} | HF={w0['hf']:.3f} | Crest={w0['crest']:.2f} | Rise={w0['rise_ms']:.1f}ms | SubBass={w0['sub_bass']:.3f} | dBFS={w0['rms_dbfs']:.1f}")
        else:
            print(f"  [MISSED] {fname:<30}")
