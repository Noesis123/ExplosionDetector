import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.test_physical_verification_rules import file_windows


def rule(w):
    return (
        w["rms_dbfs"] >= -45.0
        and w["p_exp"] >= 0.70
        and w["lf"] >= 0.40
        and w["hf"] <= 0.08
        and w["crest"] >= 3.0
        and not (w["rms_dbfs"] < -42.0 and w["p_exp"] < 0.75)
    )

print("Remaining 15 FP files:")
for fname, data in file_windows.items():
    if not data["is_pos"]:
        for w in data["windows"]:
            if rule(w):
                print(f"  {fname:<35} | {data['category']:<16} | p={w['p_exp']:.3f} | LF={w['lf']:.3f} | HF={w['hf']:.3f} | Crest={w['crest']:.2f} | Rise={w['rise_ms']:.1f}ms | SubBass={w['sub_bass']:.3f}")
                break
