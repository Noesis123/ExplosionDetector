import pickle

with open("experiments/cached_92_file_windows.pkl", "rb") as f:
    file_windows = pickle.load(f)

def test_rule(name, rule_fn):
    tp, fp = 0, 0
    fps, lost = [], []
    for fname, d in file_windows.items():
        trigs = [w for w in d["windows"] if rule_fn(w)]
        if d["is_pos"]:
            if trigs: tp += 1
            else: lost.append(fname)
        else:
            if trigs:
                fp += 1
                fps.append((fname, d["category"]))
    rec = tp / 24.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"=== {name} ===")
    print(f"TP: {tp}/24 ({rec*100:.1f}%) | FP: {fp}/68 | Prec: {prec*100:.2f}% | F1: {f1*100:.2f}%")
    if lost: print(f"  Lost: {lost}")
    print(f"  FPs ({len(fps)}): {[f'{fn} ({cat})' for fn, cat in fps]}")
    return tp, fp, prec, f1

def base_clean(w):
    if w["rms_dbfs"] < -42.0: return False
    if w["p_exp"] < 0.70: return False
    if w["lf"] < 0.40 or w["hf"] > 0.08: return False
    if w["crest"] < 2.5 or w["crest"] > 9.5: return False
    if w["rms_dbfs"] < -39.0 and w["p_exp"] < 0.75: return False
    if w["p_exp"] < 0.78 and w["lf"] < 0.65: return False
    if w["sub_bass"] < 0.045: return False
    return True

test_rule("Base clean + SubBass>=0.045 + Crest<=9.5", base_clean)

# Rule A: If RMS < -26 dBFS, require SubBass >= 0.45
def rule_a(w):
    if not base_clean(w): return False
    if w["rms_dbfs"] < -26.0 and w["sub_bass"] < 0.45:
        return False
    return True

test_rule("Rule A: RMS < -26 -> SubBass >= 0.45", rule_a)
