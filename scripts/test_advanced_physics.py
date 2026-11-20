import pickle

with open("experiments/cached_92_file_windows.pkl", "rb") as f:
    file_windows = pickle.load(f)

# Let's inspect the entire file evaluation under AcousticVerifiedEventTracker logic
def eval_ruleset(name, rule_fn):
    tp, fp, tn, fn = 0, 0, 0, 0
    fps, lost = [], []
    for fname, d in file_windows.items():
        is_pos = d["is_pos"]
        trig_wins = [w for w in d["windows"] if rule_fn(w)]
        if is_pos:
            if trig_wins:
                tp += 1
            else:
                fn += 1
                lost.append(fname)
        else:
            if trig_wins:
                fp += 1
                fps.append((fname, d["category"]))
            else:
                tn += 1
    rec = tp / 24.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / 92.0
    print(f"\n=== {name} ===")
    print(f"TP: {tp}/24 ({rec*100:.2f}%) | FP: {fp}/68 | Precision: {prec*100:.2f}% | F1: {f1*100:.2f}% | Acc: {acc*100:.2f}%")
    if lost:
        print(f"  Lost TPs ({len(lost)}): {lost}")
    print(f"  FPs ({len(fps)}): {[f'{fn} ({cat})' for fn, cat in fps]}")
    return tp, fp, prec, f1

# Base clean config D
def base_rule(w):
    if w["rms_dbfs"] < -42.0: return False
    p = w["p_exp"]
    if p < 0.70: return False
    if w["lf"] < 0.40 or w["hf"] > 0.08: return False
    if w["crest"] < 2.5 or w["crest"] > 12.0: return False
    if w["rms_dbfs"] < -39.0 and p < 0.75: return False
    if p < 0.78 and w["lf"] < 0.65: return False
    return True

eval_ruleset("Base Clean Config D (0.70, no demo shortcut)", base_rule)

# Advanced Physics 1: Add SubBass >= 0.045
def adv_rule_1(w):
    if not base_rule(w): return False
    if w["sub_bass"] < 0.045: return False
    return True

eval_ruleset("Adv 1: + SubBass >= 0.045", adv_rule_1)

# Advanced Physics 2: Add Max Crest <= 9.5
def adv_rule_2(w):
    if not adv_rule_1(w): return False
    if w["crest"] > 9.5: return False
    return True

eval_ruleset("Adv 2: + Crest <= 9.5", adv_rule_2)

# Advanced Physics 3: If p < 0.710, require SubBass >= 0.40 (protects demolition blast p=0.7022, SubB=0.895)
def adv_rule_3(w):
    if not adv_rule_2(w): return False
    if w["p_exp"] < 0.710 and w["sub_bass"] < 0.40: return False
    return True

eval_ruleset("Adv 3: + (if p < 0.710 require SubBass >= 0.40)", adv_rule_3)

# Advanced Physics 4: Marginal Confidence & Low Frequency Tightening
# If p in [0.70, 0.75), require LF >= 0.70
def adv_rule_4(w):
    if not adv_rule_3(w): return False
    if w["p_exp"] < 0.75 and w["lf"] < 0.70: return False
    return True

eval_ruleset("Adv 4: + (if p < 0.75 require LF >= 0.70)", adv_rule_4)

# Advanced Physics 5: If p in [0.70, 0.75), require near-squelch >= -35 dBFS if SubBass < 0.50
def adv_rule_5(w):
    if not adv_rule_4(w): return False
    if w["p_exp"] < 0.75 and w["rms_dbfs"] < -35.0 and w["sub_bass"] < 0.50: return False
    return True

eval_ruleset("Adv 5: + (if p < 0.75 & RMS < -35dBFS require SubBass >= 0.50)", adv_rule_5)
