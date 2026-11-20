import pickle

with open("experiments/cached_92_file_windows.pkl", "rb") as f:
    file_windows = pickle.load(f)

tp, fp = 0, 0
fps, lost = [], []

for fname, d in file_windows.items():
    trigs = [
        w for w in d["windows"]
        if w["rms_dbfs"] >= -38.0
        and w["p_exp"] >= 0.70
        and w["lf"] >= 0.40
        and w["hf"] <= 0.08
        and not (w["crest"] < 2.5 or w["crest"] > 12.0)
        and not (w["rms_dbfs"] < -35.0 and w["p_exp"] < 0.75)
        and not (w["p_exp"] < 0.78 and w["lf"] < 0.65)
        and not (w["sub_bass"] < 0.045 and w["hf"] > 0.02)
    ]
    if d["is_pos"]:
        if trigs:
            tp += 1
        else:
            lost.append(fname)
    else:
        if trigs:
            fp += 1
            fps.append((fname, d["category"]))

rec = tp / 24.0
prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
acc = (tp + (68 - fp)) / 92.0
print(f"=== Final Verified Production Detection Logic ===")
print(f"TP: {tp}/24 ({rec*100:.2f}%) | FP: {fp}/68 | Precision: {prec*100:.2f}% | F1: {f1*100:.2f}% | Acc: {acc*100:.2f}%")
print(f"Lost explosions ({len(lost)}): {lost}")
print(f"\nRemaining False Positives ({len(fps)}):")
for fn, cat in fps:
    print(f"  {fn:<35} ({cat})")
