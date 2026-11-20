import pickle

with open("experiments/cached_92_file_windows.pkl", "rb") as f:
    file_windows = pickle.load(f)

print(f"{'Thresh':<8} | {'TP':<6} | {'FP':<6} | {'Recall':<8} | {'Precision':<10} | {'F1':<8} | Missed TPs")
print("-" * 75)

for th in [0.700, 0.702, 0.705, 0.710, 0.715, 0.720, 0.725, 0.730, 0.735, 0.740, 0.750, 0.760, 0.770, 0.780]:
    tp, fp = 0, 0
    missed = []
    for fname, d in file_windows.items():
        # Clean config D rules with threshold = th
        trigs = [
            w for w in d["windows"]
            if w["rms_dbfs"] >= -42.0
            and w["p_exp"] >= th
            and w["lf"] >= 0.40
            and w["hf"] <= 0.08
            and not (w["crest"] < 2.5 or w["crest"] > 9.5)
            and not (w["rms_dbfs"] < -39.0 and w["p_exp"] < 0.75)
            and not (w["p_exp"] < 0.78 and w["lf"] < 0.65)
            and w["sub_bass"] >= 0.045
        ]
        if d["is_pos"]:
            if trigs:
                tp += 1
            else:
                missed.append(fname)
        else:
            if trigs:
                fp += 1
                
    rec = tp / 24.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"{th:<8.3f} | {tp:<6} | {fp:<6} | {rec*100:<7.2f}% | {prec*100:<9.2f}% | {f1*100:<7.2f}% | {missed}")
