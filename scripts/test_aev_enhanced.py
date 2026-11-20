import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.test_physical_verification_rules import file_windows

def test_aev_enhanced(w):
    # Base squelch and threshold
    if w["rms_dbfs"] < -45.0 or w["p_exp"] < 0.70:
        return False
        
    # Standard physics bounds
    if w["lf"] < 0.40 or w["hf"] > 0.08:
        return False
        
    # Glitch / Click rejection (extreme crest factor)
    if w["crest"] < 2.5 or w["crest"] > 12.0:
        return False
        
    # Near-squelch fluctuation rejection
    if w["rms_dbfs"] < -42.0 and w["p_exp"] < 0.75:
        return False
        
    # Confidence-Adaptive Spectral Low-Frequency Requirement:
    # If confidence is marginal (0.70 <= p < 0.78), genuine explosions (demolition rumble, quarry blasts)
    # always display deep acoustic coupling (LF >= 0.65). Low-LF negatives (transports, distant pops) are rejected.
    if w["p_exp"] < 0.78 and w["lf"] < 0.65:
        return False
        
    return True

tp, fp, tn, fn = 0, 0, 0, 0
lost_pos = []
remaining_fps = []

for fname, data in file_windows.items():
    is_pos = data["is_pos"]
    alerted = False
    
    for w in data["windows"]:
        if test_aev_enhanced(w):
            alerted = True
            break
            
    if is_pos:
        if alerted:
            tp += 1
        else:
            fn += 1
            lost_pos.append(fname)
    else:
        if alerted:
            fp += 1
            remaining_fps.append((fname, data["category"]))
        else:
            tn += 1

rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
acc = (tp + tn) / (tp + tn + fp + fn)

print("=" * 60)
print("ENHANCED AEV (CONFIG D) RESULTS ON HISTORICAL BENCHMARK:")
print("=" * 60)
print(f"Recall     : {rec*100:.2f}% ({tp}/{tp+fn})")
print(f"Precision  : {prec*100:.2f}% ({tp}/{tp+fp})")
print(f"F1-Score   : {f1*100:.2f}%")
print(f"Accuracy   : {acc*100:.2f}% ({(tp+tn)}/{len(file_windows)})")
print(f"False Pos  : {fp}/{tn+fp} files (Suppressed: {19 - fp} files)")
print(f"Lost Pos   : {len(lost_pos)} files: {lost_pos}")
print("-" * 60)
print(f"Remaining False Positives ({len(remaining_fps)}):")
for f, cat in remaining_fps:
    print(f"  - {f:<35}: {cat}")
print("=" * 60)
