import pandas as pd
from pathlib import Path
import soundfile as sf
import numpy as np

LAB_DIR = Path("D:/NOESIS/Explosion_Model_Lab")
PROD_DIR = Path("D:/NOESIS/Explosion/Explosion")
df = pd.read_csv(LAB_DIR / "dataset_manifest_phase21_expanded.csv")

exp_train = df[(df["split"] == "train") & (df["label"] == "explosion") & (df["training_approved"].astype(str).str.lower() == "true")]
print(f"Total approved explosion training recordings: {len(exp_train)}")

peak_energies = []
quiet_count = 0

for _, r in exp_train.iterrows():
    p = LAB_DIR / r["file_path"]
    if not p.exists():
        p = PROD_DIR / r["file_path"]
    sig, sr = sf.read(str(p), dtype="float32")
    if sig.ndim > 1:
        sig = np.mean(sig, axis=1)
    peak_rms = float(np.max([np.sqrt(np.mean(sig[i:i+22050]**2)) for i in range(0, max(1, len(sig)-22050+1), 5512)]))
    db = 20.0 * np.log10(max(peak_rms, 1e-9))
    peak_energies.append(db)
    if db < -35.0:
        quiet_count += 1

peak_arr = np.array(peak_energies)
print(f"Peak Window RMS (dBFS) across 277 Explosion recordings:")
print(f"  Mean: {peak_arr.mean():.2f} dBFS | Min: {peak_arr.min():.2f} dBFS | Max: {peak_arr.max():.2f} dBFS")
print(f"  Distant / Quiet Blasts (< -35 dBFS): {quiet_count} recordings ({quiet_count/len(exp_train)*100:.1f}%)")
print(f"  Very Quiet Blasts (< -45 dBFS): {np.sum(peak_arr < -45.0)} recordings")
