import json
import re
from pathlib import Path
import pandas as pd

LAB_DIR = Path("D:/NOESIS/Explosion_Model_Lab")

df = pd.read_csv(LAB_DIR / "dataset_manifest_phase21_expanded.csv")
new_nt = df[df["reviewer_id"] == "nontarget_expansion_pipeline"]

# Audit criteria for rejection:
# 1. Sci-fi, plasma, energy, synthetic, EDM kick, game FX
# 2. Contains words: 'plasma', 'energy rifle', 'explosive', 'attack pack', 'robotic', 'synth', 'laser', 'sci-fi'
flagged = []
for idx, r in new_nt.iterrows():
    text = (str(r['notes']) + " " + str(r['file_path'])).lower()
    reasons = []
    if 'plasma' in text: reasons.append("plasma rifle")
    if 'energy rifle' in text: reasons.append("energy rifle")
    if 'explosive' in text: reasons.append("explosive keyword")
    if 'attack pack' in text: reasons.append("sound design pack")
    if 'robotic' in text: reasons.append("robotic fx")
    if 'game' in text or 'fx1' in text: reasons.append("sound pack fx")
    if reasons:
        flagged.append((r['original_recording_id'], r['notes'], reasons))

print(f"Total flagged for removal from new non-targets: {len(flagged)}")
for sid, notes, reasons in flagged:
    print(f"  {sid}: {notes} -> {reasons}")
