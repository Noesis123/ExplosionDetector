import pandas as pd
from pathlib import Path

LAB_DIR = Path("D:/NOESIS/Explosion_Model_Lab")
df = pd.read_csv(LAB_DIR / "dataset_manifest_phase21_expanded.csv")

new_nt = df[df["reviewer_id"] == "nontarget_expansion_pipeline"]

suspicious = []
for idx, r in new_nt.iterrows():
    title = str(r['notes']).lower()
    user = str(r['group_id']).lower()
    # Check for synth / sci-fi / fantasy / fake
    if any(w in title or w in user for w in ['plasma', 'synth', 'laser', 'alien', 'fantasy', 'sci-fi', 'scifi', 'cartoon', '8-bit', 'retro']):
        suspicious.append((r['original_recording_id'], r['notes'], r['category']))

print(f"Total suspicious synthetic files found: {len(suspicious)}")
for sid, notes, cat in suspicious:
    print(f"  {sid}: {notes} | cat={cat}")
