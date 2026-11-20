import pandas as pd
from pathlib import Path

LAB_DIR = Path("D:/NOESIS/Explosion_Model_Lab")
df = pd.read_csv(LAB_DIR / "dataset_manifest_phase21_expanded.csv")

new_nt = df[df["reviewer_id"] == "nontarget_expansion_pipeline"]

for cat in ["gunfire", "construction_impacts", "vehicle_backfires", "thunder"]:
    sub = new_nt[new_nt["category"] == cat]
    print(f"\nCategory: {cat} ({len(sub)} recordings)")
    for idx, (_, r) in enumerate(sub.head(5).iterrows()):
        print(f"  {idx+1}. {r['original_recording_id']}: {r['notes']} | dur={r['duration_seconds']}s | lic={r['license']}")
