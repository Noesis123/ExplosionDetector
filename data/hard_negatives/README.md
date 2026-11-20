# Hard Negatives

This directory stores audio recordings that are *not* explosions but are acoustically similar, to be used as hard negative examples during model training and evaluation.

Subfolders:
- `fireworks/`
- `thunder/`
- `gunfire/`
- `vehicle_backfires/`
- `industrial_impacts/`

Populate each folder with appropriate recordings. The presence of these subfolders allows the data pipeline to locate and include hard negative samples without modifying existing manifests.

