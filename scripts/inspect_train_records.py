import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from explosion_detection.models.binary_cnn import ExplosionBinaryCNN

ckpt = torch.load('explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt', map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
model = ExplosionBinaryCNN(freeze_bn=True)
model.load_state_dict(sd)
model.eval()

fit = torch.load('training/data/train_fit_binary_pool.pt', map_location='cpu', weights_only=False)
records = fit['file_records']
artillery = [r for r in records if r.get('category') == 'artillery_firing']
backfires = [r for r in records if r.get('category') == 'vehicle_backfires']

print('=== 4 Training Artillery Files & ExpF Max Confidence ===')
for r in artillery:
    specs = torch.stack([w['spec'] for w in r['windows']]).unsqueeze(1).float()
    with torch.no_grad():
        probs = torch.softmax(model(specs), dim=-1)[:, 1]
    p_max = float(torch.max(probs))
    n_high = int((probs >= 0.70).sum())
    print(f"  {r['name']}: max_p={p_max:.4f}, high_windows={n_high}/{len(r['windows'])}, group={r.get('group_id')}")

print('\n=== 13 Training Vehicle Backfire Files & ExpF Max Confidence ===')
for r in backfires:
    specs = torch.stack([w['spec'] for w in r['windows']]).unsqueeze(1).float()
    with torch.no_grad():
        probs = torch.softmax(model(specs), dim=-1)[:, 1]
    p_max = float(torch.max(probs))
    n_high = int((probs >= 0.70).sum())
    print(f"  {r['name']}: max_p={p_max:.4f}, high_windows={n_high}/{len(r['windows'])}, group={r.get('group_id')}")
