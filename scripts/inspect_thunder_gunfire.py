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
thunder = [r for r in records if r.get('category') == 'thunder']
gunfire = [r for r in records if r.get('category') == 'gunfire']

print(f'=== 23 Training Thunder Files (ExpF >= 0.70) ===')
for r in thunder:
    specs = torch.stack([w['spec'] for w in r['windows']]).unsqueeze(1).float()
    with torch.no_grad():
        probs = torch.softmax(model(specs), dim=-1)[:, 1]
    p_max = float(torch.max(probs))
    n_high = int((probs >= 0.70).sum())
    if p_max >= 0.70:
        print(f"  {r['name']}: max_p={p_max:.4f}, high={n_high}/{len(r['windows'])}, group={r.get('group_id')}")

print(f'\n=== 51 Training Gunfire Files (ExpF >= 0.70) ===')
for r in gunfire:
    specs = torch.stack([w['spec'] for w in r['windows']]).unsqueeze(1).float()
    with torch.no_grad():
        probs = torch.softmax(model(specs), dim=-1)[:, 1]
    p_max = float(torch.max(probs))
    n_high = int((probs >= 0.70).sum())
    if p_max >= 0.70:
        print(f"  {r['name']}: max_p={p_max:.4f}, high={n_high}/{len(r['windows'])}, group={r.get('group_id')}")
