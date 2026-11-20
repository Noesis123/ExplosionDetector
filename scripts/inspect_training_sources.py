import torch
from collections import Counter

aug = torch.load('training/data/augmented_training_pool.pt', map_location='cpu', weights_only=False)
pool_neg = aug['pool_neg']
pool_pos = aug['pool_pos']

# Analyze neg pool by filename / recording
neg_by_cat_files = {}
for w in pool_neg:
    cat = w.get('category', 'unknown')
    fname = w.get('filename', 'unknown')
    if cat not in neg_by_cat_files:
        neg_by_cat_files[cat] = set()
    neg_by_cat_files[cat].add(fname)

print('=== Unique Files per Category in pool_neg ===')
for cat, files in sorted(neg_by_cat_files.items(), key=lambda x: -len(x[1])):
    tot_win = sum(1 for w in pool_neg if w.get('category') == cat)
    print(f'  {cat:<28}: {len(files):>4} unique files, {tot_win:>5} windows ({tot_win/len(files):.1f} win/file)')

pos_by_cat_files = {}
for w in pool_pos:
    cat = w.get('category', 'unknown')
    fname = w.get('filename', 'unknown')
    if cat not in pos_by_cat_files:
        pos_by_cat_files[cat] = set()
    pos_by_cat_files[cat].add(fname)

print('\n=== Unique Files per Category in pool_pos ===')
for cat, files in sorted(pos_by_cat_files.items(), key=lambda x: -len(x[1])):
    tot_win = sum(1 for w in pool_pos if w.get('category') == cat)
    print(f'  {cat:<28}: {len(files):>4} unique files, {tot_win:>5} windows ({tot_win/len(files):.1f} win/file)')

total_unique_neg = len(set(w.get('filename', 'unknown') for w in pool_neg))
total_unique_pos = len(set(w.get('filename', 'unknown') for w in pool_pos))
print(f'\nTotal unique neg files: {total_unique_neg}')
print(f'Total unique pos files: {total_unique_pos}')
print(f'Total unique training files: {total_unique_neg + total_unique_pos}')
