import hashlib
from pathlib import Path
import cv2

old_target_map = {
    46: 0, # W-208
    28: 1, # R-303
    19: 2, # P-130
    17: 3, # P-127
    13: 4  # P-123b
}
class_names = ['W-208', 'R-303', 'P-130', 'P-127', 'P-123b']

splits = {'train': 'train', 'val': 'valid', 'test': 'test'}

stats = {}
image_hashes = {}
duplicates = []
corrupted_images = []
invalid_labels = []

for split_name, folder_name in splits.items():
    lbl_dir = Path(f'dataset/{folder_name}/labels')
    img_dir = Path(f'dataset/{folder_name}/images')
    
    stats[split_name] = {
        'total_images': 0,
        'images_with_target': 0,
        'boxes_per_class': [0] * 5,
        'images_per_class': [set() for _ in range(5)],
        'filtered_label_files': {}
    }
    
    if not lbl_dir.exists():
        continue
        
    for lf in lbl_dir.glob('*.txt'):
        img_name = lf.stem + '.jpg'
        img_path = img_dir / img_name
        if not img_path.exists():
            matches = list(img_dir.glob(f'{lf.stem}.*'))
            if matches:
                img_path = matches[0]
            else:
                corrupted_images.append((split_name, lf.name, 'Missing image file'))
                continue
        
        # Read label
        try:
            content = lf.read_text(encoding='utf-8').strip()
        except Exception as e:
            invalid_labels.append((split_name, lf.name, f'Read error: {e}'))
            continue
            
        target_lines = []
        if content:
            for line_idx, line in enumerate(content.split('\n')):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    invalid_labels.append((split_name, lf.name, f'Line {line_idx} wrong length: {line}'))
                    continue
                try:
                    old_cls = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    invalid_labels.append((split_name, lf.name, f'Line {line_idx} format error: {line}'))
                    continue
                    
                # Check coords range
                cx, cy, w, h = coords
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    invalid_labels.append((split_name, lf.name, f'Line {line_idx} coords out of bounds: {coords}'))
                    continue
                    
                if old_cls in old_target_map:
                    new_cls = old_target_map[old_cls]
                    target_lines.append((new_cls, coords))
                    stats[split_name]['boxes_per_class'][new_cls] += 1
                    stats[split_name]['images_per_class'][new_cls].add(lf.name)
                    
        if target_lines:
            stats[split_name]['images_with_target'] += 1
            stats[split_name]['filtered_label_files'][img_path] = target_lines
            
            # Check image integrity & hash
            try:
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash in image_hashes:
                    duplicates.append((split_name, img_path.name, image_hashes[file_hash]))
                else:
                    image_hashes[file_hash] = (split_name, img_path.name)
            except Exception as e:
                corrupted_images.append((split_name, img_path.name, f'Hash error: {e}'))

print('=== DATASET ANALYSIS RESULT ===')
for split_name in ['train', 'val', 'test']:
    s = stats[split_name]
    num_target = s['images_with_target']
    print(f'\nSplit [{split_name}]:')
    print(f'  Images containing at least one target sign: {num_target}')
    for i, cname in enumerate(class_names):
        box_cnt = s['boxes_per_class'][i]
        img_cnt = len(s['images_per_class'][i])
        print(f'    Class {i} ({cname}): {box_cnt} boxes across {img_cnt} images')

print(f'\nCorrupted / Missing images: {len(corrupted_images)}')
for c in corrupted_images[:5]:
    print('  ', c)

print(f'Invalid labels: {len(invalid_labels)}')
for inv in invalid_labels[:5]:
    print('  ', inv)

print(f'Duplicate images across splits by MD5: {len(duplicates)}')
for d in duplicates[:5]:
    print('  ', d)

