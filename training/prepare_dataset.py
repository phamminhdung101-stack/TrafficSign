import os
import shutil
from pathlib import Path

# Mapping old 56-class IDs to new 5-class IDs
OLD_TO_NEW_MAP = {
    46: 0,  # W-208: Giao nhau với đường ưu tiên
    28: 1,  # R-303: Nơi giao nhau chạy theo vòng xuyến
    19: 2,  # P-130: Cấm dừng xe và đỗ xe
    17: 3,  # P-127: Tốc độ tối đa cho phép
    13: 4,  # P-123b: Cấm rẽ phải
}

CLASS_NAMES = ['W-208', 'R-303', 'P-130', 'P-127', 'P-123b']

SPLIT_MAP = {
    'train': 'train',
    'val': 'valid',
    'test': 'test'
}

BASE_DIR = Path('dataset')
IMAGES_DIR = BASE_DIR / 'images'
LABELS_DIR = BASE_DIR / 'labels'


def prepare():
    print("=== 1. KHỞI TẠO CẤU TRÚC THƯ MỤC CHUẨN YOLOV8 ===")
    for split in ['train', 'val', 'test']:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

    summary = {
        split: {
            'copied_images': 0,
            'background_images': 0,
            'boxes_per_class': [0] * 5,
            'images_per_class': [set() for _ in range(5)]
        }
        for split in ['train', 'val', 'test']
    }

    # Number of background images (without 5 target classes) to retain for negative training
    bg_limits = {'train': 100, 'val': 25, 'test': 15}

    for split_target, split_source in SPLIT_MAP.items():
        src_lbl_dir = BASE_DIR / split_source / 'labels'
        src_img_dir = BASE_DIR / split_source / 'images'

        dst_img_dir = IMAGES_DIR / split_target
        dst_lbl_dir = LABELS_DIR / split_target

        if not src_lbl_dir.exists() or not src_img_dir.exists():
            print(f"Warning: {src_lbl_dir} or {src_img_dir} does not exist!")
            continue

        bg_count = 0
        limit_bg = bg_limits[split_target]

        print(f"\n--- Đang xử lý tập [{split_target}] từ nguồn [{split_source}] ---")
        for lbl_file in src_lbl_dir.glob('*.txt'):
            img_file = src_img_dir / (lbl_file.stem + '.jpg')
            if not img_file.exists():
                matches = list(src_img_dir.glob(f"{lbl_file.stem}.*"))
                if matches:
                    img_file = matches[0]
                else:
                    continue

            content = lbl_file.read_text(encoding='utf-8').strip()
            remapped_lines = []

            if content:
                for line in content.split('\n'):
                    parts = line.strip().split()
                    if len(parts) == 5:
                        old_cls = int(parts[0])
                        if old_cls in OLD_TO_NEW_MAP:
                            new_cls = OLD_TO_NEW_MAP[old_cls]
                            coords = parts[1:5]
                            remapped_lines.append(f"{new_cls} {' '.join(coords)}")
                            summary[split_target]['boxes_per_class'][new_cls] += 1
                            summary[split_target]['images_per_class'][new_cls].add(img_file.name)

            # Case A: Contains at least one of the 5 target signs
            if remapped_lines:
                shutil.copy2(img_file, dst_img_dir / img_file.name)
                (dst_lbl_dir / (img_file.stem + '.txt')).write_text(
                    '\n'.join(remapped_lines) + '\n', encoding='utf-8'
                )
                summary[split_target]['copied_images'] += 1

            # Case B: Negative background image (does not contain any of the 5 signs)
            elif bg_count < limit_bg:
                shutil.copy2(img_file, dst_img_dir / img_file.name)
                # Create empty label file for negative background training
                (dst_lbl_dir / (img_file.stem + '.txt')).write_text('', encoding='utf-8')
                bg_count += 1
                summary[split_target]['background_images'] += 1

    print("\n=== 2. TỔNG KẾT DỮ LIỆU ĐÃ LỌC VÀ CHUYỂN ĐỔI ===")
    for split in ['train', 'val', 'test']:
        s = summary[split]
        total_imgs = s['copied_images'] + s['background_images']
        print(f"\nTập [{split}]:")
        print(f"  Tổng số ảnh: {total_imgs} (Ảnh có biển báo: {s['copied_images']}, Ảnh nền âm tính: {s['background_images']})")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    Class {i} ({name}): {s['boxes_per_class'][i]} bounding boxes / {len(s['images_per_class'][i])} ảnh")

    print("\n=== 3. CẬP NHẬT FILE dataset/data.yaml ===")
    data_yaml_content = f"""# Cấu hình Dataset YOLOv8 - Nhận diện 5 biển báo giao thông Việt Nam
path: {BASE_DIR.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: 5
names:
  0: W-208
  1: R-303
  2: P-130
  3: P-127
  4: P-123b
"""
    (BASE_DIR / 'data.yaml').write_text(data_yaml_content, encoding='utf-8')
    print("Đã cập nhật data.yaml thành công!")


if __name__ == '__main__':
    prepare()

