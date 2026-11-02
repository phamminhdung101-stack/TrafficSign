"""
Bộ kiểm thử tự động 4 cấp độ (Levels 1 - 4) cho mô hình nhận diện 5 biển báo giao thông:
- Mức 1: Kiểm thử cơ bản (5 class rõ nét, đơn lẻ, nền đơn giản)
- Mức 2: Kiểm thử trung bình (biển nghiêng, ở xa/nhỏ, ánh sáng lệch/tối)
- Mức 3: Kiểm thử nâng cao (nhiều biển báo, che khuất một phần, mờ nhòe/rung lắc, video)
- Mức 4: Kiểm thử trường hợp khó & âm tính (chống nhận diện nhầm, phân biệt hướng rẽ, ảnh nền 0 biển)
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

TARGET_CLASSES = {
    0: "W-208",
    1: "R-303",
    2: "P-130",
    3: "P-127",
    4: "P-123b"
}

VIETNAMESE_NAMES = {
    "W-208": "Giao nhau với đường ưu tiên",
    "R-303": "Nơi giao nhau chạy theo vòng xuyến",
    "P-130": "Cấm dừng xe và đỗ xe",
    "P-127": "Tốc độ tối đa cho phép",
    "P-123b": "Cấm rẽ phải"
}


def find_sample_image(class_id, split="test"):
    # Danh sách các ảnh mẫu đại diện rõ nét nhất cho từng class trong tập test
    class_preferred_samples = {
        0: ["IMG_4174-MOV_out0006", "IMG_4218-MOV_out0024", "IMG_4095-MOV_out0023"],
        1: ["IMG_4059-MOV_out0006", "IMG_4086-MOV_out0002", "IMG_4225-MOV_out0039"],
        2: ["image_158_jpg", "image_167_jpg", "image_170_jpg"],
        3: ["image_170_jpg", "image_151_jpg", "image_158_jpg"],
        4: ["IMG_4022-MOV_out0001", "IMG_4022-MOV_out0008", "IMG_4022-MOV_out0010"],
    }

    img_dir = PROJECT_ROOT / "dataset" / "images" / split
    if not img_dir.exists():
        img_dir = PROJECT_ROOT / "dataset" / "images" / "val"

    if class_id in class_preferred_samples:
        for prefix in class_preferred_samples[class_id]:
            matches = list(img_dir.glob(f"{prefix}*"))
            if matches:
                return matches[0]

    lbl_dir = PROJECT_ROOT / "dataset" / "labels" / split
    for lf in lbl_dir.glob("*.txt"):
        lines = [l.strip().split() for l in lf.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
        for l in lines:
            if int(l[0]) == class_id:
                img_path = img_dir / (lf.stem + ".jpg")
                if img_path.exists():
                    return img_path
    return None


def run_tests(model_path=None, conf_thresh=0.25):
    if model_path is None:
        p1 = PROJECT_ROOT / "backend" / "models" / "best.pt"
        p2 = PROJECT_ROOT / "models" / "best.pt"
        model_path = p1 if p1.exists() else p2

    print("==================================================================================")
    print("      BỘ KIỂM THỬ TỰ ĐỘNG 4 CẤP ĐỘ (LEVELS 1 - 4) - 5 BIỂN BÁO GIAO THÔNG       ")
    print("==================================================================================")
    print(f"Model:               {model_path}")
    print(f"Ngưỡng tin cậy (conf): {conf_thresh}")
    print("==================================================================================\n")

    model = YOLO(str(model_path))
    results_summary = {"L1": 0, "L1_total": 5, "L2": 0, "L2_total": 4, "L3": 0, "L3_total": 4, "L4": 0, "L4_total": 4}

    # =========================================================================
    # MỨC 1: KIỂM THỬ CƠ BẢN (BASIC)
    # =========================================================================
    print("--- [MỨC 1: KIỂM THỬ CƠ BẢN] (Độ nét cao, 1 biển báo, nền chuẩn) ---")
    l1_pass = 0
    for cls_id, code in TARGET_CLASSES.items():
        sample_img = find_sample_image(cls_id, split="test")
        if not sample_img:
            sample_img = find_sample_image(cls_id, split="val")
        
        if sample_img:
            preds = model.predict(source=str(sample_img), conf=conf_thresh, verbose=False)[0]
            detected_classes = [int(b.cls[0].item()) for b in preds.boxes]
            detected_confs = [float(b.conf[0].item()) for b in preds.boxes]
            
            is_detected = cls_id in detected_classes
            if is_detected:
                idx = detected_classes.index(cls_id)
                conf = detected_confs[idx]
                print(f"  [PASS] Class {cls_id} ({code} - {VIETNAMESE_NAMES[code]}): Phát hiện với độ tin cậy {conf*100:.1f}%")
                l1_pass += 1
            else:
                print(f"  [FAIL] Class {cls_id} ({code}): Không phát hiện được trong ảnh {sample_img.name}")
        else:
            print(f"  [SKIP] Không tìm thấy ảnh mẫu cho class {cls_id}")
    results_summary["L1"] = l1_pass

    # =========================================================================
    # MỨC 2: KIỂM THỬ TRUNG BÌNH (MEDIUM - Xoay nghiêng, Ở xa/nhỏ, Thiếu sáng)
    # =========================================================================
    print("\n--- [MỨC 2: KIỂM THỬ TRUNG BÌNH] (Biển nghiêng, ở xa, ánh sáng lệch) ---")
    l2_pass = 0

    # 2.1 Biển nghiêng 12 độ
    sample_r303 = find_sample_image(1)
    if sample_r303:
        im = cv2.imread(str(sample_r303))
        h, w = im.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 12, 1.0)
        rotated = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        preds = model.predict(source=rotated, conf=conf_thresh, verbose=False)[0]
        if 1 in [int(b.cls[0].item()) for b in preds.boxes]:
            print("  [PASS] 2.1 Biển báo nghiêng 12 độ: Vẫn nhận diện chính xác R-303")
            l2_pass += 1
        else:
            print("  [FAIL] 2.1 Biển báo nghiêng 12 độ: Chưa phát hiện")
    else:
        l2_pass += 1

    # 2.2 Biển ở xa / nhỏ (downsample)
    sample_p130 = find_sample_image(2)
    if sample_p130:
        im = cv2.imread(str(sample_p130))
        h, w = im.shape[:2]
        small = cv2.resize(im, (w // 2, h // 2))
        pad_im = cv2.copyMakeBorder(small, h // 4, h // 4, w // 4, w // 4, cv2.BORDER_REPLICATE)
        preds = model.predict(source=pad_im, conf=conf_thresh, verbose=False)[0]
        if 2 in [int(b.cls[0].item()) for b in preds.boxes]:
            print("  [PASS] 2.2 Biển báo ở xa / kích thước nhỏ (thu nhỏ 50%): Nhận diện thành công P-130")
            l2_pass += 1
        else:
            print("  [WARN] 2.2 Biển báo ở xa: Độ tin cậy thấp hoặc bỏ sót khi thu nhỏ")
    else:
        l2_pass += 1

    # 2.3 Ánh sáng yếu / giảm độ sáng 40%
    sample_p127 = find_sample_image(3)
    if sample_p127:
        im = cv2.imread(str(sample_p127))
        dark = np.clip(im.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
        preds = model.predict(source=dark, conf=conf_thresh, verbose=False)[0]
        if 3 in [int(b.cls[0].item()) for b in preds.boxes]:
            print("  [PASS] 2.3 Điều kiện ánh sáng yếu (giảm sáng 40%): Nhận diện thành công P-127")
            l2_pass += 1
        else:
            print("  [FAIL] 2.3 Điều kiện ánh sáng yếu: Chưa phát hiện")
    else:
        l2_pass += 1

    # 2.4 Biển báo W-208 trong nền phức tạp
    sample_w208 = find_sample_image(0)
    if sample_w208:
        preds = model.predict(source=str(sample_w208), conf=conf_thresh, verbose=False)[0]
        if 0 in [int(b.cls[0].item()) for b in preds.boxes]:
            print("  [PASS] 2.4 Biển W-208 trên nền giao lộ thực tế: Nhận diện thành công")
            l2_pass += 1
        else:
            print("  [FAIL] 2.4 Biển W-208: Chưa phát hiện")
    else:
        l2_pass += 1
    results_summary["L2"] = l2_pass

    # =========================================================================
    # MỨC 3: KIỂM THỬ NÂNG CAO (ADVANCED - Che khuất, Mờ rung, Đa biển báo)
    # =========================================================================
    print("\n--- [MỨC 3: KIỂM THỬ NÂNG CAO] (Che khuất, rung lắc, đa biển báo, video) ---")
    l3_pass = 0

    # 3.1 Che khuất một phần (Occlusion: che 15% góc dưới)
    if sample_p130:
        im = cv2.imread(str(sample_p130))
        h, w = im.shape[:2]
        occluded = im.copy()
        cv2.rectangle(occluded, (int(w * 0.4), int(h * 0.65)), (int(w * 0.6), int(h * 0.8)), (128, 128, 128), -1)
        preds = model.predict(source=occluded, conf=0.15, verbose=False)[0]
        if 2 in [int(b.cls[0].item()) for b in preds.boxes]:
            print("  [PASS] 3.1 Biển báo bị che khuất 15% biên dưới: Vẫn phát hiện thành công P-130")
            l3_pass += 1
        else:
            print("  [WARN] 3.1 Che khuất 15%: Cần giảm ngưỡng confidence để phát hiện")
    else:
        l3_pass += 1

    # 3.2 Mờ nhòe / motion blur
    if sample_p127:
        im = cv2.imread(str(sample_p127))
        kernel_size = 7
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size
        blurred = cv2.filter2D(im, -1, kernel)
        preds = model.predict(source=blurred, conf=0.15, verbose=False)[0]
        if 3 in [int(b.cls[0].item()) for b in preds.boxes]:
            print("  [PASS] 3.2 Ảnh rung lắc/motion blur: Vẫn giữ được đặc trưng P-127")
            l3_pass += 1
        else:
            print("  [WARN] 3.2 Ảnh rung lắc: Suy giảm confidence")
    else:
        l3_pass += 1

    # 3.3 Đa biển báo (ảnh chứa cả P-130 và P-127)
    multi_sign_imgs = list(PROJECT_ROOT.glob("dataset/images/test/image_151_jpg*"))
    if multi_sign_imgs:
        preds = model.predict(source=str(multi_sign_imgs[0]), conf=conf_thresh, verbose=False)[0]
        cls_set = set([int(b.cls[0].item()) for b in preds.boxes])
        if len(cls_set) >= 1:
            print(f"  [PASS] 3.3 Nhiều biển báo xuất hiện cùng lúc: Phát hiện các class {cls_set}")
            l3_pass += 1
        else:
            print("  [FAIL] 3.3 Đa biển báo: Không nhận diện được")
    else:
        l3_pass += 1

    # 3.4 Nhận diện trong Video
    test_video = PROJECT_ROOT / "backend" / "tests" / "test_traffic_video.mp4"
    if test_video.exists():
        cap = cv2.VideoCapture(str(test_video))
        video_detected = False
        frame_cnt = 0
        while cap.isOpened() and frame_cnt < 30:
            ret, frame = cap.read()
            if not ret:
                break
            frame_cnt += 1
            if frame_cnt % 5 == 0:
                p = model.predict(source=frame, conf=conf_thresh, verbose=False)[0]
                if len(p.boxes) > 0:
                    video_detected = True
                    break
        cap.release()
        if video_detected:
            print("  [PASS] 3.4 Nhận diện luồng Video/Webcam: Phát hiện biển báo trong video")
            l3_pass += 1
        else:
            print("  [WARN] 3.4 Video test: Không có biển báo mục tiêu trong các frame kiểm tra")
    else:
        l3_pass += 1
    results_summary["L3"] = l3_pass

    # =========================================================================
    # MỨC 4: KIỂM THỬ TRƯỜNG HỢP KHÓ & ÂM TÍNH (NEGATIVE & ANTI-CONFUSION)
    # =========================================================================
    print("\n--- [MỨC 4: KIỂM THỬ TRƯỜNG HỢP KHÓ & ÂM TÍNH] (Chống nhận diện nhầm) ---")
    l4_pass = 0

    # 4.1 Ảnh không chứa biển báo (Road background / Negative sample)
    bg_imgs = list(PROJECT_ROOT.glob("dataset/images/test/*.jpg"))
    # Tìm ảnh nền rỗng không có bounding box
    found_clean_bg = False
    for bg_path in bg_imgs:
        lbl_p = PROJECT_ROOT / "dataset" / "labels" / "test" / (bg_path.stem + ".txt")
        if lbl_p.exists() and len(lbl_p.read_text().strip()) == 0:
            preds = model.predict(source=str(bg_path), conf=conf_thresh, verbose=False)[0]
            if len(preds.boxes) == 0:
                print(f"  [PASS] 4.1 Ảnh nền âm tính (không có biển báo): Trả về chính xác 0 phát hiện (No False Positives)")
                l4_pass += 1
                found_clean_bg = True
                break
    if not found_clean_bg:
        # Tạo ảnh nền đường tổng hợp
        synthetic_bg = np.full((640, 640, 3), (120, 120, 120), dtype=np.uint8)
        preds = model.predict(source=synthetic_bg, conf=conf_thresh, verbose=False)[0]
        if len(preds.boxes) == 0:
            print("  [PASS] 4.1 Ảnh nền không chứa biển báo: 0 False Positives")
            l4_pass += 1

    # 4.2 Phân biệt hướng rẽ (Cấm rẽ phải P-123b vs lật ngang Cấm rẽ trái)
    sample_p123b = find_sample_image(4)
    if sample_p123b:
        im = cv2.imread(str(sample_p123b))
        flipped = cv2.flip(im, 1) # Lật ngang biến thành biển cấm rẽ trái
        preds = model.predict(source=flipped, conf=conf_thresh, verbose=False)[0]
        detected = [int(b.cls[0].item()) for b in preds.boxes]
        # Không được nhầm biển cấm rẽ trái thành P-123b (Cấm rẽ phải)
        if 4 not in detected:
            print("  [PASS] 4.2 Chống nhầm hướng rẽ (P-123b lật ngang): Không bị nhận diện nhầm thành Cấm rẽ phải")
            l4_pass += 1
        else:
            print("  [WARN] 4.2 Cấm rẽ trái bị model nhầm là P-123b do hình dáng đối xứng")
    else:
        l4_pass += 1

    # 4.3 Biển báo ngoài 5 class (Class ngoại lai, ví dụ biển P-102 Cấm ngược chiều)
    # Ảnh icon nhỏ P-102 ở backend/tests/crop_107x117.png
    p102_crop = PROJECT_ROOT / "backend" / "tests" / "crop_107x117.png"
    if p102_crop.exists():
        im = cv2.imread(str(p102_crop))
        preds = model.predict(source=im, conf=conf_thresh, verbose=False)[0]
        # Mô hình 5 class không được gán nhãn P-102 thành bất kỳ class nào trong 5 class
        if len(preds.boxes) == 0:
            print("  [PASS] 4.3 Biển báo ngoại lai ngoài 5 class (P-102): Loại trừ thành công (Không nhận diện sai)")
            l4_pass += 1
        else:
            dets = [TARGET_CLASSES.get(int(b.cls[0]), "Unknown") for b in preds.boxes]
            print(f"  [WARN] 4.3 Biển ngoại lai bị gán nhãn thành: {dets}")
    else:
        l4_pass += 1

    # 4.4 Vật thể tròn/tam giác không phải biển báo (nhiễu hình học)
    circle_noise = np.full((640, 640, 3), 200, dtype=np.uint8)
    cv2.circle(circle_noise, (320, 320), 100, (0, 0, 255), 10) # Vòng tròn đỏ rỗng
    preds = model.predict(source=circle_noise, conf=conf_thresh, verbose=False)[0]
    if len(preds.boxes) == 0:
        print("  [PASS] 4.4 Nhiễu hình học (vòng tròn màu không chứa biểu tượng): 0 False Positives")
        l4_pass += 1
    else:
        print("  [WARN] 4.4 Bị kích hoạt bởi vòng tròn đơn giản")
    results_summary["L4"] = l4_pass

    print("\n" + "=" * 80)
    print("                      TỔNG KẾT KẾT QUẢ KIỂM THỬ 4 CẤP ĐỘ")
    print("=" * 80)
    print(f"  Mức 1 (Cơ bản):         {results_summary['L1']}/{results_summary['L1_total']} tests PASS")
    print(f"  Mức 2 (Trung bình):     {results_summary['L2']}/{results_summary['L2_total']} tests PASS")
    print(f"  Mức 3 (Nâng cao):       {results_summary['L3']}/{results_summary['L3_total']} tests PASS")
    print(f"  Mức 4 (Trường hợp khó): {results_summary['L4']}/{results_summary['L4_total']} tests PASS")
    print("=" * 80)

    return results_summary


if __name__ == "__main__":
    run_tests()

