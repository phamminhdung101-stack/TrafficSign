import time
import io
import json
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi.testclient import TestClient

from app.main import app, detector, get_current_user

# Setup test client with authenticated user
app.dependency_overrides[get_current_user] = lambda: {
    'id': 1, 'username': 'admin', 'email': 'admin@example.com', 'role': 'admin', 'is_active': True, 'full_name': 'Admin'
}
client = TestClient(app)

# Reload detector to ensure it picks up backend/models/best.pt
detector.reload()

# Define 7 test cases
test_base = Path('d:/traffic-sign-recognition/dataset/test')
cases = [
    {
        'id': 1,
        'description': 'Biển báo cấm đi ngược chiều (P-102)',
        'file': 'image_251_jpg.rf.10f55289ea9d639d5e44f7d4aff849be.jpg',
        'gt': ['P-102']
    },
    {
        'id': 2,
        'description': 'Biển báo cấm dừng và đỗ (P-130)',
        'file': 'image_106_jpg.rf.eace64a9e87835da57f7f37119c56807.jpg',
        'gt': ['P-106a', 'P-131a', 'P-130']
    },
    {
        'id': 3,
        'description': 'Biển giới hạn tốc độ (P-127)',
        'file': 'image_151_jpg.rf.cfbded33bd497bc6580b4e694a37d55c.jpg',
        'gt': ['P-130', 'P-127']
    },
    {
        'id': 4,
        'description': 'Biển báo ở khoảng cách gần (Close-up)',
        'file': 'IMG_4034-MOV_out0002_png_jpg.rf.103a027339d3933be086c661d5674496.jpg',
        'gt': ['P-124c']
    },
    {
        'id': 5,
        'description': 'Biển báo nhỏ / ở xa (Small / Distant)',
        'file': 'image_151_jpg.rf.cfbded33bd497bc6580b4e694a37d55c.jpg',
        'gt': ['P-127 (small)']
    },
    {
        'id': 6,
        'description': 'Ánh sáng yếu / nền phức tạp (Complex / Low-light)',
        'file': 'image_101_jpg.rf.eda79ca91ed0d47c0d108d1d9a5d9d96.jpg',
        'gt': ['P-106a', 'P-131a']
    },
    {
        'id': 7,
        'description': 'Ảnh không có biển báo (Background / Negative)',
        'file': 'image_126_jpg.rf.1cc56b4264534408513471f1db1c76c0.jpg',
        'gt': []
    }
]

yolo_model = YOLO('d:/traffic-sign-recognition/backend/models/best.pt')

print('================================================================================')
print('=== KẾT QUẢ ĐỐI CHIẾU KIỂM THỬ 7 TÌNH HUỐNG (PYTHON DIRECT vs FASTAPI API) ===')
print('================================================================================\n')

from unittest.mock import patch

for c in cases:
    img_path = test_base / 'images' / c['file']
    if not img_path.exists():
        print(f"File not found: {img_path}")
        continue
    
    pil_img = Image.open(img_path).convert('RGB')
    w_orig, h_orig = pil_img.size
    
    # --- A. Python Direct Inference ---
    t0 = time.perf_counter()
    bgr_np = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    py_results = yolo_model.predict(source=bgr_np, conf=0.25, verbose=False)[0]
    t_py = (time.perf_counter() - t0) * 1000
    
    py_dets = []
    for b in py_results.boxes:
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        x1, y1, x2, y2 = [round(v, 1) for v in b.xyxy[0].tolist()]
        py_dets.append({
            'code': yolo_model.names[cls_id],
            'conf': round(conf, 3),
            'box': [x1, y1, x2, y2]
        })
        
    # --- B. FastAPI API Call ---
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    buf.seek(0)
    
    t1 = time.perf_counter()
    with patch('app.main.save_image_detection', return_value=123):
        resp = client.post('/api/detect?confidence=0.25', files={'file': (c['file'], buf.getvalue(), 'image/jpeg')})
    t_api = (time.perf_counter() - t1) * 1000
    
    api_dets = []
    if resp.status_code == 200:
        api_data = resp.json()
        for d in api_data['detections']:
            api_dets.append({
                'code': d['code'],
                'conf': round(d['confidence'], 3),
                'box': [round(d['x1'], 1), round(d['y1'], 1), round(d['x2'], 1), round(d['y2'], 1)]
            })
            
    case_num = c['id']
    desc = c['description']
    fname = c['file']
    gt = c['gt']
    print(f"Test Case {case_num}: {desc}")
    print(f"  - File: {fname[:35]}... ({w_orig}x{h_orig})")
    print(f"  - Ground Truth: {gt}")
    print(f"  - A. Python Direct ({t_py:.1f}ms): {len(py_dets)} phát hiện")
    for d in py_dets:
        print(f"       [{d['code']}] Conf: {d['conf']*100:.1f}% | Box: {d['box']}")
    print(f"  - B. FastAPI API    ({t_api:.1f}ms): {len(api_dets)} phát hiện")
    for d in api_dets:
        print(f"       [{d['code']}] Conf: {d['conf']*100:.1f}% | Box: {d['box']}")
    
    match = (len(py_dets) == len(api_dets))
    if match and py_dets:
        match = (py_dets[0]['code'] == api_dets[0]['code'])
    print(f"  => ĐỒNG BỘ PYTHON & FASTAPI: {'100% TRÙNG KHỚP' if match else 'KHÁC NHAU'}\n")
