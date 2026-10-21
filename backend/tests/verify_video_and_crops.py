import sys
import cv2
import numpy as np
from fastapi.testclient import TestClient

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from app.main import app, get_current_user

app.dependency_overrides[get_current_user] = lambda: {'id': 1, 'username': 'admin', 'role': 'admin'}
client = TestClient(app)

print("=== 1. KIEM TRA NHAN DIEN ANH ICON NHO 107x117 (USER UPLOAD) ===")
with open('d:/traffic-sign-recognition/backend/tests/crop_107x117.png', 'rb') as f:
    from unittest.mock import patch
    with patch('app.main.save_image_detection', return_value=101):
        resp = client.post('/api/detect?confidence=0.25', files={'file': ('icon_107x117.png', f, 'image/png')})
print('Status:', resp.status_code)
data = resp.json()
print(f"Phat hien {len(data['detections'])} doi tuong tren anh 107x117:")
for d in data['detections']:
    print(f"  - [{d['code']}] Do tin cay: {d['confidence']*100:.1f}% | Box: ({d['x1']:.1f}, {d['y1']:.1f}, {d['x2']:.1f}, {d['y2']:.1f})")

print("\n=== 2. KIEM TRA NHAN DIEN VIDEO MP4 ===")
# Create a test mp4 video
w, h = 640, 480
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = 'd:/traffic-sign-recognition/backend/tests/test_traffic_video.mp4'
out = cv2.VideoWriter(video_path, fourcc, 10.0, (w, h))
test_img = cv2.imread('d:/traffic-sign-recognition/dataset/test/images/image_251_jpg.rf.10f55289ea9d639d5e44f7d4aff849be.jpg')
test_img = cv2.resize(test_img, (w, h))
for _ in range(20):
    out.write(test_img)
out.release()

with open(video_path, 'rb') as f:
    with patch('app.main.save_image_detection', return_value=102):
        resp_v = client.post('/api/detect?confidence=0.25', files={'file': ('traffic.mp4', f, 'video/mp4')})
print('Video Status:', resp_v.status_code)
data_v = resp_v.json()
print(f"Phat hien {len(data_v['detections'])} doi tuong trong video:")
for d in data_v['detections']:
    print(f"  - [{d['code']}] Do tin cay: {d['confidence']*100:.1f}% | Box: ({d['x1']:.1f}, {d['y1']:.1f}, {d['x2']:.1f}, {d['y2']:.1f})")
print('Co anh keyframe base64 cho frontend:', bool(data_v.get('image_data')))

