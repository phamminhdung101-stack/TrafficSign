import sys
import requests

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

login_resp = requests.post('http://127.0.0.1:8000/api/auth/login', data={'username': 'admin', 'password': '123456'})
token = login_resp.json()['access_token']
headers = {'Authorization': f'Bearer {token}'}

print('=== 1. TEST ANH DATASET (CHUA P-102 CAM DI NGUOC CHIEU) ===')
img_p1 = 'd:/traffic-sign-recognition/dataset/test/images/image_251_jpg.rf.10f55289ea9d639d5e44f7d4aff849be.jpg'
with open(img_p1, 'rb') as f:
    resp1 = requests.post('http://127.0.0.1:8000/api/detect?confidence=0.25', headers=headers, files={'file': ('test.jpg', f, 'image/jpeg')})
data1 = resp1.json()
print('Status:', resp1.status_code)
for d in data1.get('detections', []):
    print(f"  [{d['code']}] Conf: {d['confidence']*100:.1f}% | Box: ({d['x1']:.1f}, {d['y1']:.1f}, {d['x2']:.1f}, {d['y2']:.1f})")

print('\n=== 2. TEST ANH ICON 107x117 (USER UPLOAD) ===')
img_p2 = 'd:/traffic-sign-recognition/backend/tests/crop_107x117.png'
with open(img_p2, 'rb') as f:
    resp2 = requests.post('http://127.0.0.1:8000/api/detect?confidence=0.25', headers=headers, files={'file': ('icon_107x117.png', f, 'image/png')})
data2 = resp2.json()
print('Status:', resp2.status_code)
for d in data2.get('detections', []):
    print(f"  [{d['code']}] Conf: {d['confidence']*100:.1f}% | Box: ({d['x1']:.1f}, {d['y1']:.1f}, {d['x2']:.1f}, {d['y2']:.1f})")

print('\n=== 3. TEST ANH WEBCAM CHUP TU DIEN THOAI ===')
img_p3 = 'd:/traffic-sign-recognition/backend/tests/user_webcam_exact.jpg'
with open(img_p3, 'rb') as f:
    resp3 = requests.post('http://127.0.0.1:8000/api/detect?confidence=0.25', headers=headers, files={'file': ('webcam.jpg', f, 'image/jpeg')})
data3 = resp3.json()
print('Status:', resp3.status_code)
for d in data3.get('detections', []):
    print(f"  [{d['code']}] Conf: {d['confidence']*100:.1f}% | Box: ({d['x1']:.1f}, {d['y1']:.1f}, {d['x2']:.1f}, {d['y2']:.1f})")
