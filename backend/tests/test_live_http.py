import requests

login_resp = requests.post('http://127.0.0.1:8000/api/auth/login', data={'username': 'admin', 'password': '123456'})
token = login_resp.json()['access_token']
headers = {'Authorization': f'Bearer {token}'}

print('=== 1. TEST ẢNH DATASET (CHỨA P-102 CẤM ĐI NGƯỢC CHIỀU) ===')
img_p1 = 'd:/traffic-sign-recognition/dataset/test/images/image_251_jpg.rf.10f55289ea9d639d5e44f7d4aff849be.jpg'
with open(img_p1, 'rb') as f:
    resp1 = requests.post('http://127.0.0.1:8000/api/detect?confidence=0.25', headers=headers, files={'file': ('test.jpg', f, 'image/jpeg')})
data1 = resp1.json()
print('Status:', resp1.status_code)
for d in data1.get('detections', []):
    print(f"  [{d['code']}] Conf: {d['confidence']*100:.1f}% | Box: ({d['x1']:.1f}, {d['y1']:.1f}, {d['x2']:.1f}, {d['y2']:.1f})")

print('\n=== 2. TEST ẢNH CHỤP CAMERA CỦA BẠN (ĐIỆN THOẠI GIƠ BIỂN P-102) ===')
img_p2 = 'd:/traffic-sign-recognition/backend/tests/user_webcam_exact.jpg'
with open(img_p2, 'rb') as f:
    resp2 = requests.post('http://127.0.0.1:8000/api/detect?confidence=0.25', headers=headers, files={'file': ('webcam.jpg', f, 'image/jpeg')})
data2 = resp2.json()
print('Status:', resp2.status_code)
for d in data2.get('detections', []):
    print(f"  [{d['code']}] Conf: {d['confidence']*100:.1f}% | Box: ({d['x1']:.1f}, {d['y1']:.1f}, {d['x2']:.1f}, {d['y2']:.1f})")
