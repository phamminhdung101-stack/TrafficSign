import cv2
import numpy as np
from ultralytics import YOLO

class TestDetector:
    def __init__(self):
        self.model = YOLO('d:/traffic-sign-recognition/backend/models/best.pt')

    def _run_yolo(self, img, conf):
        res = self.model.predict(source=img, conf=conf, verbose=False)[0]
        dets = []
        for b in res.boxes:
            cls_id = int(b.cls[0].item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append({
                'class_id': cls_id,
                'code': self.model.names[cls_id],
                'confidence': float(b.conf[0].item()),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })
        return dets

    def predict(self, image, conf):
        dets = self._run_yolo(image, conf)
        if not dets and (image.shape[0] < 450 or image.shape[1] < 450):
            h, w = image.shape[:2]
            pad_y = max(30, int(h * 0.45))
            pad_x = max(30, int(w * 0.45))
            padded = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
            raw_dets = self._run_yolo(padded, conf)
            adjusted_dets = []
            for d in raw_dets:
                adj_x1 = max(0.0, d['x1'] - pad_x)
                adj_y1 = max(0.0, d['y1'] - pad_y)
                adj_x2 = min(float(w), d['x2'] - pad_x)
                adj_y2 = min(float(h), d['y2'] - pad_y)
                if adj_x2 > adj_x1 and adj_y2 > adj_y1:
                    d['x1'], d['y1'], d['x2'], d['y2'] = adj_x1, adj_y1, adj_x2, adj_y2
                    adjusted_dets.append(d)
            return adjusted_dets
        return dets

td = TestDetector()
img_107 = cv2.imread('d:/traffic-sign-recognition/backend/tests/crop_107x117.png')
dets = td.predict(img_107, 0.25)
print('Results on 107x117 crop with auto-padding fallback:')
for d in dets:
    print(f"  [{d['code']}] Confidence: {d['confidence']*100:.1f}% | Box: ({d['x1']:.1f}, {d['y1']:.1f}, {d['x2']:.1f}, {d['y2']:.1f})")
