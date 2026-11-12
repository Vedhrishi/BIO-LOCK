"""Olivetti/ORL (controlled) prep: restore the 92:112 aspect, upscale, pad, then run the SAME Haar face + eye detection."""
import os, numpy as np, cv2, time
import rlib
from sklearn.datasets import fetch_olivetti_faces
d = fetch_olivetti_faces(data_home=os.path.join(rlib.R, 'data'), shuffle=False)
imgs = np.clip(np.rint(d.images * 255), 0, 255).astype(np.uint8)
print(imgs.shape, d.target[:12], 'mean', imgs.mean())
face_cc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
out_imgs, boxes, eyes = [], [], []
H, W = 280, 230            # 64x64 -> 230x280 restores ~92:112
PAD = 60
for g in imgs:
    up = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
    canvas = cv2.copyMakeBorder(up, PAD, PAD, PAD + 25, PAD + 25, cv2.BORDER_REPLICATE)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.8)        # remove upscaling blockiness
    out_imgs.append(canvas)
    eq = cv2.equalizeHist(canvas)
    det = face_cc.detectMultiScale(eq, 1.1, 5, minSize=(80, 80))
    if len(det) == 0:
        boxes.append((-1, -1, -1, -1)); eyes.append([np.nan] * 4); continue
    b = max(det, key=lambda b: b[2] * b[3])
    boxes.append(tuple(int(v) for v in b))
    e = rlib.find_eyes(eq, b)
    eyes.append([np.nan] * 4 if e is None else [e[0][0], e[0][1], e[1][0], e[1][1]])
boxes, eyes = np.array(boxes), np.array(eyes, np.float32)
print('canvas', out_imgs[0].shape, 'face found %.3f' % (boxes[:, 2] > 0).mean(), 'eyes found %.3f' % (~np.isnan(eyes[:, 0])).mean(),
      'box w mean %.0f' % boxes[boxes[:, 2] > 0, 2].mean())
np.savez(os.path.join(rlib.R, 'oliv_prep.npz'), images=np.array(out_imgs), target=d.target, boxes=boxes, eyes=eyes)
cv2.imwrite(os.path.join(rlib.R, 'oliv_sample.png'), np.hstack([out_imgs[i] for i in (0, 15, 37, 123, 250)]))
