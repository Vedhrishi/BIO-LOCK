"""A drawn face for the tests: no photograph of a real person is needed, nothing is downloaded.

`make_face(seed)` returns a 250x250 grey image of a cartoon-like face that OpenCV's Haar face and eye
cascades detect. Different seeds give different proportions and skin texture ("identities")."""
import numpy as np
import cv2


def make_face(seed=0, size=250):
    r = np.random.RandomState(seed)
    j = lambda lo, hi: float(r.uniform(lo, hi))                       # noqa: E731
    s = size / 250.0
    img = np.full((size, size), j(95, 125), np.float32)
    img += cv2.GaussianBlur(r.normal(0, 18, (size, size)).astype(np.float32), (0, 0), 9 * s)   # background
    cx, cy = size / 2.0 + j(-3, 3) * s, size / 2.0 + j(-2, 4) * s
    fw, fh = j(54, 68) * s, j(74, 90) * s                               # half axes of the face
    skin = j(165, 195)
    face = np.zeros((size, size), np.uint8)
    cv2.ellipse(face, (int(cx), int(cy)), (int(fw), int(fh)), 0, 0, 360, 255, -1)
    hair = np.zeros((size, size), np.uint8)
    cv2.ellipse(hair, (int(cx), int(cy - 0.30 * fh)), (int(fw * 1.12), int(fh * 0.95)), 0, 180, 360, 255, -1)
    img[hair > 0] = j(25, 55)
    shade = skin + cv2.GaussianBlur(r.normal(0, 10, (size, size)).astype(np.float32), (0, 0), 5 * s)
    xx = np.arange(size, dtype=np.float32)[None, :]
    shade = shade - 28.0 * np.abs((xx - cx) / fw) ** 2                    # darker towards the sides
    # Skin texture that belongs to this identity. LBP histograms describe texture statistics per region, so
    # each identity gets its own mix of oriented textures (direction, coarseness and strength differ by region).
    yy = np.arange(size, dtype=np.float32)[:, None]
    tex = np.zeros((size, size), np.float32)
    wsum = np.zeros((size, size), np.float32)
    for _ in range(4):
        ang, coarse, stretch = j(0, np.pi), j(1.2, 3.0) * s, j(2.0, 5.0)
        k = int(6 * coarse * stretch) | 1
        g = cv2.getGaussianKernel(k, coarse * stretch)
        kern = (g @ cv2.getGaussianKernel(k, coarse).T).astype(np.float32)
        rot = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), np.degrees(ang), 1.0)
        kern = cv2.warpAffine(kern, rot, (k, k))
        f = cv2.filter2D(r.normal(0, 1, (size, size)).astype(np.float32), -1, kern / (kern.sum() + 1e-9))
        f -= cv2.GaussianBlur(f, (0, 0), 8 * s)
        f *= j(10, 26) / (float(f.std()) + 1e-6)
        px, py, rad = cx + j(-0.8, 0.8) * fw, cy + j(-0.8, 0.8) * fh, j(0.5, 1.1) * fw
        w = np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * rad * rad))).astype(np.float32)
        tex += w * f
        wsum += w
    shade = shade + tex / np.maximum(wsum, 0.35)
    img[face > 0] = shade[face > 0]
    for _ in range(int(r.randint(3, 7))):                                 # a few marks / lines
        px, py = cx + j(-0.7, 0.7) * fw, cy + j(-0.1, 0.8) * fh
        if r.rand() < 0.5:
            cv2.circle(img, (int(px), int(py)), int(j(2, 4) * s), skin - j(50, 90), -1)
        else:
            cv2.line(img, (int(px), int(py)), (int(px + j(-18, 18) * s), int(py + j(-6, 6) * s)), skin - j(40, 70),
                     max(1, int(2 * s)))
    ey = cy - j(0.20, 0.26) * fh                                         # eye line
    ex = j(0.40, 0.54) * fw
    ew, eh = j(13, 19) * s, j(6.0, 9.0) * s
    for sx in (-1, 1):
        c = (int(cx + sx * ex), int(ey))
        cv2.ellipse(img, (c[0], int(ey - 2 * s)), (int(ew * 1.5), int(eh * 2.4)), 0, 0, 360, skin - 45, -1)   # socket
        cv2.ellipse(img, c, (int(ew), int(eh)), 0, 0, 360, 235, -1)                                         # sclera
        cv2.circle(img, (c[0] + int(j(-2, 2) * s), c[1]), int(eh * 0.95), 35, -1)                          # iris
        cv2.ellipse(img, c, (int(ew), int(eh)), 0, 180, 360, 30, max(1, int(2 * s)))                       # lash line
        cv2.ellipse(img, (c[0], int(ey - j(15, 19) * s)), (int(ew * 1.25), int(4 * s)), 0, 0, 360, j(30, 60), -1)   # brow
    ny = cy + j(0.10, 0.18) * fh
    cv2.line(img, (int(cx), int(ey + 6 * s)), (int(cx), int(ny)), skin + 22, max(2, int(7 * s)))          # nose ridge
    cv2.ellipse(img, (int(cx), int(ny + 3 * s)), (int(j(11, 14) * s), int(5 * s)), 0, 0, 360, skin - 55, -1)   # nostrils
    my = cy + j(0.44, 0.52) * fh
    cv2.ellipse(img, (int(cx), int(my)), (int(j(20, 26) * s), int(j(4, 6) * s)), 0, 0, 360, j(70, 95), -1)   # mouth
    cv2.ellipse(img, (int(cx), int(cy + 0.95 * fh)), (int(fw * 0.8), int(fh * 0.16)), 0, 0, 360, skin - 60, -1)   # chin shadow
    # features that some identities have and others do not (they change whole regions of the face)
    if r.rand() < 0.5:                                                    # beard: rough dark lower face
        beard = np.zeros((size, size), np.uint8)
        cv2.ellipse(beard, (int(cx), int(cy + 0.42 * fh)), (int(fw * 0.98), int(fh * 0.62)), 0, 0, 180, 255, -1)
        cv2.ellipse(beard, (int(cx), int(my)), (int(j(16, 22) * s), int(j(5, 8) * s)), 0, 0, 360, 0, -1)
        rough = cv2.GaussianBlur(r.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 1.3 * s)
        hairy = j(55, 85) + 38.0 * rough / (float(rough.std()) + 1e-6)
        m = (beard > 0) & (face > 0)
        img[m] = hairy[m]
    if r.rand() < 0.5:                                                    # moustache
        cv2.ellipse(img, (int(cx), int(ny + 11 * s)), (int(j(20, 27) * s), int(j(4, 6) * s)), 0, 0, 360, j(35, 60), -1)
    if r.rand() < 0.4:                                                    # fringe over the forehead
        cv2.ellipse(img, (int(cx + j(-10, 10) * s), int(cy - 0.62 * fh)), (int(fw * 0.95), int(fh * j(0.20, 0.30))),
                    j(-8, 8), 0, 360, j(25, 55), -1)
    for sx in (-1, 1):                                                    # cheek lines / folds
        if r.rand() < 0.6:
            x0 = cx + sx * j(0.22, 0.34) * fw
            cv2.line(img, (int(x0), int(ny)), (int(x0 + sx * j(8, 16) * s), int(my + j(0, 8) * s)), skin - j(45, 75),
                     max(1, int(j(1.5, 3) * s)))
    img = cv2.GaussianBlur(img, (0, 0), 1.6 * s)
    img += r.normal(0, 2.0, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)
