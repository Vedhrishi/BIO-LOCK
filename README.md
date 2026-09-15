# BioLock SSD

A biometric USB pendrive security system that encrypts your files with
AES-256-GCM and unlocks them only when your enrolled face is verified.
Runs as a single self-contained `.exe` on any Windows 10/11 laptop —
no installation required on the host machine.

## How it works

1. **Enrol once** — a 4-step liveness check (presence → blink ×2 → head
   turn → 120-frame capture) builds a face gallery using OpenCV's YuNet
   CNN detector and SFace deep recognition network.
2. **Lock** — every file in `SecureVault\` is encrypted with AES-256-GCM
   and the original is securely overwritten (3-pass). The drive is useless
   to anyone without your face and passphrase.
3. **Unlock** — face verification (≥0.45 cosine similarity across 5
   consecutive frames) combined with a passphrase derives the vault key
   via scrypt (128 MB per guess). Both factors are required — the drive
   alone is not enough.

## Security model

| Layer | Implementation |
|---|---|
| Face recognition | YuNet CNN detector + SFace 128-d embedding (same class as Windows Hello 2D) |
| Liveness | Motion + dual-method blink + head-turn tracking |
| Encryption | AES-256-GCM, authenticated, path bound as AAD |
| Key derivation | Two-factor scrypt N=2¹⁷ — passphrase (never stored) + face hash |
| Brute force | Fail-closed counter, exponential backoff, auto-wipe at 5 failures |
| Deletion | 3-pass random overwrite + fsync before unlink |

## Stack

- Python 3.9–3.12
- OpenCV contrib (YuNet + SFace ONNX)
- Python `cryptography` library (AES-256-GCM, scrypt)
- Tkinter UI with OpenCV overlay and numpy-vectorised particle animations
- PyInstaller — single portable EXE, ~80 MB

## Quick start

```bash
# install dependencies
pip install opencv-contrib-python Pillow cryptography

# run directly
python main.py

# or build the standalone EXE for your pendrive
BUILD.bat
```

Run `TEST.bat` (or `python selftest.py`) to execute 40 automated tests
covering the recognition engine, liveness pipeline, two-factor crypto,
tamper detection, and crash recovery.

## Academic context

Final year B.Tech project — CMR Technical Campus, CSE Section E, R-22
**Team:** Rishi (237R1A05X4) · Karthik (237R1A05U1) · Tejasree (237R1A05U4)

**SDG 9** Industry & Innovation · **SDG 16** Privacy & Justice · **SDG 17** Partnerships
PO1 PO2 PO3 PO5 PO9 PO10 PO11 · PSO1 PSO2
