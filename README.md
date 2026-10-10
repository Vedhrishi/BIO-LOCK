# BioLock SSD

[![tests](https://github.com/Vedhrishi/BIO-LOCK/actions/workflows/tests.yml/badge.svg)](https://github.com/Vedhrishi/BIO-LOCK/actions/workflows/tests.yml)

Face-recognition lock for **external drives** (USB flash drives, external SSD/HDD, SD cards) with
multi-step liveness detection and an encrypted vault marker. Windows desktop app, Python + OpenCV + Tkinter.

Author: Pasupuleti Vedh Rishi

## How it works

Every unlock runs four checks, in order:

1. **Presence**: frame-difference motion on the face region (a static photo produces almost none).
2. **Blink**: two natural blinks, detected with a Haar eye cascade.
3. **Head turn**: left, then right, tracked from the face-box position.
4. **Face match**: weighted fusion of three scores against the enrolled model:
   `score = 0.5·LBPH + 0.3·histogram + 0.2·template`, each mapped to 0–100 (weights in `config.py`).
   Accept when `score ≥ min_conf` (default 40). The score is an **uncalibrated similarity, not a probability**.

Other behaviour: one enrolled face per drive; the vault key is derived from the enrolment (PBKDF2-SHA256, 600k
iterations, Fernet); auto-lock after idle; automatic wipe after 5 failed matches; audit log; `autorun.inf` prompt when
the drive is plugged in.

### Safety: external drives only

BioLock refuses to run from an internal drive and shows *"This cannot be used to lock an internal drive, as it
would endanger the storage components of the system."* A drive counts as external only if Windows reports it as
removable, or as fixed but attached over USB / FireWire / SD / MMC (checked with `IOCTL_STORAGE_QUERY_PROPERTY`).
Anything unknown is treated as internal (fails closed).

## Run it

**Packaged app (no Python needed):** copy the folder or `BioLock.exe` to an external drive and run it there.
Windows SmartScreen may warn because the exe is not code-signed: *More info → Run anyway*.

**From source** (Python 3.10+; developed on 3.11, Windows, webcam):

```bash
pip install -r requirements.txt
python main.py
```

From source the external-drive check is off so it can be developed on an internal disk; set
`BIOLOCK_ENFORCE_EXTERNAL=1` to turn it on. `BIOLOCK_ALLOW_INTERNAL=1` disables it for the packaged exe (developers only).

**Build the exe:**

```bash
pip install pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed --name BioLock --collect-data cv2 --hidden-import cv2.face main.py
```

Use `--onedir` instead of `--onefile` for about 1 s start-up (the single file unpacks itself on each launch, about 3.5 s).

## Engineering notes

- **Responsive UI:** camera capture, preprocessing and face detection run on a worker thread; the Tk thread only
  draws the newest frame. A regression test guards the scheduling bug that previously froze the UI within a second.
- **Cleaner enrolment data:** samples must pass a quality gate (face size, lighting, sharpness relative to recent frames,
  not a near-duplicate), and the 10 % of samples furthest from the mean face are dropped before training.
- **Start-up / memory:** about 0.9 s to a visible window from source, 0.7–1.1 s for the one-folder build, about 65 MB
  idle (measured on the dev machine, camera off).

## Evaluation (status: protocol ready, results not yet published)

**Design targets (goals the system is tuned towards, not measured results):** impostor and attack false-accept rate (FAR) at or below 5 %, genuine false-reject rate (FRR) at or below 15 % in normal indoor light, unlock time of about 10–15 s including the liveness steps.

No FAR/FRR/accuracy figures are claimed here because they have not been measured yet. Each verification trial is
logged with its label (genuine / impostor / photo / phone / replay), lighting and movement tags, the match score and
each liveness step on its own (`presence_pass`, `blink_pass`, `head_pass`, `failed_stage`). Follow
[`docs/TESTING_GUIDE.md`](docs/TESTING_GUIDE.md), then run:

```bash
python experiments/summarize_live_logs.py
```

Offline sweeps over a folder of images are in `experiments/run_experiments.py` (see `docs/EXPERIMENTS.md`).

## Tests

```bash
python -m unittest tests.test_biolock -v
```

25 tests: camera worker, enrolment quality gate, outlier pruning, model save/reload, metrics, liveness logging, and the
external-drive rules. They use synthetic data and a fake camera; they do not measure real-world recognition accuracy.

## Limitations (please read)

- Classical methods (Haar cascades, LBPH). Not designed to resist high-quality deepfakes or 3-D masks.
- The vault is a hidden `SecureVault/` folder plus an encrypted key marker. It is **not** full-disk encryption; use
  BitLocker/VeraCrypt if you need that.
- Needs decent, front-facing light and a webcam; accuracy under other conditions has to be measured (see above).
- Windows only; the exe is unsigned.

## Project layout

```
main.py                   application (UI, camera worker, face engine, liveness, vault)
config.py                 tunable parameters (threshold, weights, frames, ...)
metrics.py                trial logging + FAR/FRR/accuracy/liveness breakdown
experiments/              offline sweep + live-log summariser
tests/                    unit tests (run in CI on Windows)
docs/                     TESTING_GUIDE, EXPERIMENTS, HISTORY
```

## History

The original source was lost and `main.py` was first rebuilt from a compiled build; it has since been substantially
reworked (see [`docs/HISTORY.md`](docs/HISTORY.md)).

Licensed under the MIT License.
