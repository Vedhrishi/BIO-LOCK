# BioLock SSD

[![tests](https://github.com/Vedhrishi/BIO-LOCK/actions/workflows/tests.yml/badge.svg)](https://github.com/Vedhrishi/BIO-LOCK/actions/workflows/tests.yml)

Face-recognition lock for **external drives** (USB flash drives, external SSD/HDD, SD cards). The app lives on
the drive, hides a `SecureVault` folder, and shows it again only after the enrolled face has passed three
liveness steps and a face match. Windows desktop app: Python, OpenCV, Tkinter.

Author: Pasupuleti Vedh Rishi

## How it works

Every unlock runs four steps in order. A frame only counts if exactly one face is in it.

1. **Presence.** The picture inside the face box has to change from frame to frame (you are asked to move your
   head gently). Measured on block-averaged pixels so that camera noise does not count as movement.
2. **Blink.** Two blinks: the eyes are seen, disappear for a few frames, and come back, while the head stays
   where it is and the picture changes mainly around the eyes.
3. **Head movement.** The face moves 10 % of its width to the left, then to the right.
4. **Face match.** Haar face detection, eye alignment, Tan-Triggs illumination normalisation, LBPH (radius 5,
   8 neighbours, 6x6 grid). The score is the distance to the nearest enrolled sample mapped to 0-100. The
   **median of 15 frames** must reach **35**, and so must the frames seen during the blink step, so the liveness
   steps and the match belong to the same face. The score is a similarity, not a probability.

Other behaviour:

- One enrolled face per drive: 100 quality-checked samples; only features are stored, no images.
- A key file ties the enrolment to its drive (template, drive identifier and volume serial number, PBKDF2-SHA256
  with 600,000 iterations, Fernet). A swapped template or a copy on another drive does not unlock.
- Five failed face matches erase the enrolment and the vault. Liveness time-outs are not counted, but repeated
  ones cause a waiting period.
- The vault locks itself after 5 minutes without activity, and when the app starts.
- Audit log of enrolments, unlocks, failures and wipes.

### External drives only

BioLock refuses to run from an internal drive: *"This cannot be used to lock an internal drive, as it would
endanger the storage components of the system."* A drive counts as external only if Windows reports it as
removable, or as fixed but attached over USB, FireWire, SD or MMC. Anything else, including anything unknown,
is treated as internal.

## Run it

**Packaged app** (no Python needed, 64-bit Windows 10/11): copy `BioLock.exe`, or the `BioLock_Fast` folder, to
an external drive and start it there. SmartScreen may warn because the program is not code-signed: *More info*,
then *Run anyway*. `BioLock.exe --selftest report.json` checks an installation without using the camera.

**From source** (Python 3.11 or 3.12, Windows, a webcam):

```bash
pip install -r requirements.txt
python main.py
```

From source the external-drive check is off, so the code can be developed on an internal disk;
`BIOLOCK_ENFORCE_EXTERNAL=1` switches it on.

**Build the exe and the archives:**

```bash
pip install -r requirements.txt -r requirements-dev.txt
python build.py
```

## What has been measured

`python evaluate.py` produces every number in [`docs/RESULTS.md`](docs/RESULTS.md). In short:

<!-- results:start -->
**Identity verification**, test halves of two public face datasets, through the app's own code. Parameters were chosen on the development halves only. One photograph per decision, enrolment emulating the app's 100 frames, 95 % intervals in brackets:

|  | Olivetti / ORL (controlled) | LFW (unconstrained) |
|---|---|---|
| Equal-error rate | 2.5 % | 18.4 % |
| Impostor photographs accepted at the shipped threshold (35) | 1.47 % (0.47 to 2.76 %) | 0.47 % (0.28 to 0.69 %) |
| Genuine photographs rejected at the shipped threshold | 3.3 % (1.2 to 6.5 %) | 50.1 % (44.9 to 55.8 %) |
| Equal-error rate when the median of 5 photographs decides | 0.0 % | 6.8 % |
| Equal-error rate, version 9 matcher -> version 10 (gallery of K photographs) | 6.7 % -> 3.0 % | 25.3 % -> 19.9 % |

- The matcher works on controlled photographs and rejects about half of the genuine ones on unconstrained press photographs at this threshold. A webcam session is closer to the controlled case, but that has not been measured.
- On the test halves none of the other weightings tried has a lower equal-error rate than the shipped weights on both datasets. The weighted fusion described in the manuscript is not supported by these measurements.
- **Simulated sessions** (animated photographs of the test halves, not people): a motionless photograph of the enrolled person was accepted in 0 of 92; a trembling one that never blinks in 0 of 92; liveness done by one face and the match by another in 1 of 64; the same with the identity binding switched off in 39 of 63; another person performing all cues in 2 of 208; an unseen photograph of the enrolled person performing all cues in 104 of 147 sessions.
- The eye check that the blink step depends on did not see the open eyes steadily in 61 of 387 photographs tried (16 %). Whether that also happens with a webcam has to be found out in the live trials.
- **Speed** on a 13th Gen Intel(R) Core(TM) i7-13700H: 13 ms per frame in the blink step and 26 ms per frame while matching; 97 MB of memory.
<!-- results:end -->

## What has not been measured

Liveness and presentation attacks with real people and real attack material. The simulated sessions in
`docs/RESULTS.md` use animated photographs; they show that the session logic stops a photograph that does not
blink and that the steps are tied to one face, nothing more. A replayed video of the enrolled person that shows
the blinks and the head movement is **not** stopped by these cues. The protocol for the live trials is in
[`docs/TESTING_GUIDE.md`](docs/TESTING_GUIDE.md); each attempt is logged, and

```bash
python experiments/summarize_live_logs.py
```

turns the log into FAR, FRR and per-attack tables. Three liveness thresholds were set from simulation and still
have to be confirmed by those trials.

## Tests

```bash
python -m unittest discover -s tests -t .
```

<!-- tests:start -->143 tests.<!-- tests:end --> They use drawn faces and fake cameras, so they run without a webcam, a dataset or a network connection.
They check that the software behaves as described, not how well it recognises people.

## Limitations

- Classical methods (Haar cascades, LBPH). On unconstrained photographs the matcher rejects a large share of
  genuine attempts at the shipped threshold; see the results.
- The liveness cues are simple. They do not resist a replayed video, a 3-D mask or a deepfake.
- `SecureVault` is a hidden folder, **not** an encrypted one. Anyone who can read the drive with other tools can
  read the files. Use BitLocker or VeraCrypt if you need encryption.
- The state on the drive (attempt counter, template, key) is not protected against someone who edits the files.
- Windows only, 64-bit, not code-signed.
- Version 10 has so far only been run against simulated cameras. The first runs with a real webcam and a real
  USB drive are part of the live trials.

## Project layout

```
main.py                 launcher (also: --selftest)
biolock/                the application
  ui.py                 window
  controller.py         what an attempt's outcome does to the drive
  session.py            presence -> blink -> head -> face, frame by frame
  engine.py             detection, alignment, template, matching
  liveness.py           motion, blink and head-movement detectors
  camera.py             camera access on worker threads
  vault.py, storage.py  key file; files kept on the drive
  metrics.py            attempt log, FAR/FRR arithmetic
  sim.py                animated photographs for tests and simulations
  winutil.py            Windows specifics (attributes, drive type, DPI)
experiments/            benchmark, simulation, performance, live-log summary
evaluate.py             runs all of them and writes docs/RESULTS.md
build.py                builds the exe and the archives
tests/                  unit tests
research/               the parameter search behind the matcher
docs/                   RESULTS, TESTING_GUIDE, PAPER_ALIGNMENT, EXPERIMENTS, HISTORY
results/                raw output of evaluate.py
```

## History

The original source was lost and version 9 was rebuilt from a compiled build. Version 10 is a rewrite; what
changed and why is in [`docs/HISTORY.md`](docs/HISTORY.md).

Licensed under the MIT License.
