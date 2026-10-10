# Running the parameter sweeps and getting metrics for the paper

This describes what changed to support the experiments your mentors
asked for, and exactly how to produce real numbers with it.

**Important:** nothing here can compute FAR/FRR/accuracy/etc. without
real captured data — real face images, a real printed photo, a real
phone display, and a real recorded replay video. The code below does
the measuring and the arithmetic; you still have to go collect the
samples described in `experiments/run_experiments.py`'s docstring. Any
numbers you see in example output anywhere in this repo (e.g. in this
file) are from a synthetic dry-run used only to prove the code runs
end-to-end — they are not real biometric results and must not be used
in the paper.

## What changed in the code

- **`config.py`** — every tunable parameter (`min_conf`, `motion_thresh`,
  `enrol_frames`, the three fusion weights, etc.) now lives here instead
  of as a hard-coded constant, loaded from `config.json` / environment
  variables / explicit overrides. `main.py` reads these at startup.
- **`FaceEngine.predict()`** in `main.py` — now a weighted fusion of
  three independent match signals instead of LBPH alone:
  - **LBPH** distance -> confidence (as before)
  - **Histogram** correlation of the face ROI against the mean
    histogram of the enrolled samples
  - **Template** match (normalized cross-correlation) against the
    enrolled mean-face image

  `fused_confidence = w_lbph*LBPH + w_hist*Hist + w_template*Template`,
  with `w_lbph + w_hist + w_template == 1` (enforced by
  `config.validate()`). This is the only change to the matching logic;
  the 4-step liveness pipeline around it is untouched.
- **`metrics.py`** — new module, logs:
  - every processed frame's face count (0 / 1 / 2+) to `logs/detections.csv`
  - every completed authentication attempt (accept/reject, the three
    individual sub-scores, latency, CPU%, memory, and whatever
    `min_conf`/`motion_thresh`/`enrol_frames`/weights were active) to
    `logs/attempts.csv`
  - `compute_metrics()` turns a set of logged attempts into FAR, FRR,
    accuracy, precision, recall, F1
- **Testing Mode panel** (in the app's sidebar) — three dropdowns:
  `Trial label` (genuine / impostor / attack_photo / attack_phone /
  attack_replay), `Lighting condition`, `Movement condition`. Set these
  before each attempt so the log row records what you actually tested.
  **These tags are for logging only** — they never affect the
  accept/reject decision itself.

## Two ways to run the experiments

### A. Offline batch sweep (`experiments/run_experiments.py`)

Fastest way to get the full parameter-range tables (min_conf 20-60,
motion_thresh 0.01-0.10, enrol_frames 10/20/40/60/80/100, and the full
LBPH/Hist/Template weight-combination grid) from a folder of captured
images, without re-running the camera for every combination (match
scores are computed once per image, then re-thresholded/re-weighted in
memory for every combination — this is why its latency column is
labelled "scoring latency", not end-to-end camera latency).

```bash
python experiments/run_experiments.py --dataset /path/to/your/dataset --out experiments/results
```

Read the big docstring at the top of that file for the exact dataset
folder layout and what to physically go capture (live photos per
lighting condition, a printed photo, a phone-displayed photo, frames
extracted from a replay video).

Outputs (all in `experiments/results/`):
- `min_conf_sweep.csv` — FAR/FRR/accuracy/.../latency/CPU/memory at each min_conf 20-60
- `motion_thresh_sweep.csv` — whether each motion_thresh 0.01-0.10 would block the
  phone/replay attack sequences you captured (needs sequential frames, see docstring)
- `enrol_frames_sweep.csv` — same metrics at each of 10/20/40/60/80/100 enrolment frames
- `weight_combo_sweep.csv` — same metrics across the LBPH/Hist/Template weight grid
- `detection_counts.csv` — 0/1/multi-face tally per folder (the "0/1/multiple-face
  detection" logging your mentors asked for)
- `condition_breakdown.csv` — FAR/FRR/etc. per lighting × movement folder
- `attack_breakdown.csv` — accept rate per presentation-attack type

### B. Live app sessions (`experiments/summarize_live_logs.py`)

Use this if your mentors want numbers from the real, full liveness
pipeline (blink + head-turn included), not just static-image face
matching. Run `main.py`, set the Testing Mode panel before each
attempt, physically perform it (stand in front of the camera / hold up
the photo / hold up the phone / play the replay video), then run:

```bash
python experiments/summarize_live_logs.py
```

which prints the same metric tables computed from `logs/attempts.csv`
and `logs/detections.csv` — i.e., from attempts you actually performed.

## Recommended split of effort

- Use **A** for the big min_conf / motion_thresh / enrol_frames /
  weight-combination sweeps — doing those live, one physical attempt
  per parameter value, would take enormous time for little extra
  validity once you have a decent still-image dataset per condition.
- Use **B** to report a smaller number of full-pipeline, end-to-end
  numbers (overall accuracy, overall latency including all 4 steps,
  live presentation-attack rejection rate) that reflect the complete
  system rather than just the face-match stage — this is usually what
  a project report wants headline numbers for.

## Latency / CPU / memory caveats, in the interest of not overstating them

- **Offline (A)**: "latency" is per-image match scoring only (LBPH +
  histogram + template, no camera I/O, no liveness steps). Useful for
  comparing weight combinations' relative cost, not representative of
  real unlock time.
- **Live (B)**: latency is the full `_start()` → accept/reject wall-clock
  time, which includes however long you personally took to blink and
  turn your head — report it as "end-to-end interaction time," not pure
  algorithm latency, if asked to characterise it precisely.
- CPU%/memory figures reflect the whole Python process at that instant,
  not an isolated measurement of just the face-matching code — normal
  for this kind of lightweight instrumentation, but worth stating as a
  limitation if your mentors probe on methodology.
