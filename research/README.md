# research/ - how the matcher was chosen

Scripts and outputs of the parameter search behind version 10. Everything here was run on the **development
halves** of the two datasets only (identities at even index in sorted order). The test halves were kept for the
report that `python evaluate.py` produces (`docs/RESULTS.md`).

To reproduce (needs `pip install scikit-learn`; about 1 GB of dataset cache goes to `research/data/`):

1. `python prep.py`        download LFW, detect faces once -> `lfw_images.npy`, `lfw_meta.npz`
2. `python oliv_prep.py`   download Olivetti/ORL, restore the aspect ratio, detect faces and eyes -> `oliv_prep.npz`
3. `python exp1.py`        crop / alignment x normalisation, LBPH only; also writes the eye cache `lfw_eyes.npz`
4. `python exp2.py`        LBPH radius, grid, crop size
5. `python exp3.py`, `python fuse1.py`   other scores (HOG, template, grey histograms) and fusion on LFW
6. `python exp4.py`        both datasets: radius / grid, gallery size, mirror augmentation
7. `python exp5.py dev`    final candidates, one photograph and the median of 3 / 5
8. `python exp6.py eye 6`, `python fuse2.py`   fusion weights and score calibration on both datasets
9. `python exp7.py`        effect of a large jittered gallery (emulated live enrolment) on impostor distances
10. `python exp8_engine.py`  check that `biolock/engine.py` gives the same crops, distances and error rates

`rlib.py` holds the shared code (alignment, Tan-Triggs, features, protocol, metrics); `ds.py` wraps the two
datasets. Results: `exp*.json`, `calibration.json`.

## Findings (development halves)

Protocol: per identity, K random photographs enrol (LFW K=10, Olivetti K=5), the others are genuine probes,
impostor probes are drawn over the other identities of the same half. Score = minus the LBPH chi-square distance
to the nearest enrolled sample. "m3 / m5" = median score of 3 or 5 probes.

| Configuration | LFW, one photo | LFW, median of 5 | Olivetti, one photo | Olivetti, median of 3 |
|---|---|---|---|---|
| version 9 (Haar box, equalise + blur, LBPH r=1, grid 8) | EER 24.3 %, TAR@1%FAR 31.4 % | EER 13.3 %, 46.6 % | EER 5.6 %, 90.1 % | EER 4.0 %, 93.7 % |
| **eye-aligned + Tan-Triggs, LBPH r=5, grid 6 (chosen)** | **EER 17.5 %, 57.0 %** | **EER 6.3 %, 82.7 %** | **EER 3.3 %, 94.8 %** | **EER 2.5 %, 95.3 %** |
| box geometry + Tan-Triggs, r=5, grid 6 | EER 16.7 %, 55.3 % | EER 6.7 %, 80.9 % | EER 3.1 %, 95.8 % | EER 2.6 %, 97.4 % |
| eye-aligned + Tan-Triggs, r=5, grid 8 | EER 18.1 %, 57.3 % | EER 7.4 %, 80.2 % | EER 4.2 %, 94.4 % | EER 3.7 %, 95.6 % |

- Tan-Triggs normalisation and a larger LBPH radius give most of the gain (radius 1 -> 5: LFW EER 22.2 % ->
  18.2 %). Radius 6-7, a 128 px crop, an elliptical mask and CLAHE were not better. More enrolment photographs help
  (K=20: LFW EER 16.2 %).
- The top configurations are within noise of each other. Chosen: eye alignment with box fallback, Tan-Triggs,
  radius 5, 8 neighbours, grid 6x6, 100x100 crop (eye distance 0.46 of the crop width, eye height 0.36).
- Other scores, one photograph (LFW / Olivetti EER): HOG correlation with the mean descriptor 27.6 % / 7.6 %;
  normalised cross-correlation with the mean face 33.3 % / 15.7 %; block-wise grey histograms about 37-41 % on LFW.
  The grey-level histogram score of version 9 is at chance (it compares histograms of equalised crops, which are
  flat by construction).
- Fusion does not help: on a 0.05 weight grid the best mean EER was (0.95, 0.00, 0.05) at 9.78 % against 9.83 %
  for LBPH alone. The manuscript's weights (0.45, 0.20, 0.35) gave LFW EER 23.5 %, version 9's (0.5, 0.3, 0.2)
  21.9 %, LBPH alone 17.1 %.
- Impostor distances are stable across the two datasets (percentiles 1 / 5 / 50 / 95 / 99: LFW 62.6, 64.4, 68.6,
  73.7, 76.6; Olivetti 61.4, 63.4, 68.4, 73.1, 75.5), genuine distances are not (LFW 48.4 ... 74.0, Olivetti
  29.1 ... 64.5). One distance threshold therefore gives about the same false-accept rate in both conditions,
  while the false-reject rate depends on the conditions.
- A 90-sample jittered gallery lowers impostor distances by about 0.7-0.8, so the threshold was set with such a
  gallery (experiment 7).
- Eye detection (both eyes plausible): 75.5 % of LFW faces at webcam scale, 91 % of Olivetti. Mean eye position
  inside a Haar face box: left (0.312, 0.400), right (0.668, 0.392).

The final selection (calibration, weights, threshold) was repeated through the application code by
`experiments/benchmark.py --split dev`; those are the values reported in `docs/RESULTS.md`.
