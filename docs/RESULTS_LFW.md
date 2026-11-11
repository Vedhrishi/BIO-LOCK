# Measured results: face-match stage on LFW

These numbers were produced by running `experiments/benchmark_lfw.py` (included, reproducible) with BioLock's own
`FaceEngine` code on the public **Labeled Faces in the Wild** dataset (62 people with at least 20 photos, 3,023 images).

## What this does and does not show

| Measured here | NOT measured here |
|---|---|
| Identity-verification stage only: Haar detection, preprocessing, LBPH / histogram / template scores, fusion, threshold | Liveness (motion, blink, head turn), presentation attacks (photo, phone, replay) |
| Real photographs of 62 real people | Webcam enrolment/unlock on the target hardware, the vault-key stage, latency/CPU/RAM on the device |

LFW photos are unconstrained (different days, lighting, pose, expression), so enrolling on a few photos and testing on
others is **much harder than a webcam session in one room**. Treat these as a conservative benchmark of the matcher, not as
the accuracy of the complete system. End-to-end, liveness and attack figures still require the live trials in
`docs/TESTING_GUIDE.md`.

## Protocol
- Identities are split in half with a seeded shuffle: a **dev** half (used to choose weights/threshold) and a **test**
  half (all reported numbers). Nothing reported is tuned on the test half.
- Per identity: *K* random photos enrol the model (same `FaceEngine.train`, including outlier pruning); the identity's other
  photos are genuine trials; 150 photos of other identities are impostor trials.
- Face detection found one face in 2,915 of 3,023 images, none in 32 (excluded) and several in 76 (largest taken).
- Test half sizes: 516-1,325 genuine and 3,450-4,650 impostor trials per run. The 95 % intervals in the JSON files are Wilson
  intervals that treat trials as independent; trials from one identity are correlated, so they are optimistic.

## Results (test half; K = 10 enrolment photos, mean of 3 seeds, range in brackets; K = 20 is one seed)

| Configuration | FAR | FRR | EER | AUC |
|---|---|---|---|---|
| Shipped default weights 0.5 / 0.3 / 0.2, threshold 40 | 9.5 % [7.5-12.6] | 52.1 % [46.5-61.6] | 30.4 % | 0.754 |
| Same, K = 20 | 15.8 % | 33.1 % | 25.3 % | 0.808 |
| Paper weights 0.45 / 0.20 / 0.35, threshold 40 | 38.0 % [32.8-46.3] | 25.9 % [23.6-29.7] | 30.5 % | 0.758 |
| LBPH only, threshold 40 | 0.0 % | 99.5 % | 25.3 % | 0.814 |

Single-component discriminability (AUC, 0.5 = chance; K = 10 mean): **LBPH 0.814**, template 0.673, histogram 0.490.

## What these results say
1. **Fusion did not help on this benchmark.** LBPH alone ranks genuine above impostor best (AUC 0.81); adding histogram
   correlation (near chance, AUC about 0.5) and template matching does not improve it, and the fused score has a worse equal-error
   rate (about 30 % vs 25 %). The design claim that fusion reduces dependence on a single recognition output is **not
   supported** here.
2. **The threshold of 40 is not a good operating point on LFW-style conditions.** With the shipped weights it rejects about half of
   genuine trials (FRR 52 %); with the paper's weights it accepts about a third of impostors (FAR 38 %). With LBPH alone it rejects
   nearly everything. The score scale depends on the weights, so the threshold must be calibrated for the weights and the
   conditions, which the live trials should do.
3. More enrolment photos help (K = 20 lowers EER from 30 % to 25 %), consistent with the 100-sample enrolment used in the app.
4. These results do not say how the system performs on your webcam in a fixed setting, which is likely better, and they say
   nothing about attacks.

Reproduce: `python experiments/benchmark_lfw.py --k 10 --seed 0` (downloads LFW through scikit-learn, about 200 MB, a few minutes).
