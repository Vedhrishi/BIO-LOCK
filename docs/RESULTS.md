# BioLock 10.0: measured results

Written by `python evaluate.py` on 2026-11-12. Every number below is read from the JSON files that command puts in `results/`; none was typed in. Run it again to check them.

## 1. What these results cover

| Part of the system | Status | Source |
|---|---|---|
| Identity verification (detection, alignment, LBPH match, threshold) | measured | two public face datasets, test halves, `experiments/benchmark.py` |
| Session logic (presence, blink, head movement, match, identity binding) | simulated | animated photographs, `experiments/simulate.py` |
| Resistance to photographs / replay | simulated for photographs only | animated photographs; a replay that shows the cues is not stopped |
| Speed and memory | measured on one machine | `experiments/perf.py` |
| Software behaviour (lock-out, wipe, key check, drive rule) | unit tests | `tests/` |
| Real people, a real webcam, printed photos, phones, replayed video | **not measured** | needs the live trials in `docs/TESTING_GUIDE.md` |

A simulation is not a measurement of people. Section 5 says what the simulated sessions can and cannot show.

## 2. Protocol

- Datasets: LFW (the 62 people with at least 20 photographs; unconstrained press photographs) and Olivetti / ORL (40 people, 10 photographs each, taken in one laboratory).
- Identities are split by position in sorted order: even index = development half (parameters are chosen there), odd index = test half (reported here). Nothing was tuned on the test halves.
- Each photograph is scaled so that the face is about 190 px wide and placed in a 640x480 frame. From there the application code runs unchanged: frame preparation, Haar face detection, eye detection, alignment, Tan-Triggs normalisation, LBPH (radius 5, 8 neighbours, 6x6 grid), threshold.
- Per identity and seed, K random photographs enrol; the other photographs of that person are genuine probes (at most 30 for LFW, 5 for Olivetti); 150 (LFW) or 120 (Olivetti) impostor probes are drawn uniformly over the other identities of the same half.
- Two galleries: the K photographs alone, and an emulated live enrolment (each photograph plus jittered, noisy copies that go through detection again, about 90 samples, like the 100-frame enrolment of the app). The threshold was set for the second one.
- "Median of m" means the median score of m different photographs of one person. It is not the same as the 15 consecutive camera frames the app uses, which are much more alike.
- A photograph without a detected face is left out. If several faces are detected the one nearest the centre is used (the app itself refuses such frames).
- Intervals in brackets are 95 % intervals from resampling the enrolled identities 2,000 times. Trials of one identity are correlated, so per-trial intervals would be too narrow.

Test halves:

| Dataset | Identities | Photographs | K | Seeds | Face found | Several faces | Both eyes found | Mean face width |
|---|---|---|---|---|---|---|---|---|
| Olivetti / ORL (controlled) | 20 | 200 | 5 | 6 | 90.5 % | 1 | 87.8 % | 203 px |
| LFW (unconstrained) | 31 | 1166 | 10 | 3 | 98.1 % | 133 | 70.5 % | 190 px |

## 3. Parameters chosen on the development halves

Rules were fixed before the test halves were run (they are written at the top of `experiments/benchmark.py` and `experiments/simulate.py`).

**Score calibration** (maps each raw measure to 0-100; pooled 1st / 99th percentiles):

| Constant | Shipped | Re-derived through the app code |
|---|---|---|
| lbph_d_lo | 31.2 | 31.422 |
| lbph_d_hi | 76.4 | 75.865 |
| hist_c_lo | 0.114 | 0.090 |
| hist_c_hi | 0.885 | 0.882 |
| tmpl_c_lo | -0.083 | -0.086 |
| tmpl_c_hi | 0.649 | 0.641 |

**Fusion weights** (LBPH, histogram, template). Rule: lowest mean equal-error rate over the two development sets on a 0.05 grid.

| Weights | Mean EER | EER LFW | EER Olivetti |
|---|---|---|---|
| (1.0, 0.0, 0.0) | 10.35 % | 18.2 % | 2.5 % |
| (0.95, 0.0, 0.05) | 10.42 % | 18.2 % | 2.7 % |
| (0.95, 0.05, 0.0) | 10.55 % | 18.6 % | 2.5 % |
| (0.9, 0.0, 0.1) | 10.58 % | 18.5 % | 2.7 % |
| (0.9, 0.05, 0.05) | 10.59 % | 18.6 % | 2.6 % |
| (0.45, 0.2, 0.35) (manuscript) | 14.86 % | 23.6 % | 6.1 % |
| (0.5, 0.3, 0.2) (version 9) | 13.78 % | 22.4 % | 5.1 % |

The grid minimum is LBPH alone: no mix with the histogram or template score has a lower mean equal-error rate. The mean rate is 4.5 percentage points higher for the weights of the manuscript and 3.4 percentage points higher for the weights of version 9. Shipped: (1, 0, 0).
Each development set on its own: LFW: best grid point (0.95, 0.0, 0.05) with EER 18.15 %, LBPH alone 18.16 %; Olivetti: best grid point (0.85, 0.15, 0.0) with EER 2.48 %, LBPH alone 2.54 %.

**Threshold.** Rule: the smallest whole number T such that at most 1 % of single-photograph impostor probes score T or more, emulated live enrolment, on each development set.

| Development set | Impostor probes | Score exceeded by 1 % | T | FAR at T | FRR at T |
|---|---|---|---|---|---|
| LFW | 13950 | 33.36 | 34 | 0.72 % | 43.5 % |
| Olivetti | 14400 | 34.38 | 35 | 0.85 % | 3.2 % |

Shipped: T = 35 (the larger of the two), which is an LBPH chi-square distance of 60.6.

**Liveness thresholds (simulation).** These three values were set from animated photographs of the development halves (234 photographs). They are starting values: only live trials can confirm them.

Presence measure (mean absolute frame difference inside the face box on 8x8 block-averaged pixels, mean of 20 frames):

| Input | n | 1st pct | median | 99th pct |
|---|---|---|---|---|
| motionless photograph, sensor noise 2 | 234 | 0.0008 | 0.0018 | 0.0041 |
| motionless photograph, noise 5 | 234 | 0.0017 | 0.0038 | 0.0086 |
| motionless photograph, noise 8 | 234 | 0.0027 | 0.0055 | 0.0119 |
| sway of 1.5 px | 233 | 0.0018 | 0.0034 | 0.0060 |
| sway of 5 px | 234 | 0.0051 | 0.0087 | 0.0154 |
| sway of 10 px | 234 | 0.0099 | 0.0166 | 0.0300 |
| sway of 20 px | 234 | 0.0189 | 0.0319 | 0.0585 |
| trembling photograph | 233 | 0.0153 | 0.0277 | 0.0534 |

Rule: geometric mean of the 99th percentile for a motionless photograph with noise 5 (0.0086) and the 1st percentile for a 10 px sway (0.0099) = 0.0092. Shipped `motion_thresh` = 0.009.

Blink evidence at the moment the eye detector loses the eyes (change in the eye band, change in the rest of the face, their ratio; medians and 99th percentiles):

| Event | Events | Eye band: median | 99th pct | Ratio: median | 99th pct | Counted as a blink by the shipped rule |
|---|---|---|---|---|---|---|
| simulated blink (eyes painted over) | 382 | 0.0611 | 0.2829 | 19.33 | 39.46 | 350 of 382 |
| trembling photograph | 386 | 0.0394 | 0.1246 | 1.26 | 2.16 | 1 of 386 |
| motionless photograph, noise 5 | 186 | 0.0047 | 0.0106 | 1.05 | 1.38 | 0 of 186 |

Rules: `blink_local_ratio` = 99th percentile of the ratio for a trembling photograph (2.16) rounded up to 0.5 = 2.5; `blink_min_change` = 99th percentile of the eye-band change from noise alone (0.0106) rounded up = 0.011.

## 4. Identity verification on the test halves

Shipped configuration: LBPH radius 5, 8 neighbours, grid 6x6, weights (1, 0, 0), threshold T = 35. FAR = impostor probes accepted, FRR = genuine probes rejected, both at T. EER, AUC and TAR at 1 % FAR describe the score itself, independent of T (mean over the seeds).

**Olivetti / ORL (controlled)** (18 enrolled identities, 108 enrolments over 6 seeds)

| Gallery | Decision on | Genuine | Impostor | EER | AUC | TAR at 1 % FAR | FAR at T=35 | FRR at T=35 |
|---|---|---|---|---|---|---|---|---|
| K photographs | one photograph | 510 | 12960 | 3.0 % | 0.994 | 95.1 % | 0.79 % (0.18 to 1.62 %) | 5.5 % (1.8 to 9.9 %) |
| K photographs | median of 3 photographs | 1080 | 1944 | 1.2 % | 1.000 | 99.0 % | 0.36 % (0.05 to 0.72 %) | 1.5 % (0.0 to 3.4 %) |
| K photographs | median of 5 photographs | 840 | 1944 | 0.1 % | 1.000 | 100.0 % | 0.31 % (0.00 to 0.62 %) | 0.0 % (0.0 to 0.0 %) |
| emulated live enrolment | one photograph | 510 | 12960 | 2.5 % | 0.996 | 96.5 % | 1.47 % (0.47 to 2.76 %) | 3.3 % (1.2 to 6.5 %) |
| emulated live enrolment | median of 3 photographs | 1080 | 1944 | 0.8 % | 1.000 | 99.4 % | 1.23 % (0.31 to 2.42 %) | 0.8 % (0.0 to 2.3 %) |
| emulated live enrolment | median of 5 photographs | 840 | 1944 | 0.0 % | 1.000 | 100.0 % | 1.08 % (0.15 to 2.31 %) | 0.0 % (0.0 to 0.0 %) |

**LFW (unconstrained)** (31 enrolled identities, 93 enrolments over 3 seeds)

| Gallery | Decision on | Genuine | Impostor | EER | AUC | TAR at 1 % FAR | FAR at T=35 | FRR at T=35 |
|---|---|---|---|---|---|---|---|---|
| K photographs | one photograph | 1926 | 13950 | 19.9 % | 0.871 | 51.2 % | 0.22 % (0.08 to 0.39 %) | 59.3 % (54.0 to 65.2 %) |
| K photographs | median of 3 photographs | 930 | 2790 | 12.4 % | 0.947 | 65.2 % | 0.04 % (0.00 to 0.11 %) | 61.2 % (52.2 to 70.5 %) |
| K photographs | median of 5 photographs | 930 | 2790 | 9.5 % | 0.970 | 75.9 % | 0.00 % (0.00 to 0.00 %) | 64.0 % (54.7 to 73.8 %) |
| emulated live enrolment | one photograph | 1926 | 13950 | 18.4 % | 0.882 | 55.6 % | 0.47 % (0.28 to 0.69 %) | 50.1 % (44.9 to 55.8 %) |
| emulated live enrolment | median of 3 photographs | 930 | 2790 | 10.4 % | 0.959 | 69.0 % | 0.07 % (0.00 to 0.18 %) | 51.5 % (42.8 to 60.6 %) |
| emulated live enrolment | median of 5 photographs | 930 | 2790 | 6.8 % | 0.979 | 81.7 % | 0.04 % (0.00 to 0.11 %) | 50.2 % (39.9 to 60.8 %) |

Development versus test (one photograph per decision). The threshold was chosen on the development halves, so the test columns are the honest ones:

| Dataset | Gallery | EER dev | EER test | FAR dev | FAR test | FRR dev | FRR test |
|---|---|---|---|---|---|---|---|
| Olivetti / ORL (controlled) | K photographs | 2.5 % | 3.0 % | 0.33 % | 0.79 % | 5.2 % | 5.5 % |
| Olivetti / ORL (controlled) | emulated live enrolment | 2.3 % | 2.5 % | 0.85 % | 1.47 % | 3.2 % | 3.3 % |
| LFW (unconstrained) | K photographs | 18.2 % | 19.9 % | 0.29 % | 0.22 % | 56.4 % | 59.3 % |
| LFW (unconstrained) | emulated live enrolment | 17.6 % | 18.4 % | 0.49 % | 0.47 % | 47.3 % | 50.1 % |

**Does fusing three scores help?** One photograph per decision, K-photograph gallery, test halves. The threshold T belongs to the shipped weights; the other rows show what the same T would do with other weights.

| Score | EER Olivetti | FAR at T | FRR at T | EER LFW | FAR at T | FRR at T |
|---|---|---|---|---|---|---|
| shipped (1, 0, 0) | 3.0 % | 0.79 % | 5.5 % | 19.9 % | 0.22 % | 59.3 % |
| development grid, rank 2 (0.95, 0, 0.05) | 3.0 % | 0.83 % | 6.7 % | 19.9 % | 0.29 % | 58.8 % |
| development grid, rank 3 (0.95, 0.05, 0) | 3.1 % | 1.20 % | 4.3 % | 19.9 % | 0.50 % | 53.8 % |
| manuscript (0.45, 0.2, 0.35) | 6.7 % | 13.86 % | 4.3 % | 24.3 % | 13.99 % | 32.9 % |
| version 9 (0.5, 0.3, 0.2) | 5.3 % | 18.78 % | 2.4 % | 22.8 % | 23.26 % | 23.1 % |
| LBPH distance alone | 3.0 % |  |  | 19.9 % |  |  |
| histogram (HOG) correlation alone | 10.3 % |  |  | 29.8 % |  |  |
| template correlation alone | 20.2 % |  |  | 35.0 % |  |  |

On the test halves none of the other weightings tried has a lower equal-error rate than the shipped weights on both datasets. The weighted fusion described in the manuscript is not supported by these measurements.

**Against the version 9 matcher** (box crop, double histogram equalisation and blur, LBPH radius 1, 8x8 grid) on the same detections, K-photograph gallery:

| Dataset | EER v9 | EER v10 | TAR at 1 % FAR v9 | TAR at 1 % FAR v10 | EER v9, median of 5 | EER v10, median of 5 |
|---|---|---|---|---|---|---|
| Olivetti / ORL (controlled) | 6.7 % | 3.0 % | 85.7 % | 95.1 % | 0.6 % | 0.1 % |
| LFW (unconstrained) | 25.3 % | 19.9 % | 26.0 % | 51.2 % | 15.1 % | 9.5 % |

## 5. Simulated sessions on the test halves (SIMULATION, not live trials)

Photographs of the test halves are animated and fed frame by frame to the session code the app runs behind the camera, with a clock that advances 1/30 s per frame. Enrolment is the emulated live enrolment of section 2. The simulated person sways 12 px during the presence step, blinks (the eye regions are painted over for 4 frames) when asked, moves left then right, then holds still. Time-outs and thresholds are the shipped ones.

**What this can show:** whether the session logic stops inputs that lack a cue, what the matcher does at the end of a complete session, and that the steps are tied to one face.

**What it cannot show:** how real faces, real blinks, a printed photograph, a phone screen or a camera behave. An animated photograph of the enrolled person that performs the cues is accepted like the person: for this pipeline it is the same input as a replayed video. The blink, motion and head-movement cues do not stop such a replay.

**Olivetti / ORL (controlled)**

| Simulated input | Sessions | Accepted | Accept rate (95 % interval) | Stopped at presence | at blink | at head | Rejected at match |
|---|---|---|---|---|---|---|---|
| enrolled person, all cues | 64 | 60 | 93.8 % (85.0 to 97.5 %) | 0 | 1 | 0 | 3 |
| another person, all cues | 86 | 1 | 1.2 % (0.2 to 6.3 %) | 0 | 2 | 0 | 83 |
| motionless photograph of the enrolled person, noise 2 | 18 | 0 | 0.0 % (0.0 to 17.6 %) | 18 | 0 | 0 | 0 |
| motionless photograph, noise 5 | 18 | 0 | 0.0 % (0.0 to 17.6 %) | 18 | 0 | 0 | 0 |
| trembling photograph, moved when asked, no blink | 36 | 0 | 0.0 % (0.0 to 9.6 %) | 0 | 36 | 0 | 0 |
| liveness by another face, then the enrolled face for the match | 30 | 1 | 3.3 % (0.6 to 16.7 %) | 0 | 0 | 0 | 29 |
| the same with identity binding switched off | 29 | 27 | 93.1 % (78.0 to 98.1 %) | 0 | 1 | 0 | 1 |

Of the genuine sessions rejected at the match, 0 had a passing match-stage score and failed only the identity binding (the blink-stage frames scored under the threshold).

A photograph can only play a blinking part if exactly one face is detected in it, the eye check sees its open eyes in at least 5 of 6 noisy frames, and it stops seeing them once they are painted over. Of 122 distinct photographs tried, 104 qualified; the others: 10 open eyes not detected steadily; 7 painted-over eyes still detected; 1 several faces detected.

Sessions that were planned but not run because a photograph did not qualify: 7 x enrolled person, all cues, 22 x another person, all cues, 6 x liveness by another face, then the enrolled face for the match, 7 x the same with identity binding switched off.

**LFW (unconstrained)**

| Simulated input | Sessions | Accepted | Accept rate (95 % interval) | Stopped at presence | at blink | at head | Rejected at match |
|---|---|---|---|---|---|---|---|
| enrolled person, all cues | 83 | 44 | 53.0 % (42.4 to 63.4 %) | 0 | 6 | 0 | 33 |
| another person, all cues | 122 | 1 | 0.8 % (0.1 to 4.5 %) | 0 | 8 | 0 | 113 |
| motionless photograph of the enrolled person, noise 2 | 26 | 0 | 0.0 % (0.0 to 12.9 %) | 26 | 0 | 0 | 0 |
| motionless photograph, noise 5 | 30 | 0 | 0.0 % (0.0 to 11.4 %) | 30 | 0 | 0 | 0 |
| trembling photograph, moved when asked, no blink | 56 | 0 | 0.0 % (0.0 to 6.4 %) | 0 | 56 | 0 | 0 |
| liveness by another face, then the enrolled face for the match | 34 | 0 | 0.0 % (0.0 to 10.2 %) | 0 | 3 | 0 | 31 |
| the same with identity binding switched off | 34 | 12 | 35.3 % (21.5 to 52.1 %) | 0 | 2 | 0 | 20 |

Of the genuine sessions rejected at the match, 1 had a passing match-stage score and failed only the identity binding (the blink-stage frames scored under the threshold).

A photograph can only play a blinking part if exactly one face is detected in it, the eye check sees its open eyes in at least 5 of 6 noisy frames, and it stops seeing them once they are painted over. Of 265 distinct photographs tried, 172 qualified; the others: 51 open eyes not detected steadily; 18 painted-over eyes still detected; 24 several faces detected.

Sessions that were planned but not run because a photograph did not qualify: 41 x enrolled person, all cues, 64 x another person, all cues, 6 x trembling photograph, moved when asked, no blink, 6 x motionless photograph, 28 x liveness by another face, then the enrolled face for the match, 28 x the same with identity binding switched off.

Intervals here are Wilson 95 % intervals over sessions. The genuine accept rate is mostly the matcher on one unseen photograph per session (compare the FRR for "one photograph" in section 4); it says nothing about how often a live person completes the blink step.

"Open eyes not detected steadily" is a property of the Haar eye detector on these photographs (small or turned faces, glasses). It matters for live use: someone whose open eyes it does not see steadily cannot complete the blink step. How often that happens in front of a webcam has to be measured live.

## 6. Speed and memory

Machine: 13th Gen Intel(R) Core(TM) i7-13700H, 20 logical CPUs, 15.6 GB RAM, Windows-11-10.0.26200-SP0, Python 3.12.4, OpenCV 4.10.0. Frames are 640x480 with one face about 270 px wide. Other machines will differ.

| Step | Median | 95th percentile |
|---|---|---|
| prepare frame (mirror, grey, equalise) | 0.50 ms | 0.59 ms |
| face detection (half resolution) | 6.97 ms | 7.33 ms |
| eye check for blinks (reduced band) | 2.43 ms | 3.21 ms |
| eye positions for alignment | 10.97 ms | 12.00 ms |
| aligned crop + Tan-Triggs | 0.32 ms | 0.73 ms |
| features of one crop (LBP histogram, HOG) | 0.52 ms | 0.54 ms |
| comparison with 90 enrolled samples | 1.06 ms | 1.20 ms |
| whole match of one frame (eyes, crop, score) | 14.72 ms | 16.99 ms |
| presence measure | 0.12 ms | 0.12 ms |
| blink evidence measure | 0.06 ms | 0.07 ms |

Processing per frame during a complete simulated enrolment (494 frames, 16.5 s at 30 frames/s):

| Stage | Frames | Median per frame | 95th percentile | Frames per second possible |
|---|---|---|---|---|
| presence | 11 | 9.0 ms | 10.9 ms | 160 |
| blink | 40 | 7.8 ms | 15.4 ms | 109 |
| head | 46 | 5.4 ms | 11.7 ms | 169 |
| face | 397 | 10.5 ms | 24.3 ms | 74 |

Processing per frame during a complete simulated verification (112 frames, 3.7 s at 30 frames/s):

| Stage | Frames | Median per frame | 95th percentile | Frames per second possible |
|---|---|---|---|---|
| presence | 11 | 8.5 ms | 10.8 ms | 170 |
| blink | 40 | 12.9 ms | 21.5 ms | 80 |
| head | 46 | 5.0 ms | 11.5 ms | 180 |
| face | 15 | 26.4 ms | 29.1 ms | 38 |

| Memory | MB |
|---|---|
| after imports | 44.6 |
| engine with an enrolled template | 82.4 |
| after two full sessions | 97.4 |

| Start-up and one-off costs | Value |
|---|---|
| import all modules, fresh process (ms) | 221 |
| create engine, load cascades (ms) | 19.7 |
| template file (KB) | 368.3 |
| create engine and load template (ms) | 70.2 |
| derive and store key, PBKDF2 600k (ms) | 93 |
| check key at unlock (ms) | 109 |
| self-test from source (s) | 1.87 |
| self-test of the packaged exe (s) | 5.61 |
| build the template from 100 samples (ms) | 76.6 |

## 7. Unit tests

143 tests, all passed, 59 s, Python 3.12.4. They use drawn faces and fake cameras: they check that the software does what it is meant to (median decision, lock-out and wipe, key check, drive rule, template file checks), not how well it recognises people.

## 8. What is still missing

- Live trials with people and a webcam: genuine users, other people, a printed photograph, a phone screen, a replayed video. Until those exist there is no measured liveness or presentation-attack rate for this system, and the three liveness thresholds of section 3 are unconfirmed.
- The datasets are photographs taken on different occasions, which is harder than one person in front of one webcam, and the Olivetti set is small. Treat the matcher numbers as a benchmark of the method, not as the accuracy a user will see.
- All timings are from one machine.

## 9. Reproduce

```bash
pip install -r requirements.txt -r requirements-dev.txt
python evaluate.py
```

The datasets are downloaded through scikit-learn on first use (about 200 MB, cached in `research/data/`). All random choices are seeded.
