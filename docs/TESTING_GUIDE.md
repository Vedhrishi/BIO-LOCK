# BioLock - validation protocol (answers the reviewer points)

## 1. What the displayed "match score" is
The number shown in the app ("Match score 43/100") is a **weighted similarity score on a 0-100 scale, not a probability**.
It is computed in `FaceEngine.predict()`:

    score = 0.5 * LBPH + 0.3 * Histogram + 0.2 * Template      (weights in config.py, must sum to 1)

| Component | Raw measure | Mapped to 0-100 |
|---|---|---|
| LBPH | LBPH distance to the enrolled model | `100 - distance`, clipped |
| Histogram | correlation of the face's grey histogram with the mean enrolled histogram | `(corr + 1) / 2 * 100` |
| Template | normalised cross-correlation with the mean enrolled face | `(ncc + 1) / 2 * 100` |

It is uncalibrated: it has not been validated as a probability, so do not report it as one. A face is accepted when the
score is >= `min_conf` (default 40). Report only FAR/FRR measured at that threshold.

## 2. Liveness is tested separately from face matching
Every verification trial logs, in `logs/attempts.csv`: `presence_pass`, `blink_pass`, `head_pass`, `liveness_pass`,
and `failed_stage` (`presence` / `blink` / `head` / `match` / empty when accepted).
A trial that does not finish the three liveness steps within 45 s is logged as a rejection at the stage it stopped at
(it does not count toward the lockout). Run `python experiments/summarize_live_logs.py` to get the per-attack table
"Liveness per trial label".

## 3. Procedure for the trials (do this on the real app)
1. Enrol (good light, 100 samples).
2. In **Testing mode**, tick *Disable lockout/wipe while testing* (otherwise 5 rejections wipe your enrolment).
3. For each condition set **Trial label** + lighting + movement, then press unlock and perform the trial:

| Trial label | What you present | Target N |
|---|---|---|
| genuine | your live face | 30 |
| impostor | a different live person (3+ people) | 30 |
| attack_photo | printed photo of the enrolled face | 30 |
| attack_phone | photo on a phone screen | 30 |
| attack_replay | video of the enrolled face played on a screen | 30 |

The panel shows "Trials logged: N (genuine n, ...)" so you can confirm each trial was recorded.
4. Run `python experiments/summarize_live_logs.py` and report FAR, FRR, accuracy, per-attack accept rate, liveness pass
rate, and the N and conditions for each row. Untick the testing checkbox afterwards.

## 4. Reporting rules
- Report numbers only from `logs/attempts.csv`; state N per cell and the lighting/movement tags.
- With N=30 per cell the 95 % interval is wide (e.g. 0/30 false accepts still allows a true rate of ~10 %); say so.
- The "Failed unlock attempts: x/5" counter is the lockout counter, not the trial counter.
