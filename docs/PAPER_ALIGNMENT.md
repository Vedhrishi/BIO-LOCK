# Paper vs. code: what matches, what does not

Checked against `ICSCNA_298.docx` ("BioLock SSD: A Multi-Stage Biometric Authentication Framework for Secure Removable Storage").
Every item below must be true of the shipped code *and* of the paper before resubmission.

## Mismatches found (and status)

| # | Paper says | Code did / does | Status |
|---|---|---|---|
| 1 | Fusion weights C = 0.45·LBPH + 0.20·Hist + 0.35·Temp (Eq. 6, Table II, Section VII-A) | Shipped default is 0.5 / 0.3 / 0.2 (`config.py`) | **Open, your decision.** Either edit the paper to 0.5/0.3/0.2, or set `config.py` to the paper's values. Measured effect is in `RESULTS_LFW.md`: at threshold 40 the paper weights give FAR about 38 % / FRR about 26 %, the shipped weights FAR about 9.5 % / FRR about 52 %. Neither is better in equal-error terms. |
| 2 | Head-movement baseline set over 10 calibration observations | Baseline was the first single observation | **Fixed in code**: baseline = mean of first 10 observations (`HeadTurn.CALIB_OBS`), covered by tests. |
| 3 | Abstract: "controlled experimental measurements are not available". Section VI/VII and Table IV read as if an evaluation was run ("All numerical results are reported from measured experimental trials", "was evaluated") but contain no numbers. | No live trials exist in `logs/attempts.csv` | **Fix in paper**: state plainly that Section VI is a protocol and Section VII reports implementation behaviour, plus the measured LFW matcher benchmark (below). Remove "evaluated"/"results are reported from measured trials" wording unless numbers are added. |
| 4 | "Weights reduce dependence on a single recognition output" (Section IV-B, VII-A, Discussion) | LFW benchmark: fusion does not beat LBPH alone (AUC 0.81 vs 0.75-0.76 fused) | **Fix in paper**: present fusion as a design choice whose benefit was not demonstrated, and cite the measured result. |
| 5 | Section V describes the pipeline only | The code now also has: threaded camera pipeline, enrolment quality gate and outlier pruning, 45 s liveness timeout, per-step logging, external-drive-only guard | **Update paper** if these are part of the submitted system (they change what is "implemented"). |
| 6 | Drive identification described as detecting removable drives | Code now refuses to run on internal drives (USB/SD/removable only) | Add one sentence. |

## Things that already match
LBPH radius 1 / 8 neighbours / 8x8 grid; 100x100 ROI; threshold 40; 2 blinks; head-turn 10 % of face width; motion threshold 0.03 over
20 observations; 100 enrolment frames; verification every 2 frames; exactly-one-face rule; PBKDF2-HMAC-SHA256 with 600,000 iterations and Fernet;
5 failed attempts then application-level wipe; 5-minute auto-lock; audit log.

## Wording you can use for the evaluation section (all numbers are measured, see RESULTS_LFW.md)
> To characterise the identity-verification stage independently of the acquisition hardware, we ran the matcher on the public LFW
> dataset (62 identities with at least 20 images; identities split into development and test halves; K = 10 enrolment images per
> identity; weights and threshold selected on the development half only). On the test half, LBPH alone achieved an AUC of 0.81 and an
> equal-error rate of about 25 %, whereas the weighted fusion of LBPH, histogram and template scores did not improve on it (EER about
> 30 %). At the configured threshold of 40 the fused score yielded FRR of 26-52 % and FAR of 9-38 % depending on the weights. These
> unconstrained-photo results are a conservative benchmark and do not measure liveness, presentation-attack resistance or
> in-device performance, which remain future work.

## Disclosure
The publisher's first concern is AI-generated content. If an AI assistant wrote or substantially edited the manuscript, code or
documentation, state that in the acknowledgements/disclosure section as IEEE requires, and make sure every number in the paper can be
traced to a file in this repository.
