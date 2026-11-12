# Experiment scripts

Everything that produces a number lives in `experiments/` and is driven by one command:

```bash
pip install -r requirements.txt -r requirements-dev.txt
python evaluate.py
```

That runs the steps below in order and writes `docs/RESULTS.md` from their JSON output (`results/`). It stops
before the test halves if the shipped configuration differs from what the selection rules give on the development
halves. `python evaluate.py --quick` runs a small subset of every step to check that the machinery works.

| Script | What it does | Output |
|---|---|---|
| `experiments/benchmark.py --split dev` | Matcher on the development halves of LFW and Olivetti through the app's own code; applies the selection rules for calibration, weights and threshold | `results/benchmark_dev.json` |
| `experiments/simulate.py --calibrate` | Simulation: the presence and blink measures on animated development photographs; selection rules for the three liveness thresholds | `results/liveness_calibration_dev.json` |
| `experiments/benchmark.py --split test` | Matcher on the test halves with the shipped configuration | `results/benchmark_test.json` |
| `experiments/simulate.py` | Simulation: complete sessions on the test halves (genuine, impostor, motionless and trembling photograph, liveness by one face and match by another) | `results/simulation_test.json` |
| `experiments/perf.py` | Time per step and per frame, memory, start-up, on the machine it runs on | `results/perf.json` |
| `experiments/summarize_live_logs.py` | Summary of the live trials logged by the app (`logs/attempts.csv`) | printed tables |

Shared code: `experiments/datasets.py` (loading the two datasets, the dev/test split, turning a photograph into
a camera frame, the trial protocol) and `biolock/sim.py` (animated photographs).

## Rules the scripts follow

1. Parameters are chosen on the development halves only. The selection rules are written at the top of
   `benchmark.py` and `simulate.py`.
2. The test halves are used to report, not to choose.
3. All random choices are seeded, so a second run gives the same numbers on the same library versions (recorded
   in every JSON file).
4. A simulation is called a simulation wherever its numbers appear.

## Changing a parameter

Edit `biolock/config.py`, run `python evaluate.py`, and report the new `docs/RESULTS.md` as a whole. From source a
parameter can also be overridden for one run with an environment variable (`BIOLOCK_MIN_CONF=40 python main.py`) or
a `config.json` next to `main.py`; the packaged exe ignores both for everything that decides an unlock.

## Where the matcher design came from

`research/` holds the earlier parameter search (crop and alignment, normalisation, LBPH radius and grid, other
scores, fusion, gallery size), all on the development halves. `research/README.md` lists the scripts in order;
`research/exp8_engine.py` checks that `biolock/engine.py` reproduces that pipeline exactly.
