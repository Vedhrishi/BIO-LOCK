# Live trials

The scripts in `experiments/` measure the matcher on public photographs and run simulated sessions. They cannot
measure how the system behaves with real people, a real webcam and real attack material. That needs the trials
described here. Until they are done, no liveness or presentation-attack rate can be claimed.

## 1. What the app shows and logs

**Match score.** For every frame in the last step the app computes the chi-square distance between the face's
LBP histogram and the nearest enrolled sample, and maps it to 0-100 (distance 76.4 -> 0, distance 31.2 -> 100).
The decision uses the **median of 15 frames**; it must reach **35**. The median score of the frames seen during
the blink step must reach 35 as well, so that the liveness steps and the match belong to the same face. The
score is a similarity, not a probability.

**Log.** Each attempt adds one row to `logs/attempts.csv` next to the app:

| Column | Meaning |
|---|---|
| `trial_label`, `lighting_condition`, `movement_condition` | what you set in Testing mode |
| `outcome` | `accept`, `reject`, `timeout`, `cancelled` |
| `failed_stage` | `presence`, `blink`, `head`, `match`, `key`, or empty when accepted |
| `score`, `live_score` | median match score of the 15 match frames / of the blink-step frames |
| `lbph`, `hist`, `template`, `lbph_distance` | medians of the three component scores and of the raw distance |
| `presence_pass`, `blink_pass`, `head_pass`, `liveness_pass` | which liveness steps were completed |
| `motion_avg`, `blinks` | the presence measure and the number of blinks counted |
| `blink_frames`, `eyes_seen_frames` | frames of the blink step, and in how many the eye check saw open eyes |
| `dropouts_moved`, `dropouts_not_local`, `dropouts_length` | times the eyes disappeared but no blink was counted: the head moved, the picture changed everywhere (or not at all), the gap was too short or too long |
| `faces_0`, `faces_1`, `faces_multi` | detection passes with no / one / several faces |
| `latency_ms`, `proc_ms`, `fps`, `cpu_percent`, `mem_mb` | time to decision, processing per frame, cost |
| `counted` | 1 if the attempt counted towards the lock-out |

Nothing is logged per frame, and no image is stored.

## 2. Before you start

1. Copy the app to an external drive and run it there. Pick the camera that shows your face in colour (a laptop's
   infrared camera is listed as "greyscale / infrared": do not use it).
2. Enrol in the light you normally work in. Follow the prompts until 100 samples are captured.
3. Unlock once. Tick **Testing mode**, then **Do not count failed matches**. That switch can only be turned on
   while the vault is unlocked; without it five rejected trials erase the enrolment. Lock the vault again.
4. Write down the camera, the room, the lighting and who took part.

## 3. First check: do the liveness steps work for the enrolled person?

Three thresholds were set from simulated sessions only (`motion_thresh`, `blink_local_ratio`,
`blink_min_change`, see `docs/RESULTS.md` section 3). Check them before collecting numbers:

- **Presence.** Watch "Motion" under *Live signals*. Sitting still it should stay below the threshold shown next
  to it; moving your head gently from side to side should take it above within a second or two.
- **Blink.** Hold still and blink slowly. Each counted blink gives a short beep and raises the counter.
- **Head.** Move left until the prompt changes, then right.

Do ten genuine attempts. If a step regularly fails for a live person, note what you saw and report it rather
than changing a threshold by feel. The log helps: `motion_avg` for presence; for blinks, `eyes_seen_frames`
close to `blink_frames` with no drop-outs means the eye check never lost the eyes, a low `eyes_seen_frames` means
it does not see them reliably, and the three `dropouts_*` columns say why eyes that disappeared and came back
were not counted as a blink.

Thresholds are changed in `biolock/config.py`, from source, and every change has to be stated together with the
trials that justify it.

## 4. Trials

Set **Trial label**, lighting and movement before each attempt, then press *Unlock with face ID*.

| Trial label | What is presented | Suggested number |
|---|---|---|
| `genuine` | the enrolled person, live | 30 or more, on more than one day |
| `impostor` | other people, live (at least 3 people) | 30 or more |
| `attack_photo` | a printed photograph of the enrolled person: held still, moved by hand, with the eyes covered and uncovered | 30 or more |
| `attack_phone` | the photograph on a phone or tablet screen, same variations | 30 or more |
| `attack_replay` | a video of the enrolled person blinking and moving, played on a screen | 30 or more |

Notes:
- After an accepted genuine trial the vault is open: lock it before the next trial.
- A trial that ends in a time-out is a rejection. It is logged with the step it stopped at.
- The replay attack matters most. The blink, motion and head cues do not tell a replayed video from a person;
  expect it to pass the liveness steps if the video shows them, and report what you measure.
- Repeat the genuine and impostor trials in a second lighting condition if you can.

## 5. Summary

```bash
python experiments/summarize_live_logs.py
```

It prints FAR and FRR with 95 % intervals, where each kind of trial stopped, the rates per lighting and movement
tag, what other thresholds would have given (from the logged scores), and the cost per attempt.

## 6. Reporting

- Report only what is in `logs/attempts.csv`, with the number of trials for every rate and the conditions.
- Give the interval, not just the rate. With 30 trials, 0 false accepts still allows a true rate of about 11 %.
- Say which app version and configuration produced the log (`app_version`, `min_conf` columns).
- "Failed face matches: x of 5" in the app is the lock-out counter, not the number of trials.
- Untick the testing switches when you have finished.
