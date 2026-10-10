# Project history

The original `.py` was lost; the first version of `main.py` was rebuilt from a compiled build. The raw decompiler
output used for that is not kept in this repository. Since then the camera pipeline, enrolment, liveness logging,
metrics and drive-safety checks have been rewritten and are covered by tests (see README).

## How the first version was rebuilt

`BioLock.exe` was a PyInstaller-frozen build (Python 3.12) and the
original `.py` source wasn't available, so `main.py` in this repo was
rebuilt from the compiled app itself:

1. `pyinstxtractor-ng` extracted the PyInstaller archive from the `.exe`,
   recovering the compiled `main.pyc` bytecode .
2. [`pycdc`](https://github.com/zrax/pycdc) (built from source, since it's
   one of the few decompilers with any Python 3.12 opcode support) was
   run against that bytecode. 
3. `pycdas` produced a full bytecode disassembly for cross-reference:
   a disassembly (not kept).

## Why `main.py` isn't a byte-for-byte recovery

Python 3.12's bytecode (the adaptive/specializing interpreter, opcodes
like `CALL_INTRINSIC_1`, `LOAD_FAST_CHECK`, `POP_JUMP_IF_NOT_NONE`, the
newer exception-table format, etc.) isn't fully handled by public
decompilers yet. `pycdc`'s raw output recovered roughly 55-60% of the
app cleanly — all imports, constants, the theme/colour palette, simple
I/O helpers, state/config persistence, and the wipe logic came back
correct. The rest — notably the live camera loop, the animated overlay
renderer, the Tkinter layout, and the camera-picker dialog — came back
empty (`WARNING: Decompyle incomplete`) or corrupted, and were
**rewritten by hand**, based on the recovered function signatures, how
each return value is consumed elsewhere in the file, and standard
implementations of the same technique (Haar-cascade detection, LBPH
recognition, eye-cascade blink detection, frame-diff liveness).

| Area | Status |
|---|---|
| Imports, constants, theme colours, paths | Recovered cleanly |
| `_hide/_unhide/_log/_beep/*sound*` | Recovered cleanly |
| Drive helpers, drive-id, autorun | Recovered, minor cleanup |
| State/config load & save | Recovered, minor cleanup |
| Vault key derivation (Fernet/PBKDF2) | Recovered cleanly |
| Cascade lookup, lighting quality | Recovered cleanly |
| `BlinkDetector` / `HeadTurn` / `MotionLive` | **Reconstructed** |
| `FaceEngine` (`_load`, part of `train`) | **Reconstructed**; `predict()` additionally redesigned as a weighted LBPH+Histogram+Template fusion for the experiments (see `docs/EXPERIMENTS.md`) |
| `ParticleSystem` / `Anim.render` | **Reconstructed** |
| `_scan_cams` / `CamPicker` | **Reconstructed** |
| `BioLock.__init__` / `_build_ui` | **Reconstructed** (now also includes the Testing Mode panel used for experiment logging) |
| Camera loop (`_tick`, `_loop`, `_start`, `_open_cam`, `_enrol`, `_verify`) | **Reconstructed** |
| `Splash`, `show_about` | Recovered cleanly |

## If you want to double-check a specific function

Open a disassembly (not kept) and search for the
function name — you'll get the exact Python 3.12 bytecode for it,
which you can read instruction-by-instruction if you want to confirm
or correct a specific piece of logic in `main.py`.
