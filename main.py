"""BioLock SSD  v10

Face-recognition lock for external drives (USB flash drives, external SSD/HDD, SD cards)
with motion, blink and head-movement liveness cues.

Author: Pasupuleti Vedh Rishi

This file only starts the application; the code lives in the `biolock` package:

    biolock/ui.py          window
    biolock/controller.py  what an attempt's outcome does to the drive
    biolock/session.py     presence -> blink -> head -> face state machine
    biolock/engine.py      detection, alignment, matching
    biolock/liveness.py    motion / blink / head-movement detectors
    biolock/camera.py      camera access on worker threads
    biolock/vault.py       vault-key derivation and check
    biolock/storage.py     files kept on the drive
    biolock/winutil.py     Windows specifics

Usage:
    python main.py                      start the app
    python main.py --selftest [file]    check that this installation / build works (no camera,
                                        nothing written to the drive); writes a JSON report
"""

import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '--selftest':
        from biolock.selftest import run_selftest
        return run_selftest(argv[1] if len(argv) > 1 else None)
    from biolock.ui import run
    return run()


if __name__ == '__main__':
    sys.exit(main())
