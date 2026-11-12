"""Thin Windows layer: file attributes, DPI awareness, drive identification, sounds, sign-in start.

Everything here degrades to a harmless no-op on other platforms so the engine,
the tests and the benchmarks can run anywhere.
"""

import os
import sys
import ctypes
import threading

IS_WIN = os.name == 'nt'

_HIDDEN = 0x2
_SYSTEM = 0x4
_INVALID = 0xFFFFFFFF

if IS_WIN:
    from ctypes import wintypes
    _k32 = ctypes.windll.kernel32
    _k32.GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
    _k32.GetFileAttributesW.restype = wintypes.DWORD
    _k32.SetFileAttributesW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
    _k32.SetFileAttributesW.restype = wintypes.BOOL
    _k32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    _k32.GetDriveTypeW.restype = wintypes.UINT
    _k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                 wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _k32.CreateFileW.restype = wintypes.HANDLE
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                     wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                     wintypes.LPVOID]
    _k32.DeviceIoControl.restype = wintypes.BOOL
    _k32.GetVolumeInformationW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
                                           ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                                           ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR, wintypes.DWORD]
    _k32.GetVolumeInformationW.restype = wintypes.BOOL
    _INVALID_HANDLE = wintypes.HANDLE(-1).value


# --------------------------------------------------------------------------
# File attributes (replaces spawning `attrib.exe` for every file operation)
# --------------------------------------------------------------------------
def set_hidden(path, hidden=True):
    """Set or clear the hidden+system attributes. Returns True when the file ends in the requested state."""
    if not IS_WIN or not path or not os.path.exists(path):
        return False
    attrs = _k32.GetFileAttributesW(path)
    if attrs == _INVALID:
        return False
    new = (attrs | _HIDDEN | _SYSTEM) if hidden else (attrs & ~(_HIDDEN | _SYSTEM))
    if new == attrs:
        return True
    return bool(_k32.SetFileAttributesW(path, new))


def is_hidden(path):
    if not IS_WIN or not os.path.exists(path):
        return False
    attrs = _k32.GetFileAttributesW(path)
    return attrs != _INVALID and bool(attrs & _HIDDEN)


def atomic_write(path, data, hidden=True):
    """Write bytes so that an unplugged drive can never leave a half-written file behind:
    write to a sibling temp file, flush to the device, then swap it into place."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    set_hidden(tmp, False)
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    set_hidden(path, False)          # a hidden+system target cannot be replaced on Windows
    os.replace(tmp, path)
    if hidden:
        set_hidden(path, True)


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------
def enable_dpi_awareness():
    """Ask Windows not to bitmap-stretch the window on high-DPI screens (must run before Tk starts)."""
    if not IS_WIN:
        return False
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)      # system-DPI aware (Windows 8.1+)
        return True
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()        # Windows Vista/7/8
            return True
        except Exception:
            return False


# --------------------------------------------------------------------------
# Drives
# --------------------------------------------------------------------------
# Bus types reported by IOCTL_STORAGE_QUERY_PROPERTY that mean "plugged in from outside"
EXTERNAL_BUSES = {7: 'USB', 4: '1394', 0xC: 'SD', 0xD: 'MMC'}

DRIVE_REMOVABLE = 2
DRIVE_FIXED = 3


def drive_letter(path):
    """Upper-case drive letter of a path, or '?' when it has none (UNC paths, other platforms)."""
    try:
        drive = os.path.splitdrive(os.path.abspath(path))[0]
        letter = drive.rstrip(':\\').upper()
        return letter if len(letter) == 1 and letter.isalpha() else '?'
    except Exception:
        return '?'


def drive_type(letter):
    if not IS_WIN:
        return None
    try:
        return int(_k32.GetDriveTypeW(f'{letter}:\\'))
    except Exception:
        return None


def bus_type(letter):
    """Storage bus of a drive letter (7 = USB, 11 = SATA, 17 = NVMe ...), or None if it cannot be read.
    Opens the volume with no access rights, so it needs no administrator privileges."""
    if not IS_WIN:
        return None
    try:
        h = _k32.CreateFileW(f'\\\\.\\{letter}:', 0, 3, None, 3, 0, None)
        if h is None or h == _INVALID_HANDLE:
            return None
        try:
            query = (ctypes.c_ubyte * 12)()                 # StorageDeviceProperty / PropertyStandardQuery
            out = (ctypes.c_ubyte * 1024)()
            got = wintypes.DWORD(0)
            ok = _k32.DeviceIoControl(h, 0x2D1400, query, 12, out, 1024, ctypes.byref(got), None)
            if not ok or got.value < 32:
                return None
            return int(out[28])                              # STORAGE_DEVICE_DESCRIPTOR.BusType
        finally:
            _k32.CloseHandle(h)
    except Exception:
        return None


def is_external_drive(letter):
    """True only when the drive is positively identified as external/removable.

    Fails CLOSED: anything unknown or internal (SATA / NVMe / the system disk) counts as internal."""
    dt = drive_type(letter)
    if dt == DRIVE_REMOVABLE:                    # USB sticks, SD cards
        return True
    if dt != DRIVE_FIXED:                        # network share, CD, RAM disk, unknown
        return False
    return bus_type(letter) in EXTERNAL_BUSES    # USB-attached SSD/HDD report as "fixed"


def volume_serial(letter):
    """Volume serial number (changes when the drive is formatted), or None."""
    if not IS_WIN:
        return None
    try:
        serial = wintypes.DWORD(0)
        ok = _k32.GetVolumeInformationW(f'{letter}:\\', None, 0, ctypes.byref(serial), None, None, None, 0)
        return int(serial.value) if ok else None
    except Exception:
        return None


def disk_usage_gb(letter):
    try:
        import shutil
        st = shutil.disk_usage(f'{letter}:\\')
        return st.free / 1073741824, st.total / 1073741824
    except Exception:
        return 0.0, 0.0


# --------------------------------------------------------------------------
# Sounds (never block the caller)
# --------------------------------------------------------------------------
def beep_sequence(notes):
    """notes: iterable of (frequency_hz, duration_ms). Played on a daemon thread."""
    if not IS_WIN:
        return

    def _play():
        for freq, dur in notes:
            try:
                _k32.Beep(int(freq), int(dur))
            except Exception:
                return
    threading.Thread(target=_play, daemon=True).start()


# --------------------------------------------------------------------------
# Start with Windows (only ever changed by an explicit user action in the UI)
# --------------------------------------------------------------------------
_RUN_KEY = 'Software\\Microsoft\\Windows\\CurrentVersion\\Run'
_RUN_NAME = 'BioLockUSB'


def autostart_target():
    """Command currently registered to start at sign-in, or None."""
    if not IS_WIN:
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_QUERY_VALUE) as k:
            return winreg.QueryValueEx(k, _RUN_NAME)[0]
    except Exception:
        return None


def autostart_enabled_for(exe_path):
    cur = autostart_target()
    return bool(cur) and os.path.normcase(cur.strip('"')) == os.path.normcase(exe_path)


def set_autostart(exe_path, enabled):
    """Add or remove the per-user sign-in entry. Returns True on success."""
    if not IS_WIN:
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            if enabled:
                winreg.SetValueEx(k, _RUN_NAME, 0, winreg.REG_SZ, f'"{exe_path}"')
            else:
                try:
                    winreg.DeleteValue(k, _RUN_NAME)
                except FileNotFoundError:
                    pass
        return True
    except Exception:
        return False


def app_executable():
    """Path of the packaged exe, or None when running from source (never auto-start a .py file)."""
    return sys.executable if getattr(sys, 'frozen', False) else None
