"""Files on the drive, the vault key, the Windows layer and the external-drive rule."""
import os
import unittest
from unittest import mock

from tests import helpers as H
from biolock import vault, winutil
from biolock.storage import Store, DEF_STATE


def read(path):
    with open(path, 'rb') as f:
        return f.read()


class AtomicWriteTests(H.TempDirCase, unittest.TestCase):
    def test_writes_hides_and_replaces(self):
        p = os.path.join(self.tmp, 'sub', 'state.json')
        winutil.atomic_write(p, b'one')
        self.assertEqual(read(p), b'one')
        self.assertTrue(winutil.is_hidden(p))
        winutil.atomic_write(p, b'two')                       # a hidden+system file can still be replaced
        self.assertEqual(read(p), b'two')
        self.assertEqual(os.listdir(os.path.dirname(p)), ['state.json'])   # no temp file left behind
        winutil.atomic_write(p, b'three', hidden=False)
        self.assertFalse(winutil.is_hidden(p))

    def test_stale_hidden_temp_file_does_not_block_a_write(self):
        p = os.path.join(self.tmp, 'state.json')
        with open(p + '.tmp', 'wb') as f:
            f.write(b'left over from an unplugged drive')
        winutil.set_hidden(p + '.tmp', True)
        winutil.atomic_write(p, b'new')
        self.assertEqual(read(p), b'new')

    def test_hidden_attribute_round_trip(self):
        p = os.path.join(self.tmp, 'f.txt')
        open(p, 'w').close()
        self.assertFalse(winutil.is_hidden(p))
        self.assertTrue(winutil.set_hidden(p, True))
        self.assertTrue(winutil.is_hidden(p))
        self.assertTrue(winutil.set_hidden(p, False))
        self.assertFalse(winutil.is_hidden(p))
        self.assertFalse(winutil.set_hidden(os.path.join(self.tmp, 'missing'), True))


class StoreTests(H.TempDirCase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.s = Store(self.tmp)

    def test_state_round_trip_and_defaults(self):
        self.assertEqual(self.s.load_state(), DEF_STATE)
        self.s.save_state(dict(DEF_STATE, attempts=3))
        self.assertEqual(Store(self.tmp).load_state()['attempts'], 3)
        self.assertTrue(winutil.is_hidden(self.s.lock_dir))
        winutil.atomic_write(self.s.state_file, b'{ not json')
        self.assertEqual(self.s.load_state(), DEF_STATE)      # a damaged file falls back to the defaults
        winutil.atomic_write(self.s.state_file, b'[1, 2]')
        self.assertEqual(self.s.load_state(), DEF_STATE)

    def test_drive_id_is_created_once(self):
        self.assertIsNone(self.s.drive_id(create=False))
        a = self.s.drive_id()
        self.assertEqual(len(a), 32)
        self.assertEqual(self.s.drive_id(), a)
        self.assertEqual(Store(self.tmp).drive_id(create=False), a)

    def test_audit_log_appends(self):
        self.assertEqual(self.s.read_log(), '')
        self.s.log('ONE', 'detail')
        self.s.log('TWO', level='WARN')
        lines = self.s.read_log().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn('[INFO] ONE  |  detail', lines[0])
        self.assertIn('[WARN] TWO', lines[1])
        self.assertTrue(winutil.is_hidden(self.s.log_file))

    def test_vault_show_hide_and_size(self):
        self.s.vault_hide()                                   # nothing to hide yet: no error
        self.s.vault_show()
        self.assertTrue(os.path.isdir(self.s.vault_dir))
        with open(os.path.join(self.s.vault_dir, 'a.bin'), 'wb') as f:
            f.write(b'x' * 2097152)
        self.assertAlmostEqual(self.s.vault_size_mb(), 2.0, places=2)
        self.s.vault_hide()
        self.assertTrue(winutil.is_hidden(self.s.vault_dir))
        self.s.vault_show()
        self.assertFalse(winutil.is_hidden(self.s.vault_dir))

    def test_wipe_removes_everything_including_old_version_files(self):
        s = self.s
        s.save_state(dict(DEF_STATE, enrolled=True))
        s.drive_id()
        s.log('X')
        s.save_local({'cam_index': 1})
        for name in ('face_model.npz', 'vault.key', 'face.xml', 'face_stats.pkl', 'half.tmp'):
            winutil.atomic_write(os.path.join(s.lock_dir, name), b'data')
        s.vault_show()
        os.makedirs(os.path.join(s.vault_dir, 'deep', 'er'))
        with open(os.path.join(s.vault_dir, 'deep', 'er', 'f.txt'), 'w') as f:
            f.write('x')
        s.vault_hide()
        self.assertTrue(s.has_model())
        s.wipe()
        self.assertEqual(os.listdir(s.lock_dir), [])
        self.assertFalse(os.path.exists(s.vault_dir))
        self.assertFalse(s.has_model() or s.has_legacy_model())

    def test_legacy_detection(self):
        self.assertFalse(self.s.has_legacy_model())
        winutil.atomic_write(os.path.join(self.s.lock_dir, 'face.xml'), b'<xml/>')
        self.assertTrue(self.s.has_legacy_model())
        winutil.atomic_write(self.s.model_file, b'new')
        self.assertFalse(self.s.has_legacy_model())           # a current enrolment takes precedence

    def test_writable(self):
        self.assertTrue(self.s.writable())
        with mock.patch('builtins.open', side_effect=PermissionError):
            self.assertFalse(self.s.writable())


class VaultKeyTests(H.TempDirCase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.path = os.path.join(self.tmp, 'vault.key')
        self.material = vault.key_material(b'\x01' * 32, 'drive', 42)
        with mock.patch.object(vault, 'ITERATIONS', 2000):    # keep the tests quick; the real count is checked below
            vault.save_key(self.path, self.material)

    def verify(self, material=None, path=None):
        with mock.patch.object(vault, 'ITERATIONS', 2000):
            return vault.verify_key(path or self.path, material or self.material)

    def test_right_material_verifies_wrong_does_not(self):
        self.assertTrue(self.verify())
        self.assertFalse(self.verify(vault.key_material(b'\x02' * 32, 'drive', 42)))    # other template
        self.assertFalse(self.verify(vault.key_material(b'\x01' * 32, 'other', 42)))    # other drive id
        self.assertFalse(self.verify(vault.key_material(b'\x01' * 32, 'drive', 43)))    # other volume
        self.assertFalse(self.verify(vault.key_material(b'\x01' * 32, 'drive', None)))

    def test_damaged_or_missing_key_file_fails_closed(self):
        self.assertFalse(self.verify(path=os.path.join(self.tmp, 'missing.key')))
        data = read(self.path)
        for bad in (b'', data[:10], data[:16], data[:-5], data[:20] + b'x' + data[21:], b'\x00' * 200):
            winutil.atomic_write(self.path, bad)
            self.assertFalse(self.verify())

    def test_salt_is_random_and_file_is_hidden(self):
        first = read(self.path)
        with mock.patch.object(vault, 'ITERATIONS', 2000):
            vault.save_key(self.path, self.material)
        self.assertNotEqual(first[:16], read(self.path)[:16])
        self.assertTrue(winutil.is_hidden(self.path))

    def test_parameters_are_the_documented_ones(self):
        self.assertEqual((vault.ITERATIONS, vault.SALT_BYTES), (600_000, 16))
        self.assertEqual(len(vault.key_material(b'x' * 32, 'd', 1)), 32)


class DriveRuleTests(unittest.TestCase):
    """BioLock may only run from external drives. Anything unknown counts as internal (fails closed)."""

    def check(self, drive_type, bus):
        with mock.patch.object(winutil, 'drive_type', return_value=drive_type), \
                mock.patch.object(winutil, 'bus_type', return_value=bus):
            return winutil.is_external_drive('Z')

    def test_usb_flash_drive_reported_removable_is_allowed(self):
        self.assertTrue(self.check(2, None))

    def test_usb_ssd_reported_fixed_is_allowed(self):
        self.assertTrue(self.check(3, 7))

    def test_sd_and_mmc_cards_are_allowed(self):
        self.assertTrue(self.check(3, 0xC))
        self.assertTrue(self.check(3, 0xD))

    def test_internal_nvme_and_sata_are_blocked(self):
        self.assertFalse(self.check(3, 17))
        self.assertFalse(self.check(3, 11))

    def test_unknown_bus_fails_closed(self):
        self.assertFalse(self.check(3, None))
        self.assertFalse(self.check(None, 7))

    def test_network_and_optical_drives_are_blocked(self):
        self.assertFalse(self.check(4, 7))
        self.assertFalse(self.check(5, 7))
        self.assertFalse(self.check(1, None))

    @unittest.skipUnless(winutil.IS_WIN, 'Windows only')
    def test_the_real_system_drive_is_internal(self):
        letter = os.environ.get('SystemDrive', 'C:')[0]
        self.assertFalse(winutil.is_external_drive(letter))
        self.assertEqual(winutil.drive_type(letter), winutil.DRIVE_FIXED)
        self.assertIsNotNone(winutil.bus_type(letter))
        self.assertIsNotNone(winutil.volume_serial(letter))

    def test_drive_letter(self):
        self.assertEqual(winutil.drive_letter('e:\\BioLock\\BioLock.exe'), 'E')
        self.assertEqual(winutil.drive_letter('\\\\server\\share\\x'), '?')

    def test_start_up_guard(self):
        from biolock import ui
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('BIOLOCK_ENFORCE_EXTERNAL', None)
            self.assertTrue(ui.enforce_external_drive('C:\\anything', show=False))        # off from source
        with mock.patch.dict(os.environ, {'BIOLOCK_ENFORCE_EXTERNAL': '1'}):
            with mock.patch.object(winutil, 'is_external_drive', return_value=False):
                self.assertFalse(ui.enforce_external_drive('C:\\anything', show=False))
            with mock.patch.object(winutil, 'is_external_drive', return_value=True):
                self.assertTrue(ui.enforce_external_drive('E:\\anything', show=False))
        with mock.patch.object(ui.sys, 'frozen', True, create=True), \
                mock.patch.dict(os.environ, {'BIOLOCK_ALLOW_INTERNAL': '1'}), \
                mock.patch.object(winutil, 'is_external_drive', return_value=False):
            self.assertFalse(ui.enforce_external_drive('C:\\anything', show=False))       # no escape hatch in the exe


@unittest.skipUnless(winutil.IS_WIN, 'Windows only')
class AutostartTests(unittest.TestCase):
    """The sign-in entry is only ever written by an explicit user action. The registry is mocked here."""

    def test_set_query_and_remove_without_touching_the_real_registry(self):
        import winreg
        store = {}

        class Key:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def query(_k, name):
            if name not in store:
                raise FileNotFoundError(name)
            return store[name], winreg.REG_SZ

        def delete(_k, name):
            if name not in store:
                raise FileNotFoundError(name)
            del store[name]

        with mock.patch.object(winreg, 'OpenKey', return_value=Key()) as opened, \
                mock.patch.object(winreg, 'SetValueEx', side_effect=lambda k, n, r, t, v: store.__setitem__(n, v)), \
                mock.patch.object(winreg, 'QueryValueEx', side_effect=query), \
                mock.patch.object(winreg, 'DeleteValue', side_effect=delete):
            exe = 'E:\\BioLock.exe'
            self.assertFalse(winutil.autostart_enabled_for(exe))
            self.assertTrue(winutil.set_autostart(exe, True))
            self.assertEqual(store, {'BioLockUSB': f'"{exe}"'})
            self.assertTrue(winutil.autostart_enabled_for(exe))
            self.assertFalse(winutil.autostart_enabled_for('F:\\BioLock.exe'))
            self.assertTrue(winutil.set_autostart(exe, False))
            self.assertEqual(store, {})
            self.assertTrue(winutil.set_autostart(exe, False))            # removing twice is fine
            self.assertTrue(all(c.args[0] == winreg.HKEY_CURRENT_USER for c in opened.call_args_list))

    def test_only_a_packaged_exe_can_be_registered(self):
        self.assertIsNone(winutil.app_executable())                        # running from source

    def test_nothing_registers_itself_at_start_up(self):
        for name in ('ui.py', 'controller.py', 'storage.py'):
            with open(os.path.join(H.ROOT, 'biolock', name), encoding='utf-8') as f:
                calls = [ln for ln in f.read().splitlines() if 'set_autostart(' in ln]
            if name == 'ui.py':
                self.assertEqual(len(calls), 1)                            # the checkbox handler, nowhere else
                self.assertIn('self.v_auto.get()', calls[0])
            else:
                self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
