"""Vault-key validation (stage K of the pipeline).

A 32-byte key is derived with PBKDF2-HMAC-SHA256 (16-byte random salt, 600,000
iterations) and used with Fernet to encrypt a fixed sentinel. Unlocking requires
that the sentinel decrypts with a key re-derived from

    SHA-256( enrolled-template file  |  drive identifier  |  volume serial number )

so the check fails if the enrolled template was replaced or edited, or if the
BioLock data was copied to a different drive.

What this is NOT: there is no user secret in the derivation, so it is tamper
evidence and drive binding, not encryption strength against someone who has the
drive and knows the scheme. The files in SecureVault are hidden, not encrypted.
"""

import os
import base64
import hashlib
import secrets

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from . import winutil

SENTINEL = b'BIOLOCK_VAULT_OK'
SALT_BYTES = 16
ITERATIONS = 600_000


def key_material(model_digest, drive_id, volume_serial):
    h = hashlib.sha256()
    h.update(b'BioLock-vault-v10|')
    h.update(model_digest)
    h.update(b'|' + str(drive_id or '').encode('utf-8'))
    h.update(b'|' + str(volume_serial if volume_serial is not None else 'none').encode('ascii'))
    return h.digest()


def _fernet(material, salt, iterations=ITERATIONS):
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations)
    return Fernet(base64.urlsafe_b64encode(kdf.derive(material)))


def save_key(path, material):
    salt = secrets.token_bytes(SALT_BYTES)
    token = _fernet(material, salt).encrypt(SENTINEL)
    winutil.atomic_write(path, salt + token)


def verify_key(path, material):
    """True only if the stored sentinel decrypts with the key derived from `material`."""
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError:
        return False
    if len(data) <= SALT_BYTES:
        return False
    try:
        return _fernet(material, data[:SALT_BYTES]).decrypt(data[SALT_BYTES:]) == SENTINEL
    except (InvalidToken, ValueError):
        return False
