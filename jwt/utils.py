
import base64
import binascii
import re
from typing import Union

try:
    from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurve
    from cryptography.hazmat.primitives.asymmetric.utils import (
        decode_dss_signature,
        encode_dss_signature,
    )
except ModuleNotFoundError:
    pass


# --- Base64 URL Helpers ---
def _strip_padding(b: bytes) -> bytes:
    return b.replace(b"=", b"")


def _add_base64_padding(val: bytes) -> bytes:
    rem = len(val) % 4
    if rem:
        val += b"=" * (4 - rem)
    return val


def base64url_encode(data: bytes) -> bytes:
    """Encodes data using URL‐safe base64 without padding."""
    return _strip_padding(base64.urlsafe_b64encode(data))


def base64url_decode(val: Union[bytes, str]) -> bytes:
    """Decodes URL‐safe base64, adding any missing padding."""
    if isinstance(val, str):
        val = val.encode("utf-8")
    return base64.urlsafe_b64decode(_add_base64_padding(val))


# --- Integer/Byte Conversion Helpers ---
def bytes_from_int(val: int) -> bytes:
    remaining = val
    byte_length = 0
    while remaining:
        remaining >>= 8
        byte_length += 1
    # Ensure at least one byte is returned for val==0.
    return val.to_bytes(byte_length if byte_length else 1, "big", signed=False)


def to_base64url_uint(val: int) -> bytes:
    if val < 0:
        raise ValueError("Must be a positive integer")
    int_bytes = bytes_from_int(val)
    return base64url_encode(int_bytes)


def from_base64url_uint(val: Union[bytes, str]) -> int:
    data = base64url_decode(val)
    return int.from_bytes(data, byteorder="big")


def number_to_bytes(num: int, num_bytes: int) -> bytes:
    padded_hex = f"{num:0{2 * num_bytes}x}"
    return binascii.a2b_hex(padded_hex.encode("ascii"))


def bytes_to_number(data: bytes) -> int:
    return int(binascii.b2a_hex(data), 16)


# --- Signature Conversion Helpers ---
def der_to_raw_signature(der_sig: bytes, curve: "EllipticCurve") -> bytes:
    num_bits = curve.key_size
    num_bytes = (num_bits + 7) // 8
    r, s = decode_dss_signature(der_sig)
    return number_to_bytes(r, num_bytes) + number_to_bytes(s, num_bytes)


def raw_to_der_signature(raw_sig: bytes, curve: "EllipticCurve") -> bytes:
    num_bits = curve.key_size
    num_bytes = (num_bits + 7) // 8
    if len(raw_sig) != 2 * num_bytes:
        raise ValueError("Invalid signature")
    r = bytes_to_number(raw_sig[:num_bytes])
    s = bytes_to_number(raw_sig[num_bytes:])
    return bytes(encode_dss_signature(r, s))


# --- PEM and SSH Key Helpers ---
_PEMS = {
    b"CERTIFICATE",
    b"TRUSTED CERTIFICATE",
    b"PRIVATE KEY",
    b"PUBLIC KEY",
    b"ENCRYPTED PRIVATE KEY",
    b"OPENSSH PRIVATE KEY",
    b"DSA PRIVATE KEY",
    b"RSA PRIVATE KEY",
    b"RSA PUBLIC KEY",
    b"EC PRIVATE KEY",
    b"DH PARAMETERS",
    b"NEW CERTIFICATE REQUEST",
    b"CERTIFICATE REQUEST",
    b"SSH2 PUBLIC KEY",
    b"SSH2 ENCRYPTED PRIVATE KEY",
    b"X509 CRL",
}

_PEM_PATTERN = b"----[- ]BEGIN (" + b"|".join(_PEMS) + b""")[- ]----\r?.+?\r?----[- ]END \\1[- ]----\r?\n?"""
_PEM_RE = re.compile(_PEM_PATTERN, re.DOTALL)

def is_pem_format(key: bytes) -> bool:
    return bool(_PEM_RE.search(key))


_CERT_SUFFIX = b"-cert-v01@openssh.com"
_SSH_PUBKEY_RC = re.compile(rb"\A(\S+)[ \t]+(\S+)")
_SSH_KEY_FORMATS = [
    b"ssh-ed25519",
    b"ssh-rsa",
    b"ssh-dss",
    b"ecdsa-sha2-nistp256",
    b"ecdsa-sha2-nistp384",
    b"ecdsa-sha2-nistp521",
]

def is_ssh_key(key: bytes) -> bool:
    for fmt in _SSH_KEY_FORMATS:
        if fmt in key:
            return True
    ssh_pubkey_match = _SSH_PUBKEY_RC.match(key)
    if ssh_pubkey_match:
        key_type = ssh_pubkey_match.group(1)
        if key_type.endswith(_CERT_SUFFIX):
            return True
    return False
