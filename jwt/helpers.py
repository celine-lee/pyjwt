
import base64
from typing import Union

def ensure_bytes(val: Union[str, bytes]) -> bytes:
    """
    Ensures that the passed value is in bytes. If it’s a string, it is UTF-8 encoded.
    """
    if isinstance(val, str):
        return val.encode("utf-8")
    return val

def add_base64_padding(data: bytes) -> bytes:
    """
    Adds missing '=' padding to a base64-encoded bytestring.
    """
    rem = len(data) % 4
    if rem:
        data += b"=" * (4 - rem)
    return data

def b64_encode_nopad(data: bytes) -> bytes:
    """
    Returns a URL-safe base64 encoding with any '=' padding removed.
    """
    return base64.urlsafe_b64encode(data).replace(b"=", b"")

def b64_decode_nopad(data: Union[bytes, str]) -> bytes:
    """
    Decodes URL-safe base64 data, automatically adding missing '=' padding.
    """
    data = ensure_bytes(data)
    data = add_base64_padding(data)
    return base64.urlsafe_b64decode(data)



import base64
from typing import Union

def ensure_bytes(val: Union[str, bytes]) -> bytes:
    """
    Ensures that the passed value is in bytes. If it’s a string, it is UTF-8 encoded.
    """
    if isinstance(val, str):
        return val.encode("utf-8")
    return val

def add_base64_padding(data: bytes) -> bytes:
    """
    Adds missing '=' padding to a base64-encoded bytestring.
    """
    rem = len(data) % 4
    if rem:
        data += b"=" * (4 - rem)
    return data

def b64_encode_nopad(data: bytes) -> bytes:
    """
    Returns a URL-safe base64 encoding with any '=' padding removed.
    """
    return base64.urlsafe_b64encode(data).replace(b"=", b"")

def b64_decode_nopad(data: Union[bytes, str]) -> bytes:
    """
    Decodes URL-safe base64 data, automatically adding missing '=' padding.
    """
    data = ensure_bytes(data)
    data = add_base64_padding(data)
    return base64.urlsafe_b64decode(data)
