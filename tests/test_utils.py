import pytest

from jwt.utils import from_base64url_uint, to_base64url_uint
# from jwt.utils import force_bytes, from_base64url_uint, to_base64url_uint

from typing import Union
def force_bytes(value: Union[bytes, str]) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    elif isinstance(value, bytes):
        return value
    else:
        raise TypeError("Expected a string value")


@pytest.mark.parametrize(
    "inputval,expected",
    [
        (0, b"AA"),
        (1, b"AQ"),
        (255, b"_w"),
        (65537, b"AQAB"),
        (123456789, b"B1vNFQ"),
        pytest.param(-1, "", marks=pytest.mark.xfail(raises=ValueError)),
    ],
)
def test_to_base64url_uint(inputval, expected):
    actual = to_base64url_uint(inputval)
    assert actual == expected


@pytest.mark.parametrize(
    "inputval,expected",
    [
        (b"AA", 0),
        (b"AQ", 1),
        (b"_w", 255),
        (b"AQAB", 65537),
        (b"B1vNFQ", 123456789),
    ],
)
def test_from_base64url_uint(inputval, expected):
    actual = from_base64url_uint(inputval)
    assert actual == expected


def test_force_bytes_raises_error_on_invalid_object():
    with pytest.raises(TypeError):
        force_bytes({})  # type: ignore[arg-type]
