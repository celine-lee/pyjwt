
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Union, cast, overload

from .exceptions import InvalidKeyError
from .types import HashlibHash, JWKDict
from .utils import (
    der_to_raw_signature,
    from_base64url_uint,
    is_pem_format,
    is_ssh_key,
    raw_to_der_signature,
    to_base64url_uint,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.asymmetric.ec import (
        ECDSA,
        SECP256K1,
        SECP256R1,
        SECP384R1,
        SECP521R1,
        EllipticCurve,
        EllipticCurvePrivateKey,
        EllipticCurvePrivateNumbers,
        EllipticCurvePublicKey,
        EllipticCurvePublicNumbers,
    )
    from cryptography.hazmat.primitives.asymmetric.ed448 import (
        Ed448PrivateKey,
        Ed448PublicKey,
    )
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.asymmetric.rsa import (
        RSAPrivateKey,
        RSAPrivateNumbers,
        RSAPublicKey,
        RSAPublicNumbers,
        rsa_crt_dmp1,
        rsa_crt_dmq1,
        rsa_crt_iqmp,
        rsa_recover_prime_factors,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
        load_pem_private_key,
        load_pem_public_key,
        load_ssh_public_key,
    )

    has_crypto = True
except ModuleNotFoundError:
    has_crypto = False

if TYPE_CHECKING:
    # Type aliases for convenience in algorithms method signatures
    AllowedRSAKeys = RSAPrivateKey | RSAPublicKey
    AllowedECKeys = EllipticCurvePrivateKey | EllipticCurvePublicKey
    AllowedOKPKeys = (
        Ed25519PrivateKey | Ed25519PublicKey | Ed448PrivateKey | Ed448PublicKey
    )
    AllowedKeys = AllowedRSAKeys | AllowedECKeys | AllowedOKPKeys
    AllowedPrivateKeys = (
        RSAPrivateKey | EllipticCurvePrivateKey | Ed25519PrivateKey | Ed448PrivateKey
    )
    AllowedPublicKeys = (
        RSAPublicKey | EllipticCurvePublicKey | Ed25519PublicKey | Ed448PublicKey
    )


requires_cryptography = {
    "RS256",
    "RS384",
    "RS512",
    "ES256",
    "ES256K",
    "ES384",
    "ES521",
    "ES512",
    "PS256",
    "PS384",
    "PS512",
    "EdDSA",
}


# -----------------------
# Helper functions
# -----------------------

def _ensure_bytes(val: str | bytes) -> bytes:
    """
    Convert a string to bytes if necessary.
    """
    if isinstance(val, bytes):
        return val
    elif isinstance(val, str):
        return val.encode("utf-8")
    else:
        raise TypeError("Expected a string or bytes value")


def get_default_algorithms() -> dict[str, Algorithm]:
    """
    Returns the algorithms that are implemented by the library.
    """
    default_algorithms = {
        "none": NoneAlgorithm(),
        "HS256": HMACAlgorithm(HMACAlgorithm.SHA256),
        "HS384": HMACAlgorithm(HMACAlgorithm.SHA384),
        "HS512": HMACAlgorithm(HMACAlgorithm.SHA512),
    }

    if has_crypto:
        default_algorithms.update(
            {
                "RS256": RSAAlgorithm(RSAAlgorithm.SHA256),
                "RS384": RSAAlgorithm(RSAAlgorithm.SHA384),
                "RS512": RSAAlgorithm(RSAAlgorithm.SHA512),
                "ES256": ECAlgorithm(ECAlgorithm.SHA256),
                "ES256K": ECAlgorithm(ECAlgorithm.SHA256),
                "ES384": ECAlgorithm(ECAlgorithm.SHA384),
                "ES521": ECAlgorithm(ECAlgorithm.SHA512),
                "ES512": ECAlgorithm(ECAlgorithm.SHA512),
                "PS256": RSAPSSAlgorithm(RSAPSSAlgorithm.SHA256),
                "PS384": RSAPSSAlgorithm(RSAPSSAlgorithm.SHA384),
                "PS512": RSAPSSAlgorithm(RSAPSSAlgorithm.SHA512),
                "EdDSA": OKPAlgorithm(),
            }
        )

    return default_algorithms


class Algorithm(ABC):
    """
    The interface for an algorithm used to sign and verify tokens.
    """

    def compute_hash_digest(self, bytestr: bytes) -> bytes:
        """
        Compute a hash digest using the specified algorithm's hash algorithm.
        If there is no hash algorithm, raises NotImplementedError.
        """
        hash_alg = getattr(self, "hash_alg", None)
        if hash_alg is None:
            raise NotImplementedError

        if (
            has_crypto
            and isinstance(hash_alg, type)
            and issubclass(hash_alg, hashes.HashAlgorithm)
        ):
            digest = hashes.Hash(hash_alg(), backend=default_backend())
            digest.update(bytestr)
            return bytes(digest.finalize())
        else:
            return bytes(hash_alg(bytestr).digest())

    @abstractmethod
    def prepare_key(self, key: Any) -> Any:
        """
        Performs necessary validation and conversions on the key and returns
        the key value in the proper format for sign() and verify().
        """

    @abstractmethod
    def sign(self, msg: bytes, key: Any) -> bytes:
        """
        Returns a digital signature for the specified message using the specified key.
        """

    @abstractmethod
    def verify(self, msg: bytes, key: Any, sig: bytes) -> bool:
        """
        Verifies that the specified digital signature is valid for the given message and key.
        """

    @overload
    @staticmethod
    @abstractmethod
    def to_jwk(key_obj, as_dict: Literal[True]) -> JWKDict:
        ...  # pragma: no cover

    @overload
    @staticmethod
    @abstractmethod
    def to_jwk(key_obj, as_dict: Literal[False] = False) -> str:
        ...  # pragma: no cover

    @staticmethod
    @abstractmethod
    def to_jwk(key_obj, as_dict: bool = False) -> Union[JWKDict, str]:
        """
        Serializes a given key into a JWK.
        """

    @staticmethod
    @abstractmethod
    def from_jwk(jwk: str | JWKDict) -> Any:
        """
        Deserializes a key from JWK back into a key object.
        """


class NoneAlgorithm(Algorithm):
    """
    Placeholder for use when no signing or verification operations are required.
    """

    def prepare_key(self, key: str | None) -> None:
        if key == "":
            key = None

        if key is not None:
            raise InvalidKeyError('When alg = "none", key value must be None.')

        return key

    def sign(self, msg: bytes, key: None) -> bytes:
        return b""

    def verify(self, msg: bytes, key: None, sig: bytes) -> bool:
        return False

    @staticmethod
    def to_jwk(key_obj: Any, as_dict: bool = False) -> NoReturn:
        raise NotImplementedError()

    @staticmethod
    def from_jwk(jwk: str | JWKDict) -> NoReturn:
        raise NotImplementedError()


class HMACAlgorithm(Algorithm):
    """
    Performs signing and verification using HMAC and a specified hash function.
    """

    SHA256: ClassVar[HashlibHash] = hashlib.sha256
    SHA384: ClassVar[HashlibHash] = hashlib.sha384
    SHA512: ClassVar[HashlibHash] = hashlib.sha512

    def __init__(self, hash_alg: HashlibHash) -> None:
        self.hash_alg = hash_alg

    def prepare_key(self, key: str | bytes) -> bytes:
        key_bytes = _ensure_bytes(key)
        if is_pem_format(key_bytes) or is_ssh_key(key_bytes):
            raise InvalidKeyError(
                "The specified key is an asymmetric key or x509 certificate and"
                " should not be used as an HMAC secret."
            )
        return key_bytes

    @overload
    @staticmethod
    def to_jwk(key_obj: str | bytes, as_dict: Literal[True]) -> JWKDict:
        ...  # pragma: no cover

    @overload
    @staticmethod
    def to_jwk(key_obj: str | bytes, as_dict: Literal[False] = False) -> str:
        ...  # pragma: no cover

    @staticmethod
    def to_jwk(key_obj: str | bytes, as_dict: bool = False) -> Union[JWKDict, str]:
        key_as_bytes = _ensure_bytes(key_obj)
        jwk = {
            "k": base64.urlsafe_b64encode(key_as_bytes).replace(b"=", b"").decode(),
            "kty": "oct",
        }
        if as_dict:
            return jwk
        else:
            return json.dumps(jwk)

    @staticmethod
    def from_jwk(jwk: str | JWKDict) -> bytes:
        try:
            if isinstance(jwk, str):
                obj: JWKDict = json.loads(jwk)
            elif isinstance(jwk, dict):
                obj = jwk
            else:
                raise ValueError
        except ValueError:
            raise InvalidKeyError("Key is not valid JSON")

        if obj.get("kty") != "oct":
            raise InvalidKeyError("Not an HMAC key")

        k_val = obj["k"]
        k_as_bytes = _ensure_bytes(k_val)
        rem = len(k_as_bytes) % 4
        if rem > 0:
            k_as_bytes += b"=" * (4 - rem)
        return base64.urlsafe_b64decode(k_as_bytes)

    def sign(self, msg: bytes, key: bytes) -> bytes:
        return hmac.new(key, msg, self.hash_alg).digest()

    def verify(self, msg: bytes, key: bytes, sig: bytes) -> bool:
        return hmac.compare_digest(sig, self.sign(msg, key))


if has_crypto:

    class RSAAlgorithm(Algorithm):
        """
        Performs signing and verification using RSASSA-PKCS-v1_5 with a given hash function.
        """

        SHA256: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA256
        SHA384: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA384
        SHA512: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA512

        def __init__(self, hash_alg: type[hashes.HashAlgorithm]) -> None:
            self.hash_alg = hash_alg

        def prepare_key(self, key: AllowedRSAKeys | str | bytes) -> AllowedRSAKeys:
            if isinstance(key, (RSAPrivateKey, RSAPublicKey)):
                return key

            if not isinstance(key, (bytes, str)):
                raise TypeError("Expecting a PEM-formatted key.")

            key_bytes = _ensure_bytes(key)
            try:
                if key_bytes.startswith(b"ssh-rsa"):
                    return cast(RSAPublicKey, load_ssh_public_key(key_bytes))
                else:
                    return cast(
                        RSAPrivateKey, load_pem_private_key(key_bytes, password=None)
                    )
            except ValueError:
                return cast(RSAPublicKey, load_pem_public_key(key_bytes))

        @overload
        @staticmethod
        def to_jwk(key_obj: AllowedRSAKeys, as_dict: Literal[True]) -> JWKDict:
            ...  # pragma: no cover

        @overload
        @staticmethod
        def to_jwk(key_obj: AllowedRSAKeys, as_dict: Literal[False] = False) -> str:
            ...  # pragma: no cover

        @staticmethod
        def to_jwk(key_obj: AllowedRSAKeys, as_dict: bool = False) -> Union[JWKDict, str]:
            obj: dict[str, Any] | None = None

            if hasattr(key_obj, "private_numbers"):
                numbers = key_obj.private_numbers()
                obj = {
                    "kty": "RSA",
                    "key_ops": ["sign"],
                    "n": to_base64url_uint(numbers.public_numbers.n).decode(),
                    "e": to_base64url_uint(numbers.public_numbers.e).decode(),
                    "d": to_base64url_uint(numbers.d).decode(),
                    "p": to_base64url_uint(numbers.p).decode(),
                    "q": to_base64url_uint(numbers.q).decode(),
                    "dp": to_base64url_uint(numbers.dmp1).decode(),
                    "dq": to_base64url_uint(numbers.dmq1).decode(),
                    "qi": to_base64url_uint(numbers.iqmp).decode(),
                }
            elif hasattr(key_obj, "verify"):
                numbers = key_obj.public_numbers()
                obj = {
                    "kty": "RSA",
                    "key_ops": ["verify"],
                    "n": to_base64url_uint(numbers.n).decode(),
                    "e": to_base64url_uint(numbers.e).decode(),
                }
            else:
                raise InvalidKeyError("Not a public or private key")

            if as_dict:
                return obj
            else:
                return json.dumps(obj)

        @staticmethod
        def from_jwk(jwk: str | JWKDict) -> AllowedRSAKeys:
            try:
                if isinstance(jwk, str):
                    obj = json.loads(jwk)
                elif isinstance(jwk, dict):
                    obj = jwk
                else:
                    raise ValueError
            except ValueError:
                raise InvalidKeyError("Key is not valid JSON")

            if obj.get("kty") != "RSA":
                raise InvalidKeyError("Not an RSA key")

            if "d" in obj and "e" in obj and "n" in obj:
                if "oth" in obj:
                    raise InvalidKeyError(
                        "Unsupported RSA private key: > 2 primes not supported"
                    )

                other_props = ["p", "q", "dp", "dq", "qi"]
                props_found = [prop in obj for prop in other_props]
                if any(props_found) and not all(props_found):
                    raise InvalidKeyError(
                        "RSA key must include all parameters if any are present besides d"
                    )

                public_numbers = RSAPublicNumbers(
                    from_base64url_uint(obj["e"]),
                    from_base64url_uint(obj["n"]),
                )
                if any(props_found):
                    numbers = RSAPrivateNumbers(
                        d=from_base64url_uint(obj["d"]),
                        p=from_base64url_uint(obj["p"]),
                        q=from_base64url_uint(obj["q"]),
                        dmp1=from_base64url_uint(obj["dp"]),
                        dmq1=from_base64url_uint(obj["dq"]),
                        iqmp=from_base64url_uint(obj["qi"]),
                        public_numbers=public_numbers,
                    )
                else:
                    d = from_base64url_uint(obj["d"])
                    p, q = rsa_recover_prime_factors(
                        public_numbers.n, d, public_numbers.e
                    )
                    numbers = RSAPrivateNumbers(
                        d=d,
                        p=p,
                        q=q,
                        dmp1=rsa_crt_dmp1(d, p),
                        dmq1=rsa_crt_dmq1(d, q),
                        iqmp=rsa_crt_iqmp(p, q),
                        public_numbers=public_numbers,
                    )
                return numbers.private_key()
            elif "n" in obj and "e" in obj:
                return RSAPublicNumbers(
                    from_base64url_uint(obj["e"]),
                    from_base64url_uint(obj["n"]),
                ).public_key()
            else:
                raise InvalidKeyError("Not a public or private key")

        def sign(self, msg: bytes, key: RSAPrivateKey) -> bytes:
            return key.sign(msg, padding.PKCS1v15(), self.hash_alg())

        def verify(self, msg: bytes, key: RSAPublicKey, sig: bytes) -> bool:
            try:
                key.verify(sig, msg, padding.PKCS1v15(), self.hash_alg())
                return True
            except InvalidSignature:
                return False


    class ECAlgorithm(Algorithm):
        """
        Performs signing and verification using ECDSA and a specified hash function.
        """

        SHA256: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA256
        SHA384: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA384
        SHA512: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA512

        def __init__(self, hash_alg: type[hashes.HashAlgorithm]) -> None:
            self.hash_alg = hash_alg

        def prepare_key(self, key: AllowedECKeys | str | bytes) -> AllowedECKeys:
            if isinstance(key, (EllipticCurvePrivateKey, EllipticCurvePublicKey)):
                return key

            if not isinstance(key, (bytes, str)):
                raise TypeError("Expecting a PEM-formatted key.")

            key_bytes = _ensure_bytes(key)
            try:
                if key_bytes.startswith(b"ecdsa-sha2-"):
                    crypto_key = load_ssh_public_key(key_bytes)
                else:
                    crypto_key = load_pem_public_key(key_bytes)
            except ValueError:
                crypto_key = load_pem_private_key(key_bytes, password=None)

            if not isinstance(
                crypto_key, (EllipticCurvePrivateKey, EllipticCurvePublicKey)
            ):
                raise InvalidKeyError(
                    "Expecting an EllipticCurve key. Wrong key provided for ECDSA algorithms."
                )

            return crypto_key

        def sign(self, msg: bytes, key: EllipticCurvePrivateKey) -> bytes:
            der_sig = key.sign(msg, ECDSA(self.hash_alg()))
            return der_to_raw_signature(der_sig, key.curve)

        def verify(self, msg: bytes, key: AllowedECKeys, sig: bytes) -> bool:
            try:
                der_sig = raw_to_der_signature(sig, key.curve)
            except ValueError:
                return False

            try:
                public_key = (
                    key.public_key()
                    if isinstance(key, EllipticCurvePrivateKey)
                    else key
                )
                public_key.verify(der_sig, msg, ECDSA(self.hash_alg()))
                return True
            except InvalidSignature:
                return False

        @overload
        @staticmethod
        def to_jwk(key_obj: AllowedECKeys, as_dict: Literal[True]) -> JWKDict:
            ...  # pragma: no cover

        @overload
        @staticmethod
        def to_jwk(key_obj: AllowedECKeys, as_dict: Literal[False] = False) -> str:
            ...  # pragma: no cover

        @staticmethod
        def to_jwk(key_obj: AllowedECKeys, as_dict: bool = False) -> Union[JWKDict, str]:
            if isinstance(key_obj, EllipticCurvePrivateKey):
                public_numbers = key_obj.public_key().public_numbers()
            elif isinstance(key_obj, EllipticCurvePublicKey):
                public_numbers = key_obj.public_numbers()
            else:
                raise InvalidKeyError("Not a public or private key")

            if isinstance(key_obj.curve, SECP256R1):
                crv = "P-256"
            elif isinstance(key_obj.curve, SECP384R1):
                crv = "P-384"
            elif isinstance(key_obj.curve, SECP521R1):
                crv = "P-521"
            elif isinstance(key_obj.curve, SECP256K1):
                crv = "secp256k1"
            else:
                raise InvalidKeyError(f"Invalid curve: {key_obj.curve}")

            obj: dict[str, Any] = {
                "kty": "EC",
                "crv": crv,
                "x": to_base64url_uint(public_numbers.x).decode(),
                "y": to_base64url_uint(public_numbers.y).decode(),
            }
            if isinstance(key_obj, EllipticCurvePrivateKey):
                obj["d"] = to_base64url_uint(key_obj.private_numbers().private_value).decode()

            if as_dict:
                return obj
            else:
                return json.dumps(obj)

        @staticmethod
        def from_jwk(jwk: str | JWKDict) -> AllowedECKeys:
            try:
                if isinstance(jwk, str):
                    obj = json.loads(jwk)
                elif isinstance(jwk, dict):
                    obj = jwk
                else:
                    raise ValueError
            except ValueError:
                raise InvalidKeyError("Key is not valid JSON")

            if obj.get("kty") != "EC":
                raise InvalidKeyError("Not an Elliptic curve key")

            if "x" not in obj or "y" not in obj:
                raise InvalidKeyError("Not an Elliptic curve key")

            x_val = obj.get("x")
            x_as_bytes = _ensure_bytes(x_val)
            rem = len(x_as_bytes) % 4
            if rem > 0:
                x_as_bytes += b"=" * (4 - rem)
            x = base64.urlsafe_b64decode(x_as_bytes)

            y_val = obj.get("y")
            y_as_bytes = _ensure_bytes(y_val)
            rem = len(y_as_bytes) % 4
            if rem > 0:
                y_as_bytes += b"=" * (4 - rem)
            y = base64.urlsafe_b64decode(y_as_bytes)

            curve = obj.get("crv")
            curve_obj: EllipticCurve
            if curve == "P-256":
                if len(x) == len(y) == 32:
                    curve_obj = SECP256R1()
                else:
                    raise InvalidKeyError("Coords should be 32 bytes for curve P-256")
            elif curve == "P-384":
                if len(x) == len(y) == 48:
                    curve_obj = SECP384R1()
                else:
                    raise InvalidKeyError("Coords should be 48 bytes for curve P-384")
            elif curve == "P-521":
                if len(x) == len(y) == 66:
                    curve_obj = SECP521R1()
                else:
                    raise InvalidKeyError("Coords should be 66 bytes for curve P-521")
            elif curve == "secp256k1":
                if len(x) == len(y) == 32:
                    curve_obj = SECP256K1()
                else:
                    raise InvalidKeyError("Coords should be 32 bytes for curve secp256k1")
            else:
                raise InvalidKeyError(f"Invalid curve: {curve}")

            public_numbers = EllipticCurvePublicNumbers(
                x=int.from_bytes(x, byteorder="big"),
                y=int.from_bytes(y, byteorder="big"),
                curve=curve_obj,
            )
            if "d" not in obj:
                return public_numbers.public_key()

            d_val = obj.get("d")
            d_as_bytes = _ensure_bytes(d_val)
            rem = len(d_as_bytes) % 4
            if rem > 0:
                d_as_bytes += b"=" * (4 - rem)
            d = base64.urlsafe_b64decode(d_as_bytes)

            if len(d) != len(x):
                raise InvalidKeyError("D should be the same length as x coordinate for curve " + curve)

            return EllipticCurvePrivateNumbers(
                int.from_bytes(d, byteorder="big"), public_numbers
            ).private_key()


    class RSAPSSAlgorithm(RSAAlgorithm):
        """
        Performs signing using RSASSA-PSS with MGF1.
        """

        def sign(self, msg: bytes, key: RSAPrivateKey) -> bytes:
            return key.sign(
                msg,
                padding.PSS(
                    mgf=padding.MGF1(self.hash_alg()),
                    salt_length=self.hash_alg().digest_size,
                ),
                self.hash_alg(),
            )

        def verify(self, msg: bytes, key: RSAPublicKey, sig: bytes) -> bool:
            try:
                key.verify(
                    sig,
                    msg,
                    padding.PSS(
                        mgf=padding.MGF1(self.hash_alg()),
                        salt_length=self.hash_alg().digest_size,
                    ),
                    self.hash_alg(),
                )
                return True
            except InvalidSignature:
                return False


    class OKPAlgorithm(Algorithm):
        """
        Performs signing and verification using EdDSA.
        Requires cryptography>=2.6.
        """

        def __init__(self, **kwargs: Any) -> None:
            pass

        def prepare_key(self, key: AllowedOKPKeys | str | bytes) -> AllowedOKPKeys:
            if isinstance(key, (bytes, str)):
                key_str = _ensure_bytes(key).decode("utf-8")
                key_bytes = _ensure_bytes(key)
                if "-----BEGIN PUBLIC" in key_str:
                    key = load_pem_public_key(key_bytes)
                elif "-----BEGIN PRIVATE" in key_str:
                    key = load_pem_private_key(key_bytes, password=None)
                elif key_str.startswith("ssh-"):
                    key = load_ssh_public_key(key_bytes)

            if not isinstance(
                key,
                (Ed25519PrivateKey, Ed25519PublicKey, Ed448PrivateKey, Ed448PublicKey),
            ):
                raise InvalidKeyError(
                    "Expecting an EdDSA key. Wrong key provided for EdDSA algorithms."
                )
            return key

        def sign(self, msg: str | bytes, key: Ed25519PrivateKey | Ed448PrivateKey) -> bytes:
            msg_bytes = _ensure_bytes(msg)
            return key.sign(msg_bytes)

        def verify(self, msg: str | bytes, key: AllowedOKPKeys, sig: str | bytes) -> bool:
            try:
                msg_bytes = _ensure_bytes(msg)
                sig_bytes = _ensure_bytes(sig)
                public_key = key.public_key() if hasattr(key, "public_key") else key
                public_key.verify(sig_bytes, msg_bytes)
                return True
            except InvalidSignature:
                return False

        @overload
        @staticmethod
        def to_jwk(key: AllowedOKPKeys, as_dict: Literal[True]) -> JWKDict:
            ...  # pragma: no cover

        @overload
        @staticmethod
        def to_jwk(key: AllowedOKPKeys, as_dict: Literal[False] = False) -> str:
            ...  # pragma: no cover

        @staticmethod
        def to_jwk(key: AllowedOKPKeys, as_dict: bool = False) -> Union[JWKDict, str]:
            if isinstance(key, (Ed25519PublicKey, Ed448PublicKey)):
                x = key.public_bytes(
                    encoding=Encoding.Raw,
                    format=PublicFormat.Raw,
                )
                crv = "Ed25519" if isinstance(key, Ed25519PublicKey) else "Ed448"
                obj = {
                    "x": base64.urlsafe_b64encode(x).replace(b"=", b"").decode(),
                    "kty": "OKP",
                    "crv": crv,
                }
                if as_dict:
                    return obj
                else:
                    return json.dumps(obj)

            if isinstance(key, (Ed25519PrivateKey, Ed448PrivateKey)):
                d = key.private_bytes(
                    encoding=Encoding.Raw,
                    format=PrivateFormat.Raw,
                    encryption_algorithm=NoEncryption(),
                )
                x = key.public_key().public_bytes(
                    encoding=Encoding.Raw,
                    format=PublicFormat.Raw,
                )
                crv = "Ed25519" if isinstance(key, Ed25519PrivateKey) else "Ed448"
                obj = {
                    "x": base64.urlsafe_b64encode(x).replace(b"=", b"").decode(),
                    "d": base64.urlsafe_b64encode(d).replace(b"=", b"").decode(),
                    "kty": "OKP",
                    "crv": crv,
                }
                if as_dict:
                    return obj
                else:
                    return json.dumps(obj)
            raise InvalidKeyError("Not a public or private key")

        @staticmethod
        def from_jwk(jwk: str | JWKDict) -> AllowedOKPKeys:
            try:
                if isinstance(jwk, str):
                    obj = json.loads(jwk)
                elif isinstance(jwk, dict):
                    obj = jwk
                else:
                    raise ValueError
            except ValueError:
                raise InvalidKeyError("Key is not valid JSON")

            if obj.get("kty") != "OKP":
                raise InvalidKeyError("Not an Octet Key Pair")

            curve = obj.get("crv")
            if curve not in ("Ed25519", "Ed448"):
                raise InvalidKeyError(f"Invalid curve: {curve}")

            if "x" not in obj:
                raise InvalidKeyError('OKP should have "x" parameter')
            x = base64.urlsafe_b64decode(_ensure_bytes(obj["x"]) + b"=" * ((4 - len(_ensure_bytes(obj["x"])) % 4) % 4))

            try:
                if "d" not in obj:
                    if curve == "Ed25519":
                        return Ed25519PublicKey.from_public_bytes(x)
                    return Ed448PublicKey.from_public_bytes(x)
                d = base64.urlsafe_b64decode(_ensure_bytes(obj["d"]) + b"=" * ((4 - len(_ensure_bytes(obj["d"])) % 4) % 4))
                if curve == "Ed25519":
                    return Ed25519PrivateKey.from_private_bytes(d)
                return Ed448PrivateKey.from_private_bytes(d)
            except ValueError as err:
                raise InvalidKeyError("Invalid key parameter") from err
