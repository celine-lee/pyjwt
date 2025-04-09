
from __future__ import annotations

import base64
import binascii
import json
import warnings
from typing import TYPE_CHECKING, Any

from .algorithms import (
    Algorithm,
    get_default_algorithms,
    has_crypto,
    requires_cryptography,
)
from .exceptions import (
    DecodeError,
    InvalidAlgorithmError,
    InvalidSignatureError,
    InvalidTokenError,
)
from .warnings import RemovedInPyjwt3Warning

if TYPE_CHECKING:
    from .algorithms import AllowedPrivateKeys, AllowedPublicKeys


# -----------------------
# Helper functions
# -----------------------

def _ensure_bytes(val: str | bytes) -> bytes:
    if isinstance(val, bytes):
        return val
    elif isinstance(val, str):
        return val.encode("utf-8")
    else:
        raise TypeError("Expected a string or bytes value")


def _b64encode(data: bytes) -> bytes:
    """Return a URL-safe Base64 encoding without padding."""
    return base64.urlsafe_b64encode(data).replace(b"=", b"")


def _b64decode(data: bytes) -> bytes:
    """Decode URL-safe Base64, adding any missing padding."""
    data = _ensure_bytes(data)
    rem = len(data) % 4
    if rem:
        data += b"=" * (4 - rem)
    return base64.urlsafe_b64decode(data)


class PyJWS:
    header_typ = "JWT"

    def __init__(
        self,
        algorithms: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self._algorithms = get_default_algorithms()
        self._valid_algs = set(algorithms) if algorithms is not None else set(self._algorithms)

        # Remove algorithms that are not in the whitelist.
        for key in list(self._algorithms.keys()):
            if key not in self._valid_algs:
                del self._algorithms[key]

        if options is None:
            options = {}
        self.options = {**self._get_default_options(), **options}

    @staticmethod
    def _get_default_options() -> dict[str, bool]:
        return {"verify_signature": True}

    def register_algorithm(self, alg_id: str, alg_obj: Algorithm) -> None:
        if alg_id in self._algorithms:
            raise ValueError("Algorithm already has a handler.")
        if not isinstance(alg_obj, Algorithm):
            raise TypeError("Object is not of type `Algorithm`")
        self._algorithms[alg_id] = alg_obj
        self._valid_algs.add(alg_id)

    def unregister_algorithm(self, alg_id: str) -> None:
        if alg_id not in self._algorithms:
            raise KeyError("The specified algorithm is not registered and cannot be removed.")
        del self._algorithms[alg_id]
        self._valid_algs.remove(alg_id)

    def get_algorithms(self) -> list[str]:
        return list(self._valid_algs)

    def get_algorithm_by_name(self, alg_name: str) -> Algorithm:
        try:
            return self._algorithms[alg_name]
        except KeyError as e:
            if not has_crypto and alg_name in requires_cryptography:
                raise NotImplementedError(
                    f"Algorithm '{alg_name}' not found. Do you have cryptography installed?"
                ) from e
            raise NotImplementedError("Algorithm not supported") from e

    def encode(
        self,
        payload: bytes,
        key: AllowedPrivateKeys | str | bytes,
        algorithm: str | None = "HS256",
        headers: dict[str, Any] | None = None,
        json_encoder: type[json.JSONEncoder] | None = None,
        is_payload_detached: bool = False,
        sort_headers: bool = True,
    ) -> str:
        segments = []
        algorithm_ = algorithm if algorithm is not None else "none"

        if headers:
            if "alg" in headers:
                algorithm_ = headers["alg"]
            if headers.get("b64") is False:
                is_payload_detached = True

        header: dict[str, Any] = {"typ": self.header_typ, "alg": algorithm_}
        if headers:
            self._validate_headers(headers)
            header.update(headers)
        if not header.get("typ"):
            header.pop("typ", None)
        if is_payload_detached:
            header["b64"] = False
        elif "b64" in header:
            header.pop("b64")
        json_header = json.dumps(
            header, separators=(",", ":"), cls=json_encoder, sort_keys=sort_headers
        ).encode("utf-8")
        segments.append(_b64encode(json_header))

        if is_payload_detached:
            msg_payload = payload
        else:
            msg_payload = _b64encode(payload)
        segments.append(msg_payload)

        signing_input = b".".join(segments)
        alg_obj = self.get_algorithm_by_name(algorithm_)
        prepared_key = alg_obj.prepare_key(key)
        signature = alg_obj.sign(signing_input, prepared_key)
        segments.append(_b64encode(signature))

        if is_payload_detached:
            segments[1] = b""
        encoded_string = b".".join(segments)
        return encoded_string.decode("utf-8")

    def decode_complete(
        self,
        jwt: str | bytes,
        key: AllowedPublicKeys | str | bytes = "",
        algorithms: list[str] | None = None,
        options: dict[str, Any] | None = None,
        detached_payload: bytes | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if kwargs:
            warnings.warn(
                "Passing additional kwargs to decode_complete() is deprecated "
                "and will be removed in pyjwt version 3. Unsupported kwargs: "
                f"{tuple(kwargs.keys())}",
                RemovedInPyjwt3Warning,
            )
        if options is None:
            options = {}
        merged_options = {**self.options, **options}
        verify_signature = merged_options["verify_signature"]
        if verify_signature and not algorithms:
            raise DecodeError(
                'You must pass an "algorithms" argument when verifying a token.'
            )

        payload, signing_input, header, signature = self._load(jwt)
        if header.get("b64", True) is False:
            if detached_payload is None:
                raise DecodeError(
                    'A "detached_payload" argument is required when the b64 header is set to false.'
                )
            payload = detached_payload
            signing_input = b".".join([signing_input.rsplit(b".", 1)[0], payload])

        if verify_signature:
            self._verify_signature(signing_input, header, signature, key, algorithms)

        return {
            "payload": payload,
            "header": header,
            "signature": signature,
        }

    def decode(
        self,
        jwt: str | bytes,
        key: AllowedPublicKeys | str | bytes = "",
        algorithms: list[str] | None = None,
        options: dict[str, Any] | None = None,
        detached_payload: bytes | None = None,
        **kwargs,
    ) -> Any:
        if kwargs:
            warnings.warn(
                "Passing additional kwargs to decode() is deprecated "
                "and will be removed in pyjwt version 3. Unsupported kwargs: "
                f"{tuple(kwargs.keys())}",
                RemovedInPyjwt3Warning,
            )
        decoded = self.decode_complete(
            jwt, key, algorithms, options, detached_payload=detached_payload
        )
        return decoded["payload"]

    def get_unverified_header(self, jwt: str | bytes) -> dict[str, Any]:
        headers = self._load(jwt)[2]
        self._validate_headers(headers)
        return headers

    def _load(self, jwt: str | bytes) -> tuple[bytes, bytes, dict[str, Any], bytes]:
        jwt = _ensure_bytes(jwt)
        try:
            signing_input, crypto_segment = jwt.rsplit(b".", 1)
            header_segment, payload_segment = signing_input.split(b".", 1)
        except ValueError as err:
            raise DecodeError("Not enough segments") from err

        try:
            header_bytes = _ensure_bytes(header_segment)
            header_bytes = _b64decode(header_bytes)
        except (TypeError, binascii.Error) as err:
            raise DecodeError("Invalid header padding") from err

        try:
            header = json.loads(header_bytes)
        except ValueError as e:
            raise DecodeError(f"Invalid header string: {e}") from e

        if not isinstance(header, dict):
            raise DecodeError("Header is not a valid JSON object")

        try:
            payload_bytes = _ensure_bytes(payload_segment)
            payload_bytes = _b64decode(payload_bytes)
            payload = payload_bytes
        except (TypeError, binascii.Error) as err:
            raise DecodeError("Invalid payload padding") from err

        try:
            crypto_bytes = _ensure_bytes(crypto_segment)
            crypto_bytes = _b64decode(crypto_bytes)
            signature = crypto_bytes
        except (TypeError, binascii.Error) as err:
            raise DecodeError("Invalid crypto padding") from err

        return (payload, signing_input, header, signature)

    def _verify_signature(
        self,
        signing_input: bytes,
        header: dict[str, Any],
        signature: bytes,
        key: AllowedPublicKeys | str | bytes = "",
        algorithms: list[str] | None = None,
    ) -> None:
        try:
            alg = header["alg"]
        except KeyError:
            raise InvalidAlgorithmError("Algorithm not specified in header")
        if not alg or (algorithms is not None and alg not in algorithms):
            raise InvalidAlgorithmError("The specified alg value is not allowed")
        try:
            alg_obj = self.get_algorithm_by_name(alg)
        except NotImplementedError as e:
            raise InvalidAlgorithmError("Algorithm not supported") from e

        prepared_key = alg_obj.prepare_key(key)
        if not alg_obj.verify(signing_input, prepared_key, signature):
            raise InvalidSignatureError("Signature verification failed")

    def _validate_headers(self, headers: dict[str, Any]) -> None:
        if "kid" in headers:
            self._validate_kid(headers["kid"])

    def _validate_kid(self, kid: Any) -> None:
        if not isinstance(kid, str):
            raise InvalidTokenError("Key ID header parameter must be a string")


_jws_global_obj = PyJWS()
encode = _jws_global_obj.encode
decode_complete = _jws_global_obj.decode_complete
decode = _jws_global_obj.decode
register_algorithm = _jws_global_obj.register_algorithm
unregister_algorithm = _jws_global_obj.unregister_algorithm
get_algorithm_by_name = _jws_global_obj.get_algorithm_by_name
get_unverified_header = _jws_global_obj.get_unverified_header
