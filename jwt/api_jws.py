
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


class PyJWS:
    header_typ = "JWT"

    def __init__(
        self,
        algorithms: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self._algorithms = get_default_algorithms()
        self._valid_algs = set(algorithms) if algorithms is not None else set(self._algorithms)
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
        """
        Registers a new Algorithm for use in token creation/verification.
        """
        if alg_id in self._algorithms:
            raise ValueError("Algorithm already has a handler.")
        if not isinstance(alg_obj, Algorithm):
            raise TypeError("Object is not of type `Algorithm`")
        self._algorithms[alg_id] = alg_obj
        self._valid_algs.add(alg_id)

    def unregister_algorithm(self, alg_id: str) -> None:
        """
        Unregisters an Algorithm.
        """
        if alg_id not in self._algorithms:
            raise KeyError("The specified algorithm could not be removed because it is not registered.")
        del self._algorithms[alg_id]
        self._valid_algs.remove(alg_id)

    def get_algorithms(self) -> list[str]:
        """
        Returns supported 'alg' parameters.
        """
        return list(self._valid_algs)

    def get_algorithm_by_name(self, alg_name: str) -> Algorithm:
        """
        Returns the Algorithm object matching the given name.
        """
        try:
            return self._algorithms[alg_name]
        except KeyError as e:
            if not has_crypto and alg_name in requires_cryptography:
                raise NotImplementedError(
                    f"Algorithm '{alg_name}' could not be found. Do you have cryptography installed?"
                ) from e
            raise NotImplementedError("Algorithm not supported") from e

    @staticmethod
    def _b64_encode(data: bytes) -> bytes:
        """
        Base64url-encode data without padding.
        """
        return base64.urlsafe_b64encode(data).replace(b"=", b"")

    @staticmethod
    def _b64_decode(segment: bytes) -> bytes:
        """
        Decodes a base64url string, adding padding if necessary.
        """
        rem = len(segment) % 4
        if rem:
            segment += b"=" * (4 - rem)
        return base64.urlsafe_b64decode(segment)

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
            headers_alg = headers.get("alg")
            if headers_alg:
                algorithm_ = headers["alg"]
            if headers.get("b64") is False:
                is_payload_detached = True
        header: dict[str, Any] = {"typ": self.header_typ, "alg": algorithm_}
        if headers:
            self._validate_headers(headers)
            header.update(headers)
        if not header.get("typ"):
            del header["typ"]
        if is_payload_detached:
            header["b64"] = False
        elif "b64" in header:
            del header["b64"]
        json_header = json.dumps(header, separators=(",", ":"), cls=json_encoder, sort_keys=sort_headers).encode()
        segments.append(self._b64_encode(json_header))
        msg_payload = payload if is_payload_detached else self._b64_encode(payload)
        segments.append(msg_payload)
        signing_input = b".".join(segments)
        alg_obj = self.get_algorithm_by_name(algorithm_)
        prepared_key = alg_obj.prepare_key(key)
        signature = alg_obj.sign(signing_input, prepared_key)
        segments.append(self._b64_encode(signature))
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
                "passing additional kwargs to decode_complete() is deprecated and will be removed in pyjwt version 3. "
                f"Unsupported kwargs: {tuple(kwargs.keys())}",
                RemovedInPyjwt3Warning,
            )
        if options is None:
            options = {}
        merged_options = {**self.options, **options}
        verify_signature = merged_options["verify_signature"]
        if verify_signature and not algorithms:
            raise DecodeError(
                'It is required that you pass in a value for the "algorithms" argument when calling decode().'
            )
        payload, signing_input, header, signature = self._load(jwt)
        if header.get("b64", True) is False:
            if detached_payload is None:
                raise DecodeError(
                    'A "detached_payload" argument is required to decode a message with b64 header set to false.'
                )
            payload = detached_payload
            signing_input = b".".join([signing_input.rsplit(b".", 1)[0], payload])
        if verify_signature:
            self._verify_signature(signing_input, header, signature, key, algorithms)
        return {"payload": payload, "header": header, "signature": signature}

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
                "passing additional kwargs to decode() is deprecated and will be removed in pyjwt version 3. "
                f"Unsupported kwargs: {tuple(kwargs.keys())}",
                RemovedInPyjwt3Warning,
            )
        decoded = self.decode_complete(jwt, key, algorithms, options, detached_payload=detached_payload)
        return decoded["payload"]

    def get_unverified_header(self, jwt: str | bytes) -> dict[str, Any]:
        """
        Returns the JWT header parameters without verifying the signature.
        """
        headers = self._load(jwt)[2]
        self._validate_headers(headers)
        return headers

    def _load(self, jwt: str | bytes) -> tuple[bytes, bytes, dict[str, Any], bytes]:
        if isinstance(jwt, str):
            jwt = jwt.encode("utf-8")
        if not isinstance(jwt, bytes):
            raise DecodeError(f"Invalid token type. Token must be bytes")
        try:
            signing_input, crypto_segment = jwt.rsplit(b".", 1)
            header_segment, payload_segment = signing_input.split(b".", 1)
        except ValueError as err:
            raise DecodeError("Not enough segments") from err
        try:
            header_bytes = header_segment if isinstance(header_segment, bytes) else header_segment.encode("utf-8")
            header_bytes = self._b64_decode(header_bytes)
        except (TypeError, binascii.Error) as err:
            raise DecodeError("Invalid header padding") from err
        try:
            header = json.loads(header_bytes)
        except ValueError as e:
            raise DecodeError(f"Invalid header string: {e}") from e
        if not isinstance(header, dict):
            raise DecodeError("Invalid header string: must be a JSON object")
        try:
            payload_bytes = payload_segment if isinstance(payload_segment, bytes) else payload_segment.encode("utf-8")
            payload = self._b64_decode(payload_bytes)
        except (TypeError, binascii.Error) as err:
            raise DecodeError("Invalid payload padding") from err
        try:
            crypto_bytes = crypto_segment if isinstance(crypto_segment, bytes) else crypto_segment.encode("utf-8")
            signature = self._b64_decode(crypto_bytes)
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
            raise InvalidAlgorithmError("Algorithm not specified")
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
