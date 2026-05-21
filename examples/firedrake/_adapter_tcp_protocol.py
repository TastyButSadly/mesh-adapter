from __future__ import annotations

import json
import socket
import struct
from typing import Any

import numpy as np

_HEADER_STRUCT = struct.Struct("!Q")


def send_message(sock: socket.socket, *, scalars: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    prepared_arrays = {}
    array_meta = []
    for name, array in arrays.items():
        contiguous = np.ascontiguousarray(array)
        prepared_arrays[name] = contiguous
        array_meta.append(
            {
                "name": name,
                "dtype": contiguous.dtype.str,
                "shape": contiguous.shape,
                "nbytes": contiguous.nbytes,
            }
        )

    header = json.dumps({"scalars": scalars, "arrays": array_meta}, separators=(",", ":")).encode("utf-8")
    sock.sendall(_HEADER_STRUCT.pack(len(header)))
    sock.sendall(header)
    for meta in array_meta:
        sock.sendall(memoryview(prepared_arrays[meta["name"]]).cast("B"))


def recv_message(sock: socket.socket) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    size_data = _recv_exact(sock, _HEADER_STRUCT.size)
    if size_data is None:
        return None
    header_size = _HEADER_STRUCT.unpack(size_data)[0]
    header_data = _recv_exact(sock, header_size)
    if header_data is None:
        return None
    header = json.loads(header_data.decode("utf-8"))
    arrays = {}
    for meta in header["arrays"]:
        payload = _recv_exact(sock, int(meta["nbytes"]))
        if payload is None:
            return None
        array = np.frombuffer(payload, dtype=np.dtype(meta["dtype"])).reshape(tuple(meta["shape"]))
        arrays[meta["name"]] = array
    return header["scalars"], arrays


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
