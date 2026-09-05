"""Read-only BrainVision Recorder RDA client (BrainAmp MR via Recorder).

Wire format: Recorder manual, Remote Data Access chapter. Scaling for BOTH
DATA16 and DATA32 follows the LSL BrainVisionRDA reference implementation:
https://github.com/labstreaminglayer/App-BrainProducts/blob/master/BrainVisionRDA/rda_message.h
RDA values must be multiplied by the per-channel resolution in microvolts.
"""
from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
import time

import numpy as np

GUID = bytes.fromhex("8e45584396c9864caf4a98bbf6c91450")
HEADER = struct.Struct("<16sII")
MAX_PACKET = 32 * 1024 * 1024


class RDAError(ValueError):
    """Invalid stream or a discontinuity that cannot safely be corrected."""


@dataclass(frozen=True)
class StreamInfo:
    names: tuple[str, ...]
    sfreq: float
    resolutions_uv: np.ndarray


@dataclass(frozen=True)
class Marker:
    position: int
    points: int
    channel: int
    kind: str
    description: str


@dataclass(frozen=True)
class DataBlock:
    number: int
    volts: np.ndarray  # channels x samples; calibrated once, at the boundary
    markers: tuple[Marker, ...]


def _cstring(body, offset):
    end = body.find(b"\0", offset)
    if end < 0:
        raise RDAError("Unterminated RDA text field")
    value = body[offset:end]
    try:
        return value.decode("utf-8"), end + 1
    except UnicodeDecodeError:
        return value.decode("cp1252", errors="replace"), end + 1


def parse_start(body):
    if len(body) < 12:
        raise RDAError("Truncated RDA start message")
    channels, interval = struct.unpack_from("<Id", body)
    if not 1 <= channels <= 512 or not np.isfinite(interval) or interval <= 0:
        raise RDAError("Invalid RDA channel count or sampling interval")
    offset = 12 + channels * 8
    if len(body) < offset:
        raise RDAError("Missing RDA channel resolutions")
    resolutions = np.frombuffer(body, "<f8", channels, 12).copy()
    if not np.all(np.isfinite(resolutions) & (resolutions > 0)):
        raise RDAError("RDA channel resolutions must be finite and positive")
    names = []
    for _ in range(channels):
        name, offset = _cstring(body, offset)
        names.append(name.strip())
    if any(not name for name in names) or len(set(names)) != channels:
        raise RDAError("RDA channel names must be nonempty and unique")
    if offset != len(body):
        raise RDAError("Unexpected bytes after RDA start message")
    return StreamInfo(tuple(names), 1e6 / interval, resolutions)


def parse_data(body, kind, info):
    if kind not in (2, 4) or len(body) < 12:
        raise RDAError("Invalid RDA data message")
    number, points, nmarkers = struct.unpack_from("<III", body)
    dtype = np.dtype("<i2" if kind == 2 else "<f4")
    values = points * len(info.names)
    offset = 12 + values * dtype.itemsize
    if not points or offset > len(body) or nmarkers > (len(body) - offset) // 18:
        raise RDAError("Truncated or invalid RDA sample/marker count")
    data = np.frombuffer(body, dtype, values, 12).reshape(points, len(info.names)).T
    volts = data.astype(np.float64) * (info.resolutions_uv[:, None] * 1e-6)
    if not np.all(np.isfinite(volts)):
        raise RDAError("Non-finite samples in the RDA stream")
    markers = []
    for _ in range(nmarkers):
        if offset + 16 > len(body):
            raise RDAError("Truncated RDA marker")
        size, pos, length, channel = struct.unpack_from("<IIIi", body, offset)
        if size < 18 or offset + size > len(body) or pos >= points:
            raise RDAError("Invalid RDA marker size or position")
        marker_body = body[offset + 16:offset + size]
        marker_type, idx = _cstring(marker_body, 0)
        description, idx = _cstring(marker_body, idx)
        if idx != len(marker_body):
            raise RDAError("Unexpected bytes after RDA marker")
        markers.append(Marker(pos, length, channel, marker_type, description))
        offset += size
    if offset != len(body):
        raise RDAError("Unexpected bytes after RDA samples/markers")
    return DataBlock(number, volts, tuple(markers))


class BlockSequence:
    def __init__(self):
        self.last = None

    def check(self, number):
        if self.last is not None and number != (self.last + 1) % 2**32:
            raise RDAError(f"RDA block discontinuity: {self.last} -> {number}. "
                           "Reconnect to rebuild the artifact template.")
        self.last = number


def receive_packet(sock, cancelled=lambda: False, timeout_s=10):
    """Read partial TCP frames without losing bytes on socket timeouts.

    The caller sets a short socket timeout for responsive cancellation. A
    deadline also rejects a peer that trickles an unfinished packet forever.
    """
    deadline = time.monotonic() + timeout_s

    def exact(size):
        data = bytearray()
        while len(data) < size:
            if cancelled():
                raise InterruptedError("Disconnected")
            if time.monotonic() > deadline:
                raise TimeoutError("No complete RDA message for 10 seconds. Check Recorder monitoring.")
            try:
                part = sock.recv(min(size - len(data), 65536))
            except socket.timeout:
                continue
            if not part:
                raise EOFError("Recorder closed the RDA connection")
            data.extend(part)
        return bytes(data)

    guid, size, kind = HEADER.unpack(exact(HEADER.size))
    if guid != GUID:
        raise RDAError("This server did not send a BrainVision RDA header. Check host and port.")
    if not HEADER.size <= size <= MAX_PACKET:
        raise RDAError(f"Invalid RDA packet length: {size}")
    return kind, exact(size - HEADER.size)
