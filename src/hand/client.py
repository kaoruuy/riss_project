from __future__ import annotations

import socket
import struct
from collections.abc import Sequence


DOF_NAMES = (
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotation",
)

POS_SET = 1474
ANGLE_SET = 1486
SPEED_SET = 1522
POS_ACT = 1534
ANGLE_ACT = 1546
FORCE_ACT = 1582


class ModbusError(RuntimeError):
    """Raised when the hand returns an invalid or Modbus exception response."""


class InspireHand:
    def __init__(
        self,
        host: str,
        port: int = 6000,
        unit_id: int = 1,
        timeout: float = 2.0,
    ) -> None:
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.timeout = timeout
        self._socket: socket.socket | None = None
        self._transaction_id = 0

    def __enter__(self) -> InspireHand:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def connect(self) -> None:
        if self._socket is None:
            try:
                self._socket = socket.create_connection(
                    (self.host, self.port), timeout=self.timeout
                )
            except TimeoutError as error:
                raise TimeoutError(
                    f"timed out connecting to {self.host}:{self.port}"
                ) from error
            self._socket.settimeout(self.timeout)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def read_holding_registers(self, address: int, count: int) -> list[int]:
        response = self._request(struct.pack(">BHH", 0x03, address, count))
        if response[0] != 0x03:
            raise ModbusError(f"unexpected read function code: 0x{response[0]:02x}")
        if response[1] != count * 2 or len(response) != 2 + count * 2:
            raise ModbusError("invalid read response length")
        return list(struct.unpack(f">{count}H", response[2:]))

    def write_registers(self, address: int, values: Sequence[int]) -> None:
        encoded = [_encode_int16(value) for value in values]
        payload = struct.pack(
            f">BHHB{len(encoded)}H",
            0x10,
            address,
            len(encoded),
            len(encoded) * 2,
            *encoded,
        )
        response = self._request(payload)
        if response[0] != 0x10 or len(response) != 5:
            raise ModbusError("invalid write response")
        response_address, response_count = struct.unpack(">HH", response[1:])
        if response_address != address or response_count != len(encoded):
            raise ModbusError("write acknowledgement does not match request")

    def _request(self, pdu: bytes) -> bytes:
        self.connect()
        assert self._socket is not None

        self._transaction_id = (self._transaction_id + 1) & 0xFFFF
        header = struct.pack(
            ">HHHB", self._transaction_id, 0, len(pdu) + 1, self.unit_id
        )
        self._socket.sendall(header + pdu)

        try:
            response_header = _recv_exact(self._socket, 7)
        except TimeoutError as error:
            raise TimeoutError("timed out waiting for a response from the hand") from error
        _, protocol_id, response_length, response_unit = struct.unpack(
            ">HHHB", response_header
        )
        if protocol_id != 0:
            raise ModbusError(f"unexpected protocol ID: {protocol_id}")
        if response_unit != self.unit_id:
            raise ModbusError(f"unexpected unit ID: {response_unit}")
        if response_length < 2:
            raise ModbusError("invalid response length")

        # RH56 firmware may echo an incorrect transaction ID, so it is drained
        # but intentionally not validated.
        try:
            response = _recv_exact(self._socket, response_length - 1)
        except TimeoutError as error:
            raise TimeoutError("timed out while receiving the hand response") from error
        if response[0] & 0x80:
            code = response[1] if len(response) > 1 else -1
            raise ModbusError(f"Modbus exception code: {code}")
        return response


def _encode_int16(value: int) -> int:
    if not -32768 <= value <= 65535:
        raise ValueError(f"value outside 16-bit range: {value}")
    return value & 0xFFFF


def decode_int16(value: int) -> int:
    """Decode a Modbus register as a signed 16-bit value."""
    return value - 0x10000 if value & 0x8000 else value


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("hand closed the TCP connection")
        chunks.extend(chunk)
    return bytes(chunks)
