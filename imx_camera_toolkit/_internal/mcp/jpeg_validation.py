"""Bounded structural checks for JPEGs from application-owned encoders."""

from __future__ import annotations

import re

_SCAN_MARKER = re.compile(rb"\xff([^\x00\xd0-\xd7\xff])")


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Read the dimensions of one complete JPEG without decoding pixels.

    Reject EXIF, malformed markers and scan boundaries, and trailing data.
    A trusted encoder is responsible for the validity of the entropy data.

    Raises:
        ValueError: If the JPEG structure violates these requirements.
    """
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("missing JPEG start marker")

    offset = 2
    dimensions: tuple[int, int] | None = None
    scanned = False
    components = 0
    quantization_table = False
    huffman_table = False

    while offset < len(data):
        if data[offset] != 0xFF:
            raise ValueError("missing JPEG marker")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset == len(data):
            break

        marker = data[offset]
        offset += 1
        if marker == 0xD9:
            if dimensions is not None and scanned and offset == len(data):
                return dimensions
            break
        if marker < 0xC0 or 0xD0 <= marker <= 0xD8:
            raise ValueError("invalid JPEG segment marker")

        size = int.from_bytes(data[offset : offset + 2], "big")
        end = offset + size
        start = offset + 2
        if size < 2 or end > len(data):
            raise ValueError("truncated JPEG segment")

        if marker == 0xE1 and data.startswith(b"Exif\x00\x00", start, end):
            raise ValueError("JPEG contains EXIF")
        if marker == 0xDB:
            if size < 67:
                raise ValueError("truncated JPEG quantization table")
            quantization_table = True
        if marker == 0xC4:
            if size < 20:
                raise ValueError("truncated JPEG Huffman table")
            huffman_table = True
        if marker in (0xC0, 0xC1, 0xC2):
            if dimensions is not None or size < 8:
                raise ValueError("invalid JPEG frame header")
            height = int.from_bytes(data[start + 1 : start + 3], "big")
            width = int.from_bytes(data[start + 3 : start + 5], "big")
            components = data[start + 5]
            if not width or not height or not components or size != 8 + 3 * components:
                raise ValueError("invalid JPEG frame dimensions or components")
            dimensions = width, height

        offset = end
        if marker == 0xDA:
            if (
                dimensions is None
                or not quantization_table
                or not huffman_table
                or size < 6
            ):
                raise ValueError("invalid JPEG scan header")
            scan_components = data[start]
            if (
                not 1 <= scan_components <= components
                or size != 6 + 2 * scan_components
            ):
                raise ValueError("invalid JPEG scan components")
            match = _SCAN_MARKER.search(data, end)
            if match is None or match.start() == end:
                raise ValueError("missing JPEG scan data or end marker")
            scanned = True
            offset = match.start()

    raise ValueError("incomplete JPEG or trailing data")
