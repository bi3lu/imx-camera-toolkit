"""Exercise local stdio shutdown and admission with disposable POSIX pipes."""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from imx_camera_toolkit import CameraConfig, CameraOpenError, CameraStats
from imx_camera_toolkit._internal.camera.publishing.jpeg_slot import JPEGPreview
from imx_camera_toolkit._internal.mcp import launcher

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.name != "posix", reason="POSIX pipes"),
]
_ROOT = Path(__file__).resolve().parents[2]
_META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
}


class _Camera:
    """Make lifecycle calls visible even when stdout writes bypass Python."""

    def __init__(self, config: CameraConfig, *, fail: bool = False) -> None:
        self.config = config
        self.fail = fail
        self.running = False

    def start(self) -> None:
        """Simulate a noisy native capture startup and optional partial failure."""
        assert not self.running
        self.running = True
        os.write(1, b"capture started\n")
        if self.fail:
            raise CameraOpenError("synthetic startup failure")

    def stop(self) -> None:
        """Record the first stop without emitting wire data."""
        assert self.running
        self.running = False
        print("capture stopped")

    def stats(self) -> CameraStats:
        """Return diagnostics for the one owned camera."""
        assert self.running
        return CameraStats(
            1, 0, 30.0, None, 0, 0, True, state="running", has_published_frame=True
        )

    def _jpeg_preview(self) -> JPEGPreview:
        """Keep positive snapshot waits pending until cancelled."""
        assert self.running
        print("sampled preview", flush=True)
        return JPEGPreview("running", self.config.enable_preview, True, 1, 1, 65, None)


def _launch(profile: str) -> int:
    """Run the actual entry point with a noisy camera and process exit hook."""

    def camera(config: CameraConfig) -> _Camera:
        """Construct the fixture using the real launcher's configuration."""
        assert config.sensor_id == 2
        return _Camera(config, fail=profile == "fail")

    atexit.register(os.write, 1, b"process exited\n")
    with patch.object(launcher, "Camera", camera):
        args = ["--sensor-id", "2"]
        if profile == "snapshot":
            args.append("--enable-snapshot")
        return launcher.main(args)


@asynccontextmanager
async def _process(
    profile: str = "observer", *, stdout: int = asyncio.subprocess.PIPE
) -> AsyncIterator[asyncio.subprocess.Process]:
    """Launch a fresh process and reap it even if a pipe assertion fails."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import runpy, sys; sys.exit(runpy.run_path(sys.argv[1])['_launch'](sys.argv[2]))",
        str(Path(__file__).resolve()),
        profile,
        cwd=_ROOT,
        stdin=asyncio.subprocess.PIPE,
        stdout=stdout,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        yield process
    finally:
        if process.stdin is not None:
            process.stdin.close()
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await asyncio.wait_for(process.wait(), 5)


async def _request(
    process: asyncio.subprocess.Process,
    request_id: int,
    method: str,
    **params: object,
) -> None:
    """Write one modern JSON-RPC request."""
    assert process.stdin is not None
    process.stdin.write(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": {"_meta": _META, **params},
            }
        ).encode()
        + b"\n"
    )
    await process.stdin.drain()


async def _response(process: asyncio.subprocess.Process) -> dict[str, Any]:
    """Parse exactly one wire line, rejecting incidental camera output."""
    assert process.stdout is not None
    response: dict[str, Any] = json.loads(await process.stdout.readline())
    return response


def test_owned_stdio_start_call_eof_and_relaunch() -> None:
    """Keep capture diagnostics on stderr, including startup and process exit."""
    pytest.importorskip("mcp")

    async def scenario() -> None:
        for _ in range(2):
            async with _process() as process:
                await _request(process, 1, "server/discover")
                assert (await _response(process))["id"] == 1
                await _request(process, 2, "tools/call", name="camera_status")
                assert (await _response(process))["result"]["structuredContent"][
                    "state"
                ] == "ready"
                assert process.stdin is not None and process.stdout is not None
                process.stdin.close()
                assert await process.stdout.read() == b""
                assert await process.wait() == 0
                assert process.stderr is not None
                stderr = (await process.stderr.read()).splitlines()
                assert stderr.count(b"capture started") == 1
                assert stderr.count(b"capture stopped") == 1
                assert stderr.count(b"process exited") == 1

    asyncio.run(asyncio.wait_for(scenario(), 15))


def test_owned_stdio_releases_partial_startup() -> None:
    """Fail before dispatch and stop capture after a partial camera startup."""
    pytest.importorskip("mcp")

    async def scenario() -> None:
        async with _process("fail") as process:
            stdout, stderr = await process.communicate()
            assert process.returncode == 1
            assert stdout == b""
            assert b"synthetic startup failure" in stderr
            assert stderr.count(b"capture started\n") == 1
            assert stderr.count(b"capture stopped\n") == 1

    asyncio.run(asyncio.wait_for(scenario(), 10))


@pytest.mark.parametrize("disconnect", [False, True], ids=["cancel", "eof"])
def test_stdio_cancels_an_active_snapshot(disconnect: bool) -> None:
    """Cancel an active preview wait explicitly or by closing the client write pipe."""
    pytest.importorskip("mcp")

    async def scenario() -> None:
        async with _process("snapshot") as process:
            await _request(
                process,
                1,
                "tools/call",
                name="camera_snapshot",
                arguments={"wait_ms": 1000},
            )
            assert process.stderr is not None and process.stdin is not None
            assert process.stdout is not None
            assert await process.stderr.readline() == b"capture started\n"
            assert await process.stderr.readline() == b"sampled preview\n"
            if not disconnect:
                process.stdin.write(
                    b'{"jsonrpc":"2.0","method":"notifications/cancelled",'
                    b'"params":{"requestId":1}}\n'
                )
                await _request(process, 2, "tools/list")
                assert (await _response(process))["id"] == 2
            process.stdin.close()
            output = await process.stdout.read()
            for line in output.splitlines():
                assert "error" in json.loads(line)
            assert await process.wait() == 0
            assert (await process.stderr.read()).count(b"capture stopped\n") == 1

    asyncio.run(asyncio.wait_for(scenario(), 10))


@pytest.mark.parametrize("limit", ["frame", "duplicate", "active", "backpressure"])
def test_stdio_closes_over_budget_pipes(limit: str) -> None:
    """Stop capture even when the client holds its stdin open past a limit."""
    pytest.importorskip("mcp")

    async def scenario() -> None:
        read_end, write_end = os.pipe()
        try:
            blocked_stdout = limit in {"active", "backpressure"}
            if blocked_stdout:
                os.set_blocking(write_end, False)
                with suppress(BlockingIOError):
                    while True:
                        os.write(write_end, b"x" * 4096)
                with suppress(BlockingIOError):
                    while True:
                        os.write(write_end, b"x")
            stdout = write_end if blocked_stdout else asyncio.subprocess.PIPE
            async with _process("snapshot", stdout=stdout) as process:
                assert process.stdin is not None and process.stderr is not None
                if limit == "frame":
                    process.stdin.write(b"x" * (64 * 1024 + 1))
                    await process.stdin.drain()
                    expected = b"request exceeds 64 KiB"
                elif limit == "duplicate":
                    for _ in range(2):
                        await _request(
                            process,
                            1,
                            "tools/call",
                            name="camera_snapshot",
                            arguments={"wait_ms": 1000},
                        )
                    expected = b"duplicate or excess active request"
                elif limit == "active":
                    for request_id in range(9):
                        await _request(process, request_id, "server/discover")
                    expected = b"duplicate or excess active request"
                else:
                    await _request(process, 1, "server/discover")
                    expected = b"response write timed out"
                assert await process.wait() == 0
                stderr = await process.stderr.read()
                assert expected in stderr
                assert stderr.count(b"capture stopped\n") == 1
        finally:
            os.close(read_end)
            os.close(write_end)

    asyncio.run(asyncio.wait_for(scenario(), 10))
