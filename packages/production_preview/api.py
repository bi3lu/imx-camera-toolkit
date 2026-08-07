"""Optional FastAPI signaling and HLS delivery for production preview."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from .config import PreviewTransport
from .transport import ProductionPreviewServer

VIEW_PATH = Path(__file__).parents[2] / "view" / "production.html"
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}


def _serialize_health(server: ProductionPreviewServer) -> dict[str, object]:
    """Convert immutable production metrics into a JSON-ready mapping."""
    stats = server.stats()
    return {
        "running": server.running,
        "capture": server.source_diagnostics(),
        "transport": stats.transport.value,
        "codec": stats.codec.value,
        "encoder_backend": stats.encoder_backend,
        "stream": {
            "stream_format": stats.stream.stream_format,
            "alignment": stats.stream.alignment,
            "profile": stats.stream.profile,
            "level": stats.stream.level,
            "profile_level_id": stats.stream.profile_level_id,
            "width": stats.stream.width,
            "height": stats.stream.height,
            "fps": stats.stream.fps,
            "has_codec_data": stats.stream.codec_data is not None,
            "has_sps": stats.stream.sps is not None,
            "has_pps": stats.stream.pps is not None,
        },
        "encode_fps": stats.encode.encode_fps,
        "bitrate_bps": stats.encode.bitrate_bps,
        "encoded_frames": stats.encode.encoded_frames,
        "encoded_bytes": stats.encode.encoded_bytes,
        "active_clients": stats.active_clients,
        "clients": [
            {
                "client_id": client.client_id,
                "transport": client.transport.value,
                "frames_sent": client.frames_sent,
                "bytes_sent": client.bytes_sent,
                "frames_pushed": client.frames_pushed,
                "pushed_bytes": client.pushed_bytes,
                "rtp_packets_sent": client.rtp_packets_sent,
                "rtp_bytes_sent": client.rtp_bytes_sent,
                "dropped_frames": client.dropped_frames,
                "drop_rate": client.drop_rate,
                "media_status": client.media_status,
                "last_rtp_packet_ns": client.last_rtp_packet_ns,
                "signaling_state": client.signaling_state,
                "ice_connection_state": client.ice_connection_state,
                "connection_state": client.connection_state,
                "parser_flow": client.parser_flow,
                "payloader_flow": client.payloader_flow,
                "negotiated_codec": client.negotiated_codec,
                "negotiated_fmtp": client.negotiated_fmtp,
                "last_bus_error": client.last_bus_error,
                "last_bus_warning": client.last_bus_warning,
                "rtt_ms": client.rtt_ms,
                "jitter_ms": client.jitter_ms,
                "packets_lost": client.packets_lost,
                "packets_received": client.packets_received,
                "bytes_received": client.bytes_received,
                "frames_received": client.frames_received,
                "frames_decoded": client.frames_decoded,
                "connected_at_ns": client.connected_at_ns,
                "last_seen_ns": client.last_seen_ns,
            }
            for client in stats.clients
        ],
    }


def create_production_preview_app(
    server: ProductionPreviewServer,
    *,
    manage_server: bool = True,
) -> FastAPI:
    """Create browser signaling/HLS endpoints around an existing camera source.

    The camera lifecycle always remains application-owned. With
    ``manage_server=True`` only the transport workers and peer pipelines are
    started and stopped by FastAPI.
    """
    if not isinstance(server, ProductionPreviewServer):
        raise TypeError("server must be a ProductionPreviewServer")

    if not isinstance(manage_server, bool):
        raise ValueError("manage_server must be a boolean")

    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        """Manage only production transport resources."""
        if manage_server:
            server.start()

        try:
            yield

        finally:
            if manage_server:
                server.stop()

    application = FastAPI(
        title="IMX Production Preview",
        description="Shared hardware encode with WebRTC or HLS delivery",
        version="0.5.0",
        lifespan=lifespan,
    )
    application.state.production_preview = server
    application.state.manage_server = manage_server

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        """Serve a video-element client that negotiates the configured mode."""
        try:
            html = VIEW_PATH.read_text("utf-8")

        except OSError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

        return HTMLResponse(html, headers=NO_CACHE_HEADERS)

    @application.get("/api/preview/health")
    def health() -> dict[str, object]:
        """Expose encode FPS/bitrate, clients, and per-client drop rates."""
        return _serialize_health(server)

    @application.post("/api/preview/session")
    def create_session() -> dict[str, object]:
        """Create a WebRTC peer or register one HLS browser."""
        try:
            if server.config.transport is PreviewTransport.WEBRTC:
                client_id, offer = server.create_webrtc_session()
                return {
                    "client_id": client_id,
                    "transport": "webrtc",
                    "offer": {"type": "offer", "sdp": offer},
                }

            client_id = server.create_hls_session()
            return {
                "client_id": client_id,
                "transport": "hls",
                "playlist_url": (
                    f"/api/preview/hls/{client_id}/playlist.m3u8"
                ),
            }

        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @application.post("/api/preview/webrtc/{client_id}/answer")
    def set_answer(
        client_id: str,
        values: dict[str, Any],
    ) -> dict[str, bool]:
        """Apply one browser SDP answer."""
        try:
            sdp = values.get("sdp")
            if not isinstance(sdp, str):
                raise ValueError("sdp must be a string")
            server.set_webrtc_answer(client_id, sdp)

        except KeyError as error:
            raise HTTPException(status_code=404, detail="unknown client") from error

        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        return {"accepted": True}

    @application.post("/api/preview/webrtc/{client_id}/candidate")
    def add_candidate(
        client_id: str,
        values: dict[str, Any],
    ) -> dict[str, bool]:
        """Apply one trickled browser ICE candidate."""
        try:
            mline_index = values.get("sdpMLineIndex")
            candidate = values.get("candidate")

            if (
                isinstance(mline_index, bool)
                or not isinstance(mline_index, int)
            ):
                raise ValueError("sdpMLineIndex must be an integer")

            if not isinstance(candidate, str):
                raise ValueError("candidate must be a string")

            server.add_webrtc_candidate(
                client_id,
                mline_index,
                candidate,
            )

        except KeyError as error:
            raise HTTPException(status_code=404, detail="unknown client") from error

        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        return {"accepted": True}

    @application.get("/api/preview/webrtc/{client_id}/candidates")
    def candidates(
        client_id: str,
        after: int = Query(default=0, ge=0),
    ) -> dict[str, object]:
        """Poll trickled server ICE candidates after a stable cursor."""
        try:
            items = server.webrtc_candidates(client_id, after)

        except KeyError as error:
            raise HTTPException(status_code=404, detail="unknown client") from error

        return {"candidates": items, "next": after + len(items)}

    @application.post("/api/preview/webrtc/{client_id}/feedback")
    def feedback(
        client_id: str,
        values: dict[str, Any],
    ) -> dict[str, bool]:
        """Record receiver-side WebRTC statistics from the browser view."""
        required = (
            "packets_received",
            "bytes_received",
            "frames_received",
            "frames_decoded",
        )
        try:
            for name in required:
                value = values.get(name)

                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(f"{name} must be a non-negative integer")

            packets_lost = values.get("packets_lost")

            if packets_lost is not None and (
                isinstance(packets_lost, bool)
                or not isinstance(packets_lost, int)
            ):
                raise ValueError("packets_lost must be an integer or null")

            optional_rates: dict[str, float | None] = {}

            for name in ("jitter_ms", "rtt_ms"):
                value = values.get(name)

                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value < 0
                ):
                    raise ValueError(
                        f"{name} must be a non-negative number or null"
                    )

                optional_rates[name] = (
                    None if value is None else float(value)
                )

            server.record_webrtc_feedback(
                client_id,
                packets_received=values["packets_received"],
                bytes_received=values["bytes_received"],
                frames_received=values["frames_received"],
                frames_decoded=values["frames_decoded"],
                packets_lost=packets_lost,
                jitter_ms=optional_rates["jitter_ms"],
                rtt_ms=optional_rates["rtt_ms"],
            )

        except KeyError as error:
            raise HTTPException(status_code=404, detail="unknown client") from error
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        return {"accepted": True}

    @application.get("/api/preview/hls/{client_id}/{asset}")
    def hls_asset(client_id: str, asset: str) -> FileResponse:
        """Serve a safe rolling playlist or MPEG-TS segment."""
        try:
            path = server.hls_asset(client_id, asset)

        except KeyError as error:
            raise HTTPException(status_code=404, detail="unknown client") from error

        except FileNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail="HLS asset unavailable",
            ) from error

        media_type = (
            "application/vnd.apple.mpegurl"
            if asset == "playlist.m3u8"
            else "video/mp2t"
        )

        headers = NO_CACHE_HEADERS if asset == "playlist.m3u8" else {}
        return FileResponse(path, media_type=media_type, headers=headers)

    @application.delete("/api/preview/session/{client_id}", status_code=204)
    def delete_session(client_id: str) -> None:
        """Close one browser session and release its latest-frame slot."""
        server.disconnect(client_id)

    return application


__all__ = ["create_production_preview_app"]
