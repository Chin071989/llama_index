import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator

import pytest

from llama_index.core import utils
from llama_index.core.utils import fetch_url_bytes, resolve_binary


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/ok":
            body = b"hello"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/big":
            body = b"x" * 2048
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/redirect-local":
            self.send_response(302)
            self.send_header("Location", "/ok")
            self.end_headers()
        elif self.path == "/redirect-metadata":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture()
def server_url() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture()
def trust_loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat 127.0.0.1 as public so the local test server is reachable."""
    original = utils._is_blocked_ip
    monkeypatch.setattr(
        utils,
        "_is_blocked_ip",
        lambda ip: False if str(ip) == "127.0.0.1" else original(ip),
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://0.0.0.0/",
    ],
)
def test_non_public_addresses_are_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="non-public"):
        resolve_binary(url=url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x"])
def test_non_http_schemes_are_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="scheme"):
        fetch_url_bytes(url)


def test_fetch_and_follow_safe_redirect(
    server_url: str, trust_loopback_only: None
) -> None:
    assert resolve_binary(url=f"{server_url}/ok").read() == b"hello"
    assert fetch_url_bytes(f"{server_url}/redirect-local") == b"hello"


def test_redirect_to_metadata_is_rejected(
    server_url: str, trust_loopback_only: None
) -> None:
    with pytest.raises(ValueError, match="non-public"):
        fetch_url_bytes(f"{server_url}/redirect-metadata")


def test_size_cap(server_url: str, trust_loopback_only: None) -> None:
    with pytest.raises(ValueError, match="byte limit"):
        fetch_url_bytes(f"{server_url}/big", max_bytes=1024)


def test_size_cap_from_env(
    server_url: str, trust_loopback_only: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(utils.MAX_URL_FETCH_BYTES_ENV, "1024")
    with pytest.raises(ValueError, match="byte limit"):
        resolve_binary(url=f"{server_url}/big")


def test_private_urls_allowed_by_env_opt_in(
    server_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="non-public"):
        fetch_url_bytes(f"{server_url}/ok")
    monkeypatch.setenv(utils.ALLOW_PRIVATE_URLS_ENV, "1")
    assert fetch_url_bytes(f"{server_url}/ok") == b"hello"
