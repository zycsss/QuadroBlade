"""Serve the replay site together with the debug-run logs it replays.

The site itself is static and can be opened straight from disk: drop a ``.npz`` log onto it and it
reads the file in the browser. This little server exists so that a page opened at
``http://127.0.0.1:8765/`` can instead list every log in the repository's ``logs/`` directory and
open one from a picker, and so that the matplotlib figure for the same run is one click away.

Run it with the repository's python, from anywhere:

    pixi run python replay_site/serve.py

or, without the environment:

    python replay_site/serve.py --port 8765 --no-browser

Pass ``--log-dir`` to browse another directory of run logs, e.g. a sweep written somewhere else:

    python replay_site/serve.py --log-dir .tmp/logs

Routes:

* ``/`` and ``/assets/...`` - the site itself.
* ``/api/logs`` - JSON list of the ``.npz`` logs in ``logs/``, newest first, with the verdict from
  the sibling ``.json`` summary and whether ``plots/<name>.png`` exists.
* ``/logs/<file>`` - one log or summary from ``logs/``.
* ``/plots/<file>`` - one figure from ``plots/``.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

SITE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SITE_ROOT.parent
LOG_DIR = REPO_ROOT / "logs"
PLOT_DIR = REPO_ROOT / "plots"

#: Only these suffixes are ever served from outside the site folder.
ALLOWED_SUFFIXES = {".npz", ".json", ".png"}


def list_runs(directory: Path, plot_directory: Path | None = None, limit: int = 200) -> list[dict]:
    """Describe the run logs in a directory, newest first.

    Args:
        directory: Directory to scan for ``.npz`` run logs.
        plot_directory: Directory to look for ``<stem>.png`` figures in. Defaults to ``plots/``.
        limit: Maximum number of logs to report.

    Returns:
        One entry per log with its name, size, modification time, the parsed summary next to it (or
        ``None``) and whether a figure exists for it.
    """
    if not directory.is_dir():
        return []
    figures = PLOT_DIR if plot_directory is None else plot_directory
    paths = sorted(directory.glob("*.npz"), key=lambda path: path.stat().st_mtime, reverse=True)
    runs = []
    for path in paths[:limit]:
        summary_path = path.with_suffix(".json")
        summary = None
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except (OSError, ValueError):
                summary = None
        stat = path.stat()
        runs.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "summary": summary,
                "plot": (figures / f"{path.stem}.png").is_file(),
            }
        )
    return runs


class ReplayHandler(SimpleHTTPRequestHandler):
    """Serve the site folder, plus the log, plot and listing routes above it."""

    #: Where the run logs are read from, and where their figures are looked for. Overridden by the
    #: command line so the site can be pointed at another run directory.
    log_dir: Path = LOG_DIR
    plot_dir: Path = PLOT_DIR

    def __init__(self, *args: Any, **kwargs: Any):
        """Serve from the site folder by default."""
        super().__init__(*args, directory=str(SITE_ROOT), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 (http.server names this method)
        """Route one GET request."""
        path = urlparse(self.path).path
        if path == "/api/logs":
            self._send_json({"runs": list_runs(self.log_dir, self.plot_dir)})
            return
        if path.startswith("/logs/"):
            self._send_external(self.log_dir, unquote(path[len("/logs/") :]))
            return
        if path.startswith("/plots/"):
            self._send_external(self.plot_dir, unquote(path[len("/plots/") :]))
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def _send_json(self, payload: object) -> None:
        """Write a JSON response that is never cached, so a fresh run shows up on reload."""
        body = json.dumps(payload).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_external(self, directory: Path, name: str) -> None:
        """Write one file from a directory outside the site folder, refusing anything else."""
        if not name or "/" in name or "\\" in name or name.startswith("."):
            self.send_error(HTTPStatus.FORBIDDEN, "invalid file name")
            return
        path = directory / name
        if path.suffix not in ALLOWED_SUFFIXES or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, f"no such file: {name}")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: Any) -> None:
        """Stay quiet: this is a viewer, not a service whose access log anyone reads."""
        return


def main() -> None:
    """Parse the arguments and serve until interrupted."""
    parser = argparse.ArgumentParser(
        description="Serve the debug-run replay site and the logs it reads."
    )
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="port to bind (default 8765)")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR,
        help=f"directory holding the .npz run logs (default {LOG_DIR})",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=PLOT_DIR,
        help=f"directory holding the .png figures (default {PLOT_DIR})",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open the page in a browser"
    )
    args = parser.parse_args()

    ReplayHandler.log_dir = args.log_dir.resolve()
    ReplayHandler.plot_dir = args.plot_dir.resolve()

    try:
        server = ThreadingHTTPServer((args.host, args.port), ReplayHandler)
    except OSError as error:
        sys.exit(f"cannot listen on {args.host}:{args.port} ({error}); try --port <other>")
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    count = len(list(ReplayHandler.log_dir.glob("*.npz")))
    print(f"replay site  {url}")
    print(f"site folder  {SITE_ROOT}")
    print(f"run logs     {ReplayHandler.log_dir}  ({count} .npz, newest first)")
    print(f"figures      {ReplayHandler.plot_dir}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
