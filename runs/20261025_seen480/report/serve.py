"""Serve runs/20261025_seen480 so report/index.html can play the rollout videos.

Python's http.server ignores Range requests, and browsers need them to seek inside an mp4 (these videos keep their
index at the end of the file), so this adds byte-range support.
usage: python report/serve.py [port]   then open http://localhost:<port>/report/  (VS Code forwards the port)"""
import os, re, sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RangeHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        self._left = None
        path = self.translate_path(self.path)
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        if m.group(1):
            start, end = int(m.group(1)), int(m.group(2)) if m.group(2) else size - 1
        else:
            start, end = max(0, size - int(m.group(2) or 0)), size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._left = end - start + 1
        return f

    def copyfile(self, src, dst):
        if self._left is None:
            return super().copyfile(src, dst)
        while self._left > 0:
            buf = src.read(min(1 << 16, self._left))
            if not buf:
                break
            dst.write(buf)
            self._left -= len(buf)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = ThreadingHTTPServer(("127.0.0.1", port), partial(RangeHandler, directory=ROOT))
    print(f"serving {ROOT} on http://localhost:{port}/report/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
