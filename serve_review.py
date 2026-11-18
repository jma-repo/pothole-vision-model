"""Serve a folder on localhost so review.html also works in the Claude app's browser pane.

    python serve_review.py 8765 <folder holding both the results and the clips>

Python's own http.server ignores HTTP Range requests, so the review page's video could not seek; this one honours them.
Opening review.html directly in Chrome or Edge needs no server.
"""
import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    remaining: int | None = None

    def send_head(self):
        m = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start = int(m.group(1))
        end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        if self.remaining is None:
            return super().copyfile(source, outputfile)
        while self.remaining > 0 and (chunk := source.read(min(1 << 16, self.remaining))):
            outputfile.write(chunk)
            self.remaining -= len(chunk)


if __name__ == "__main__":
    port, root = int(sys.argv[1]), sys.argv[2]
    ThreadingHTTPServer(("127.0.0.1", port), partial(RangeHandler, directory=root)).serve_forever()
