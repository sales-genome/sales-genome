"""Small artifact I/O replacing private review-run dependencies."""
import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_paths(root):
    root = Path(root)
    for path in root.rglob("*"):
        if path.relative_to(root).parts[0] != ".git":
            yield path


repair = SimpleNamespace(source=SimpleNamespace(
    now=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
))
