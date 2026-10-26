"""Offline standalone/credential-signature audit and sealed-file verification."""
import argparse
import json
from pathlib import Path
import re

from paper_submission.artifacts import load, package_paths, save, sha256
from paper_submission.validate import standalone

EXCLUDED_OUTPUTS = {"provenance/package-files.json", "validation/package-audit.json"}
SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9]{30,}\b"),
]
PRIVATE_PATH = re.compile(rb"(?<![\w:/.-])/(?:mnt/localssd|home/[A-Za-z0-9_.-]+|root)(?:/|\b)")


def audit(root):
    root = Path(root)
    standalone(root)
    scanned, size = 0, 0
    for path in sorted(package_paths(root)):
        relative = str(path.relative_to(root))
        if not path.is_file() or relative in EXCLUDED_OUTPUTS:
            continue
        scanned += 1
        size += path.stat().st_size
        with path.open("rb") as stream:
            tail = b""
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                content = tail + chunk
                if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                    raise ValueError("Potential credential signature; inspect locally before release: " + relative)
                if PRIVATE_PATH.search(content):
                    raise ValueError("Private machine locator found in package: " + relative)
                tail = content[-512:]
    return {
        "status": "PASS_LOCAL_PACKAGE_AUDIT", "files_scanned": scanned,
        "bytes_scanned": size, "symlinks": 0, "full_databases_or_checkpoints": 0,
        "known_credential_signature_matches": 0, "private_machine_locator_matches": 0,
        "scope": "Finite offline pattern/structure checks, not a comprehensive secrets or source-support certification.",
        "release_approval_granted": False,
    }


def verify_registry(root):
    root = Path(root)
    registry = load(root / "provenance/package-files.json")
    actual = {
        str(path.relative_to(root)) for path in package_paths(root)
        if path.is_file() and path != root / "provenance/package-files.json"
    }
    if actual != set(registry["files"]):
        raise ValueError("Package file set differs from the seal (new or missing files).")
    for relative, row in registry["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe sealed-file path.")
        path = root / relative
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("Sealed file changed or missing: " + relative)
    return len(registry["files"])


def seal_registry(root):
    root = Path(root)
    files = {}
    for path in sorted(package_paths(root)):
        relative = str(path.relative_to(root))
        if path.is_file() and relative != "provenance/package-files.json":
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    save(root / "provenance/package-files.json", {
        "status": "LOCAL DRAFT FILE CHECKSUMS; not publication approval",
        "excludes": [
            "provenance/package-files.json (self-reference)",
            ".git (version-control metadata)",
        ],
        "files": files,
    })
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--seal", action="store_true",
                        help="Explicitly refresh the complete package file/checksum inventory.")
    args = parser.parse_args()
    report = audit(args.root)
    if args.report:
        save(args.report, report)
    if args.seal:
        seal_registry(args.root)
    if args.verify_checksums or args.seal:
        report["sealed_files_verified"] = verify_registry(args.root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
