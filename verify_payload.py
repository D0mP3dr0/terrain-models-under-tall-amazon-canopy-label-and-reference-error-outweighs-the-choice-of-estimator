"""Check that the published JSON artifacts carry the same numbers as the originals.

The artifacts under audit/ were published with their free-text notes translated into
English and internal-note paths renamed. Everything else is untouched. This script
computes, for each JSON file, the sha256 of its "payload": the canonical JSON with every
free-text string replaced by a fixed placeholder. A string counts as free text when it
has four or more words, names a .md note, or is an absolute path into the authors'
document library; fact ids and artifact names (keys "id" and "artefato") never count as
free text. Numbers, booleans, nulls, keys, structure, short labels, sha256 values, dotted
field pointers, fact ids and artifact names all enter the digest.

PAYLOAD_SHA256 lists the digest of each ORIGINAL artifact, computed before publication.
A match here shows that translation changed prose only.

Usage (from the repository root):  python verify_payload.py
"""
import hashlib
import json
import pathlib
import sys

PROTECTED = {"artefato", "id"}
PLACEHOLDER = "<text>"


def is_free_text(s, key):
    if key in PROTECTED:
        return False
    return len(s.split()) >= 4 or ".md" in s or s.startswith("/arquivo/")


def mask(o, key=None):
    if isinstance(o, dict):
        return {k: mask(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [mask(v, key) for v in o]
    if isinstance(o, str) and is_free_text(o, key):
        return PLACEHOLDER
    return o


def payload_sha256(path):
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    canon = json.dumps(mask(data), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def main():
    root = pathlib.Path(__file__).resolve().parent
    expected = {}
    for line in (root / "PAYLOAD_SHA256").read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            expected[rel.strip()] = digest
    bad = 0
    for rel, digest in sorted(expected.items()):
        got = payload_sha256(root / rel)
        ok = got == digest
        bad += not ok
        if not ok:
            print(f"MISMATCH {rel}")
    print(f"{len(expected) - bad}/{len(expected)} artifacts: payload identical to the original")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
