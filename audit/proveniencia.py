"""PROVENANCE STAMP -- closes the chain from article to verdict artifact to run.

THE BROKEN LINK. The manuscript cites verdict artifacts via the macro
`\\prov{...}`, and the canonical registry can say whether a RUN is
canonical. But the verdict artifacts in the middle of the chain did not
declare what they had read: a reviewer tracing the origin of a stated
`t=6.13` would reach `comparar_regimes_v23.json` and stop there. Of the six
most-cited verdict artifacts, ZERO declared their sources.

This module gives every audit script two lines to close that gap:

    import proveniencia as PROV
    ...
    PROV.gravar(SAIDA, dados, fontes=[ARQ1, ARQ2], script=__file__)

and the resulting JSON gains two blocks:

    "_proveniencia": script, git commit, dirty tree, host, date, versions
    "_fontes":       for every file read: path, sha256, bytes, mtime

The sha256 is of the file AT THE MOMENT IT WAS READ. If the input changes
afterwards, the record detects the divergence instead of letting the number
silently go stale -- which is exactly what happened when an earlier LiDAR
reference was invalidated and the verdict artifacts built on it kept
looking valid.

Usage: see `gravar` and `carimbar_existente`.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent


def _git(*args) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(RAIZ), *args],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None


def _versoes() -> dict:
    v = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "scipy", "torch", "torch_geometric",
                "xgboost"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            pass
    return v


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def bloco(script: str | None = None) -> dict:
    """The stamp of what produced the artifact, when, and with what code state."""
    sp = Path(script) if script else None
    return {
        "script": sp.name if sp else None,
        # sha256 of the script ITSELF: with a dirty working tree or an
        # untracked script, the commit hash alone does not identify the code
        "sha256_script": sha256(sp) if (sp and sp.exists()) else None,
        "commit": _git("rev-parse", "--short", "HEAD"),
        "tag": _git("describe", "--tags", "--abbrev=0"),
        "arvore_suja": bool(_git("status", "--porcelain")),
        "host": platform.node(),
        "em": time.strftime("%Y-%m-%dT%H:%M"),
        "versoes": _versoes(),
    }


def fontes(caminhos) -> list:
    """Identity of every input AT THE MOMENT it is read."""
    out = []
    for c in caminhos:
        p = Path(c)
        if not p.is_absolute():
            p = RAIZ / p
        if not p.exists():
            out.append({"caminho": str(c), "estado": "AUSENTE"})
            continue
        out.append({
            "caminho": str(p.relative_to(RAIZ)).replace("\\", "/")
            if str(p).startswith(str(RAIZ)) else str(p),
            "sha256": sha256(p), "bytes": p.stat().st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M",
                                   time.localtime(p.stat().st_mtime))})
    return out


def gravar(saida, dados: dict, fontes_lidas=(), script: str | None = None):
    """Writes the audit JSON ALREADY stamped. Replaces a raw `write_text`."""
    p = Path(saida)
    corpo = {"_proveniencia": bloco(script), "_fontes": fontes(fontes_lidas),
             **dados}
    p.write_text(json.dumps(corpo, indent=2, ensure_ascii=False, default=float),
                 encoding="utf-8")
    return p


def carimbar_existente(alvo, fontes_lidas=(), script: str | None = None,
                       nota: str | None = None) -> bool:
    """Adds the stamp to a JSON that ALREADY exists, without touching its content.

    Used for artifacts produced before this module existed -- and for runs
    still in progress, which are stamped AFTER they finish, so as not to
    write to a file training still has open.
    """
    p = Path(alvo)
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(d, dict):
        return False
    b = bloco(script)
    b["retroativo"] = True
    # IN A RETROACTIVE STAMP THE SCRIPT SHA IS TODAY'S, NOT THE ONE FROM WHEN
    # THE ARTIFACT WAS PRODUCED. `bloco()` hashes the script as it sits on
    # disk NOW; the artifact may have been produced by an earlier version
    # (a measured case showed a difference of hundreds of changed lines
    # between the script version used to generate the artifact and the one
    # present at stamping time). The key is renamed so it never asserts an
    # unverified identity.
    b["sha256_script_no_momento_do_carimbo"] = b.pop("sha256_script")
    if nota:
        b["nota"] = nota
    # RE-STAMPING MUST OVERWRITE. An earlier version of this function did
    # `{"_proveniencia": b, "_fontes": ..., **d}` -- and if `d` already had
    # those keys, `**d` put the OLD ones back on top of the new ones. The
    # re-stamp was a silent no-op, which defeated the module's stated
    # purpose (detecting a changed input): the old sha256 would stay
    # forever.
    anterior = d.pop("_proveniencia", None)
    d.pop("_fontes", None)
    if anterior:
        b["carimbo_anterior"] = {k: anterior.get(k)
                                 for k in ("script", "commit", "em")}
    d = {"_proveniencia": b, "_fontes": fontes(fontes_lidas), **d}
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=float),
                 encoding="utf-8")
    return True
